"""Environment-selected Seedance gateway: protocol, admission, ledger and pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cli
from app.config import config
from app.models.schema import VideoParams
from app.services import material, storyboard_timing, video
from app.services import task as tm
from app.services import video_run_config as vrc
from app.services import volcengine_seedance as seedance
from app.services import wan3
from app.services import wan3_coordinator as coordinator
from app.services.state import MemoryState
from app.services.storyboard import Storyboard, StoryboardShot
from app.utils import utils

MODEL = "Doubao-Seedance-2-0-Fast-260128-Huoshan-New"
ENV = {
    "MPT_SEEDANCE_API_KEY": "gateway-secret",
    "MPT_SEEDANCE_BASE_URL": "https://gw.invalid",
    "MPT_SEEDANCE_MODEL_NAME": MODEL,
}
TASKS_URL = "https://gw.invalid/api/v3/contents/generations/tasks"
REAL_POST, REAL_GET = requests.post, requests.get


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    monkeypatch.setattr(seedance, "RETRY_BASE_SECONDS", 0.01)
    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: pytest.fail("unexpected POST"))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: pytest.fail("unexpected GET"))


def _config(**overrides):
    resolved = vrc.resolve_video_run_config({**ENV, **overrides})
    assert isinstance(resolved, vrc.VideoRunConfig), resolved
    return resolved.with_runtime(proxies={"https": "http://u:proxy-secret@proxy.invalid"},
                                 tls_verify=False, poll_interval=0.01, run_timeout=5)


def _response(body=None, status=200, headers=None, unreadable=False):
    def payload():
        if unreadable:
            raise ValueError("not json")
        return body
    return SimpleNamespace(status_code=status, json=payload, headers=headers or {})


def _succeeded(task_id="cgt-1"):
    return _response({"id": task_id, "status": "succeeded",
                      "content": {"video_url": f"https://media.invalid/{task_id}.mp4"}})


def _records(tmp_path):
    return [json.loads(path.read_text()) for path in sorted(tmp_path.glob(".agent-logs/seedance-*/slot-*.json"))]


def _clip(run_config, **kwargs):
    return seedance.generate_coordinated_clip(
        "Wide: 海边日落", kwargs.pop("duration", 4), "9:16",
        endpoint=seedance.endpoint_from_run_config(run_config),
        identity=run_config.credential_fingerprint(), **kwargs,
    )


# --- TokenHub protocol ----------------------------------------------------------

def test_success_sends_platform_model_verbatim_with_snapshot_network(monkeypatch, tmp_path):
    run_config = _config()
    calls, recorded = [], []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _response({"id": "cgt-1"})

    def record(remote_id):
        recorded.append(remote_id)
        assert [r["phase"] for r in _records(tmp_path) if r] == ["submitted"]

    monkeypatch.setattr(seedance.requests, "post", post)
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: _succeeded())
    item = _clip(run_config, on_task_created=record)
    url, kwargs = calls[0]
    assert url == TASKS_URL
    assert kwargs["json"] == {
        "model": MODEL, "content": [{"type": "text", "text": "Wide: 海边日落"}],
        "ratio": "9:16", "duration": 4, "resolution": "720p", "watermark": False,
        "generate_audio": False,
    }
    assert kwargs["headers"]["Authorization"] == "Bearer gateway-secret"
    assert kwargs["proxies"] == {"https": "http://u:proxy-secret@proxy.invalid"}
    assert kwargs["verify"] is False and kwargs["allow_redirects"] is False
    assert recorded == ["cgt-1"] and item.source_info["asset_id"] == "cgt-1"
    assert _records(tmp_path) == [None]


def test_models_without_audio_switch_do_not_receive_it(monkeypatch):
    run_config = _config(MPT_SEEDANCE_MODEL_NAME="doubao-seedance-1-0-pro-250528")
    payloads = []
    monkeypatch.setattr(seedance.requests, "post",
                        lambda *_a, **k: payloads.append(k["json"]) or _response({"id": "cgt-1"}))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: _succeeded())
    _clip(run_config, duration=2)
    assert "generate_audio" not in payloads[0] and payloads[0]["resolution"] == "1080p"


@pytest.mark.parametrize("response, unconfirmed", [
    (requests.exceptions.ReadTimeout("timeout gateway-secret"), True),
    (_response({}, 502), True),
    (_response({}, 302), True),
    (_response({"error": {"code": "UpstreamTimeout", "type": "InternalServerError"}}, 422), True),
    (_response({"error": {"code": "Custom", "type": "InternalServerError"}}, 422), True),
    (_response({}, 200), True),
    (_response(None, 200, unreadable=True), True),
    (_response({"error": {"code": "InvalidParameter.TaskTypeConstraint", "type": "BadRequest"}}, 422), False),
    (_response({"error": {"code": "InvalidParameter", "type": "BadRequest"}}, 400), False),
    (_response({"error": {"code": "AuthenticationError", "type": "Unauthorized"}}, 401), False),
    (_response({"error": {"code": "InvalidEndpointOrModel.NotFound", "type": "NotFound"}}, 404), False),
    (_response({"error": {"code": 429202, "message": "User RPM limit"}}, 429,
               {"x-ratelimit-reset": "3"}), False),
])
def test_submission_is_never_retried_and_only_refusals_free_the_slot(
        monkeypatch, tmp_path, response, unconfirmed):
    posts = []

    def post(*_a, **_k):
        posts.append(1)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(seedance.requests, "post", post)
    expected = seedance.VolcEngineSeedanceUnconfirmedTaskError if unconfirmed else seedance.VolcEngineSeedanceError
    with pytest.raises(expected) as caught:
        _clip(_config())
    assert len(posts) == 1
    assert unconfirmed == isinstance(caught.value, seedance.VolcEngineSeedanceUnconfirmedTaskError)
    records = _records(tmp_path)
    if unconfirmed:
        assert records[0]["phase"] == "submitting"
    else:
        assert records == [None]
    message = str(caught.value)
    assert "gateway-secret" not in message and "proxy-secret" not in message
    if getattr(response, "status_code", 0) == 429:
        assert "429202" in message and "MPT_SEEDANCE_CONCURRENCY" in message


def test_polling_retries_only_the_same_task_with_rate_limit_reset(monkeypatch, tmp_path):
    sequence = [
        _response({"error": {"code": "InternalError", "type": "InternalServerError"}}, 422),
        _response({"error": {"code": 429101, "message": "concurrency"}}, 429, {"x-ratelimit-reset": "0.25"}),
        _response({"error": {"code": "RateLimit"}}, 429, {"x-ratelimit-reset": "soon"}),
        _response({}, 503),
        _response({"id": "cgt-1", "status": "running"}),
        _succeeded(),
    ]
    urls, delays = [], []
    real_sleep = time.sleep
    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: _response({"id": "cgt-1"}))
    monkeypatch.setattr(seedance.requests, "get", lambda url, **_k: urls.append(url) or sequence.pop(0))
    monkeypatch.setattr(seedance.time, "sleep", lambda seconds: delays.append(round(seconds, 3)) or real_sleep(0))
    _clip(_config())
    assert urls == [f"{TASKS_URL}/cgt-1"] * 6
    # Retry delays: backoff, the valid reset header, then backoff for the invalid one.
    assert delays[-5:-1] == [0.01, 0.25, 0.03, 0.04]
    assert _records(tmp_path) == [None]


def test_rate_limit_wait_is_bounded_by_the_total_deadline(monkeypatch, tmp_path):
    run_config = vrc.resolve_video_run_config(ENV).with_runtime(
        proxies={}, tls_verify=True, poll_interval=0.01, run_timeout=0.3)
    delays = []
    real_sleep = time.sleep
    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: _response({"id": "cgt-1"}))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: _response(
        {"error": {"code": 429201}}, 429, {"x-ratelimit-reset": "100"}))
    monkeypatch.setattr(seedance.time, "sleep", lambda s: delays.append(s) or real_sleep(s))
    with pytest.raises(seedance.VolcEngineSeedanceUnconfirmedTaskError) as caught:
        _clip(run_config)
    assert caught.value.task_id == "cgt-1" and max(delays) <= 0.3 + 1e-3
    record = _records(tmp_path)[0]
    assert (record["phase"], record["task_id"]) == ("submitted", "cgt-1")


@pytest.mark.parametrize("response", [
    _response({"error": {"code": "ResourceNotFound", "message": "video task not found"}}, 404),
    _response({"error": {"code": "InvalidParameter", "type": "BadRequest"}}, 422),
    _response({"error": {"code": "AuthenticationError"}}, 401),
])
def test_unknown_query_results_keep_id_and_slot(monkeypatch, tmp_path, response):
    posts = []
    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: posts.append(1) or _response({"id": "cgt-9"}))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: response)
    with pytest.raises(seedance.VolcEngineSeedanceUnconfirmedTaskError) as caught:
        _clip(_config())
    assert caught.value.task_id == "cgt-9" and posts == [1]
    record = _records(tmp_path)[0]
    assert (record["phase"], record["task_id"]) == ("submitted", "cgt-9")
    assert "gateway-secret" not in json.dumps(record)


def test_confirmed_remote_failure_frees_the_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: _response({"id": "cgt-2"}))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: _response(
        {"status": "failed", "error": {"code": "OutputVideoSensitiveContentDetected"}}))
    with pytest.raises(seedance.VolcEngineSeedanceError) as caught:
        _clip(_config())
    assert not isinstance(caught.value, seedance.VolcEngineSeedanceUnconfirmedTaskError)
    assert caught.value.task_id == "cgt-2" and _records(tmp_path) == [None]


# --- admission identity -------------------------------------------------------------

def _fill(run_config, count=coordinator.CAPACITY):
    namespace = f"seedance-{seedance._endpoint_namespace(run_config.endpoint)}"
    slots = [coordinator.acquire_slot(base_url=run_config.endpoint, probe=lambda _id: False,
                                      namespace=namespace, identity=run_config.credential_fingerprint(),
                                      label="Seedance") for _ in range(count)]
    for index, slot in enumerate(slots):
        slot.save("submitted", task_id=f"old-{index}")
        slot.close()


def test_rotated_key_never_queries_or_clears_old_records(monkeypatch, tmp_path):
    _fill(_config())
    before = _records(tmp_path)
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: pytest.fail("old ID queried with new key"))
    with pytest.raises(seedance.VolcEngineSeedanceUnconfirmedTaskError, match="cannot be safely reclaimed"):
        _clip(_config(MPT_SEEDANCE_API_KEY="rotated-key"))
    assert _records(tmp_path) == before
    assert all("gateway-secret" not in json.dumps(record) for record in before)


def test_same_identity_recovers_only_confirmed_terminal_records(monkeypatch, tmp_path):
    run_config = _config()
    _fill(run_config)
    queried = []

    def get(url, **_kwargs):
        queried.append(url.rsplit("/", 1)[-1])
        if url.endswith("/old-0"):
            return _response({"status": "succeeded"})
        if url.endswith("/new"):
            return _succeeded("new")
        return _response({"status": "running"})

    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: _response({"id": "new"}))
    monkeypatch.setattr(seedance.requests, "get", get)
    _clip(run_config)
    assert queried[0] == "old-0" and queried[-1] == "new"
    assert sum(bool(record) for record in _records(tmp_path)) == 4


def test_endpoints_and_providers_use_separate_slots(monkeypatch, tmp_path):
    _fill(_config())
    other = _config(MPT_SEEDANCE_BASE_URL="https://other.invalid/v1")
    monkeypatch.setattr(seedance.requests, "post", lambda url, **_k: _response({"id": "cgt-other"}))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: _succeeded("cgt-other"))
    _clip(other)
    assert len(list(tmp_path.glob(".agent-logs/seedance-*"))) == 2
    assert not (tmp_path / ".agent-logs/wan3").exists()


# --- storyboard scheduling and ledger ---------------------------------------------

def _plans(*seconds):
    return [SimpleNamespace(index=i + 1, prompt=f"prompt-{i}", request_seconds=value,
                            required_source_seconds=value - 0.5, target_playback_seconds=value - 0.5,
                            remote_task_id=None, material_seconds=None, material_path=None)
            for i, value in enumerate(seconds)]


def _storyboard(run_config, plans, **kwargs):
    return material._download_storyboard_seedance(
        task_id="local", shot_plan=plans, video_aspect="9:16", material_directory="",
        video_run_config=run_config, **kwargs,
    )


def _ledger():
    return json.loads((Path(utils.task_dir("local")) / material.SEEDANCE_LEDGER_NAME).read_text())


@pytest.mark.parametrize("seconds", [(4, 3), (4, 16)])
def test_every_shot_is_validated_before_the_first_submission(monkeypatch, seconds):
    monkeypatch.setattr(seedance, "generate_coordinated_clip", lambda *_a, **_k: pytest.fail("submitted"))
    with pytest.raises(seedance.VolcEngineSeedanceError, match="from 4 to 15 seconds"):
        _storyboard(_config(), _plans(*seconds))


def test_storyboard_requires_the_frozen_configuration():
    with pytest.raises(ValueError, match="--video-config-from-env"):
        material.storyboard_generation_options("volcengine_seedance")
    assert material.storyboard_generation_options("volcengine_seedance", _config()) == {
        "minimum_clip_seconds": 4, "maximum_clip_seconds": 15, "provider_name": "Seedance兼容网关",
    }


def _fake_media(monkeypatch, durations):
    monkeypatch.setattr(material, "_save_generated_video_with_retry",
                        lambda url, *_a, **_k: url.rsplit("/", 1)[-1] if "broken" not in url else "")
    monkeypatch.setattr(material, "_measure_generated_video",
                        lambda path: (durations.get(path, 5.0), 720, 1280))
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)


def test_window_order_counts_and_ledger_with_out_of_order_completion(monkeypatch):
    run_config = _config(MPT_SEEDANCE_CONCURRENCY="2")
    started = [threading.Event() for _ in range(4)]
    finish = [threading.Event() for _ in range(4)]
    counts = []

    def clip(prompt, seconds, aspect, *, endpoint, identity, on_task_created, stop_event):
        index = int(prompt.split("-")[-1])
        assert (seconds, endpoint.model, identity) == (5, MODEL, run_config.credential_fingerprint())
        on_task_created(f"cgt-{index}")
        started[index].set()
        assert finish[index].wait(5)
        return seedance._material(_succeeded(f"cgt-{index}").json(), f"cgt-{index}",
                                  "volcengine_seedance", prompt, seconds, {"id": f"cgt-{index}"})

    _fake_media(monkeypatch, {})
    monkeypatch.setattr(seedance, "generate_coordinated_clip", clip)
    plans = _plans(5, 5, 5, 5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_storyboard, run_config, plans,
                                 on_clip_completed=lambda *a: counts.append(a))
        try:
            assert started[0].wait(3) and started[1].wait(3)
            assert not started[2].is_set()
            finish[1].set()
            assert started[2].wait(3) and not started[3].is_set()
            for index in (3, 2, 0):
                finish[index].set()
            assert future.result(timeout=5) == [f"cgt-{i}.mp4" for i in range(4)]
        finally:
            for event in finish:
                event.set()
    assert counts == [(i, 4) for i in range(5)]
    assert [plan.remote_task_id for plan in plans] == [f"cgt-{i}" for i in range(4)]
    ledger = _ledger()
    assert sorted(ledger["task_ids"]) == [f"cgt-{i}" for i in range(4)]
    assert {shot["status"] for shot in ledger["shots"].values()} == {"downloaded"}
    assert ledger["shots"]["2"]["request_seconds"] == 5 and ledger["shots"]["2"]["model"] == MODEL
    assert "gateway-secret" not in json.dumps(ledger)


@pytest.mark.parametrize("failure", ["short", "download", "unconfirmed"])
def test_first_failure_stops_dispatch_and_keeps_in_flight_ids(monkeypatch, failure):
    run_config = _config(MPT_SEEDANCE_CONCURRENCY="2")
    barrier = threading.Barrier(2)
    calls = []

    def clip(prompt, seconds, aspect, *, endpoint, identity, on_task_created, stop_event):
        index = int(prompt.split("-")[-1])
        calls.append(index)
        on_task_created(f"cgt-{index}")
        barrier.wait(3)
        if index == 0 and failure == "unconfirmed":
            raise seedance.VolcEngineSeedanceUnconfirmedTaskError("still running", task_id="cgt-0")
        if index == 1:
            assert stop_event.wait(3)
        url_id = "broken" if index == 0 and failure == "download" else f"cgt-{index}"
        return seedance._material(_succeeded(url_id).json(), f"cgt-{index}",
                                  "volcengine_seedance", prompt, seconds, {"id": f"cgt-{index}"})

    _fake_media(monkeypatch, {"cgt-0.mp4": 1.0 if failure == "short" else 5.0})
    monkeypatch.setattr(seedance, "generate_coordinated_clip", clip)
    plans = _plans(5, 5, 5, 5)
    with pytest.raises(seedance.VolcEngineSeedanceError) as caught:
        _storyboard(run_config, plans)
    assert sorted(calls) == [0, 1]
    assert caught.value.task_id == "cgt-0" and "storyboard shot 1" in str(caught.value)
    assert sorted(_ledger()["task_ids"]) == ["cgt-0", "cgt-1"]
    assert plans[1].remote_task_id == "cgt-1" and plans[1].material_path == "cgt-1.mp4"


def test_ledger_write_failure_stops_new_submissions_and_returns_the_id(monkeypatch, tmp_path):
    original = coordinator.atomic_json
    posts = []

    def write(target, data):
        if target.name == material.SEEDANCE_LEDGER_NAME:
            raise PermissionError("ledger unwritable")
        original(target, data)

    monkeypatch.setattr(coordinator, "atomic_json", write)
    monkeypatch.setattr(seedance.requests, "post", lambda *_a, **_k: posts.append(1) or _response({"id": "cgt-0"}))
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)
    with pytest.raises(seedance.VolcEngineSeedanceUnconfirmedTaskError) as caught:
        _storyboard(_config(MPT_SEEDANCE_CONCURRENCY="1"), _plans(5, 5, 5))
    assert caught.value.task_id == "cgt-0" and posts == [1]
    assert _records(tmp_path)[0]["task_id"] == "cgt-0"


def test_parallel_ledger_updates_do_not_lose_shots():
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: material._update_seedance_ledger(
            "local", i, shot=i, remote_id=f"cgt-{i}", status="submitted"), range(24)))
    ledger = _ledger()
    assert len(ledger["shots"]) == 24 and len(ledger["task_ids"]) == 24


# --- the frozen snapshot is used everywhere ---------------------------------------

def test_later_environment_and_config_changes_do_not_affect_the_task(monkeypatch):
    run_config = _config()
    monkeypatch.setenv("MPT_SEEDANCE_API_KEY", "changed")
    monkeypatch.setenv("MPT_SEEDANCE_MODEL_NAME", "changed-model")
    monkeypatch.setenv("VOLCENGINE_ARK_API_KEY", "changed")
    monkeypatch.setitem(config.app, "volcengine_seedance_model", "config-model")
    monkeypatch.setitem(config.app, "volcengine_seedance_base_url", "https://config.invalid")
    monkeypatch.setitem(config.app, "volcengine_seedance_api_key", "config-key")
    monkeypatch.setattr(config, "proxy", {"https": "http://global-proxy.invalid"})
    seen = []
    monkeypatch.setattr(seedance.requests, "post", lambda url, **k: seen.append(
        (url, k["json"]["model"], k["headers"]["Authorization"], k["proxies"])) or _response({"id": "cgt-0"}))
    monkeypatch.setattr(seedance.requests, "get", lambda *_a, **_k: _succeeded("cgt-0"))
    _fake_media(monkeypatch, {})
    _storyboard(run_config, _plans(5))
    assert seen == [(TASKS_URL, MODEL, "Bearer gateway-secret",
                     {"https": "http://u:proxy-secret@proxy.invalid"})]


def test_wan3_snapshot_ignores_later_environment_and_config(monkeypatch):
    resolved = vrc.resolve_video_run_config({
        "MPT_WAN3_API_KEY": "wan3-key", "MPT_WAN3_BASE_URL": "https://ws.invalid/api/v1",
        "MPT_WAN3_MODEL_NAME": "wan3.0-video", "MPT_WAN3_RESOLUTION": "720p",
    }).with_runtime(proxies={}, tls_verify=True, poll_interval=0.01, run_timeout=5)
    monkeypatch.setenv("MPT_WAN3_API_KEY", "changed")
    monkeypatch.setenv("MPT_WAN3_BASE_URL", "https://changed.invalid")
    monkeypatch.setitem(config.app, "wan3_resolution", "480P")
    monkeypatch.setitem(config.app, "wan3_model_name", "config-model")
    monkeypatch.setattr(wan3, "validate_config", lambda *_a: pytest.fail("global config was read"))
    seen = []
    monkeypatch.setattr(wan3.requests, "post", lambda url, **k: seen.append(
        (url, k["json"], k["headers"]["Authorization"], k["proxies"])) or _response({"output": {"task_id": "w1"}}))
    monkeypatch.setattr(wan3.requests, "get", lambda url, **k: _response(
        {"output": {"task_status": "SUCCEEDED", "video_url": "https://media.invalid/w1.mp4"}}))
    wan3.generate_videos("prompt", 5, "9:16", run_config=resolved)
    url, payload, auth, proxies = seen[0]
    assert url == "https://ws.invalid/api/v1/services/aigc/video-generation/video-synthesis"
    assert (payload["model"], payload["parameters"]["resolution"], auth, proxies) == (
        "wan3.0-video", "720P", "Bearer wan3-key", {})


@pytest.mark.parametrize("failure", [False, True])
def test_download_keeps_snapshot_network_and_redacts_proxy_errors(monkeypatch, tmp_path, failure):
    snapshot = _config()
    proxy = snapshot.proxy_map()
    calls, messages = [], []

    class Response:
        headers = {"Content-Length": "5"}

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, **_kwargs):
            yield b"video"

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if failure:
            raise requests.exceptions.ProxyError(f"connection failed: {proxy['https']}")
        return Response()

    monkeypatch.setattr(material.requests, "get", get)
    monkeypatch.setattr(config, "proxy", {"https": "http://changed.invalid"})
    monkeypatch.setattr(material, "_get_tls_verify", lambda: pytest.fail("global TLS setting was read"))
    monkeypatch.setattr(material, "_is_valid_video_file", lambda _path: True)
    monkeypatch.setattr(material.time, "sleep", lambda _seconds: None)
    sink = material.logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        result = material._save_generated_video_with_retry(
            "https://media.invalid/paid.mp4?signature=private", str(tmp_path), "volcengine_seedance",
            network_options={"proxies": proxy, "tls_verify": False},
        )
    finally:
        material.logger.remove(sink)
    assert bool(result) is not failure
    assert all(kwargs["proxies"] == proxy and kwargs["verify"] is False for _url, kwargs in calls)
    assert "proxy-secret" not in "".join(messages) and "signature=private" not in "".join(messages)


# --- task and CLI ----------------------------------------------------------------

def _pcm(path, seconds):
    storyboard_timing.write_silence(seconds, path)


def _board(*lines):
    shots = tuple(StoryboardShot(i + 1, f"Wide: 画面{i}", line, i * 3.0, i * 3.0 + 3.0)
                  for i, line in enumerate(lines))
    return Storyboard("".join(lines), [shot.prompt for shot in shots], shots=shots)


def _params(**overrides):
    values = dict(video_subject="示例", video_script="第一句。", video_source="volcengine_seedance",
                  voice_name="zh-CN-XiaoxiaoNeural-Female", voice_rate=1.0, subtitle_enabled=False,
                  bgm_volume=0, video_clip_duration=5, video_concat_mode="sequential",
                  match_materials_to_script=True)
    values.update(overrides)
    return VideoParams(**values)


def test_seedance_storyboard_without_snapshot_fails_preflight_without_tts(monkeypatch):
    monkeypatch.setattr(tm.sm, "state", MemoryState())
    monkeypatch.setattr(tm.voice, "tts", lambda **_k: pytest.fail("tts started"))
    monkeypatch.setattr(tm.utils, "check_ffmpeg_ready", lambda: True)
    result = tm.start("t-pre", _params(), storyboard_data=_board("第一句。"))
    assert result["failed_stage"] == "preflight" and "--video-config-from-env" in result["error"]


@pytest.mark.parametrize("narration_seconds, explicit, expected", [
    (1.2, None, 4), (6.3, None, 7), (2.0, 3, "at least 4"), (16.0, None, "single-clip limit"),
    (5.0, 4, "Raise that value"),
])
def test_timing_uses_model_limits_before_any_submission(monkeypatch, narration_seconds, explicit, expected):
    monkeypatch.setattr(tm.sm, "state", MemoryState())
    monkeypatch.setattr(tm.utils, "check_ffmpeg_ready", lambda: True)
    monkeypatch.setattr(tm.voice, "tts", lambda **k: _pcm(k["voice_file"], narration_seconds) or SimpleNamespace())
    submitted = []
    monkeypatch.setattr(material, "_download_storyboard_seedance",
                        lambda **k: submitted.append([p.request_seconds for p in k["shot_plan"]]) or ["a.mp4"])
    result = tm.start("t-plan", _params(), stop_at="materials", storyboard_data=_board("第一句。"),
                      storyboard_clip_duration=explicit, video_run_config=_config())
    if isinstance(expected, int):
        assert submitted == [[expected]] and result == {"materials": ["a.mp4"]}
    else:
        assert submitted == [] and result["failed_stage"] == "materials"
        assert expected in result["error"] and "本次尚未提交" in result["error"]


def test_task_failure_keeps_every_seedance_id(monkeypatch):
    state = MemoryState()
    monkeypatch.setattr(tm.sm, "state", state)
    monkeypatch.setattr(tm.utils, "check_ffmpeg_ready", lambda: True)
    monkeypatch.setattr(tm.voice, "tts", lambda **k: _pcm(k["voice_file"], 2.0) or SimpleNamespace())
    _fake_media(monkeypatch, {"cgt-1.mp4": 0.5})

    def clip(prompt, seconds, aspect, *, endpoint, identity, on_task_created, stop_event):
        index = int(prompt[-1])
        on_task_created(f"cgt-{index}")
        return seedance._material(_succeeded(f"cgt-{index}").json(), f"cgt-{index}",
                                  "volcengine_seedance", prompt, seconds, {"id": f"cgt-{index}"})

    monkeypatch.setattr(seedance, "generate_coordinated_clip", clip)
    result = tm.start("t-fail", _params(), stop_at="materials", storyboard_data=_board("一。", "二。"),
                      video_run_config=_config(MPT_SEEDANCE_CONCURRENCY="1"))
    assert result["failed_stage"] == "materials" and "too short" in result["error"]
    assert result["volcengine_seedance_task_id"] == "cgt-1"
    assert result["volcengine_seedance_task_ids"] == ["cgt-0", "cgt-1"]
    assert "gateway-secret" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("argv, message", [
    (["--video-subject", "x", "--video-config-from-env"], "explicit --video-source"),
    (["--video-subject", "x", "--video-source", "pexels", "--video-config-from-env"], "supports --video-source"),
    (["--batch-file", "t.json", "--video-source", "wan3", "--video-config-from-env"], "--batch-file"),
])
def test_cli_environment_mode_argument_rules(capsys, argv, message):
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(argv)
    assert caught.value.code == 2 and message in capsys.readouterr().err


def test_cli_freezes_one_configuration_and_never_reads_config_keys(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(config.app, "volcengine_seedance_api_key", "config-key")
    monkeypatch.setattr(config, "proxy", {"http": "http://proxy.invalid"})
    monkeypatch.setattr(cli, "build_video_params", lambda _args: object())
    monkeypatch.setattr(cli, "prepare_cli_files", lambda *_a, **_k: None)
    captured = {}

    def start(task_id, **kwargs):
        captured.update(kwargs)
        os.environ["MPT_SEEDANCE_MODEL_NAME"] = "changed-during-task"
        return {"videos": ["final.mp4"]}

    monkeypatch.setattr(tm, "start", start)
    assert cli.run_cli(["--video-subject", "x", "--video-source", "volcengine_seedance",
                        "--video-config-from-env", "--confirm-seedance-charge"]) == 0
    frozen = captured["video_run_config"]
    assert (frozen.model, frozen.api_key, frozen.endpoint) == (MODEL, "gateway-secret", TASKS_URL)
    assert frozen.proxy_map() == {"http": "http://proxy.invalid"}
    assert frozen.poll_interval == seedance.DEFAULT_POLL_INTERVAL_SECONDS


def test_cli_without_environment_mode_keeps_the_old_call(monkeypatch):
    monkeypatch.setattr(cli, "build_video_params", lambda _args: object())
    monkeypatch.setattr(cli, "prepare_cli_files", lambda *_a, **_k: None)
    captured = {}
    monkeypatch.setattr(tm, "start", lambda task_id, **kwargs: captured.update(kwargs) or {"videos": []})
    assert cli.run_cli(["--video-subject", "x"]) == 0
    assert "video_run_config" not in captured


def test_cli_incomplete_environment_is_an_input_error(monkeypatch, capsys):
    monkeypatch.setenv("MPT_SEEDANCE_API_KEY", "secret-value")
    for name in ("MPT_SEEDANCE_BASE_URL", "MPT_SEEDANCE_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "build_video_params", lambda _args: object())
    monkeypatch.setattr(cli, "prepare_cli_files", lambda *_a, **_k: None)
    monkeypatch.setattr(tm, "start", lambda *_a, **_k: pytest.fail("started"))
    assert cli.run_cli(["--video-subject", "x", "--video-source", "volcengine_seedance",
                        "--video-config-from-env", "--confirm-seedance-charge"]) == 2
    assert "secret-value" not in capsys.readouterr().err


# --- local HTTP end to end ---------------------------------------------------------

@pytest.fixture
def gateway(tmp_path):
    ffmpeg = utils.get_ffmpeg_binary()
    media = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc=s=96x160:d=5:r=24",
                    "-pix_fmt", "yuv420p", str(media)], check=True, capture_output=True)
    payload = media.read_bytes()
    state = {"posts": [], "models": [], "hold": False, "starts": [], "peak": 0, "active": set(),
             "full": threading.Event(), "release": threading.Event()}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _json(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer local-secret"
            with lock:
                state["models"].append(body["model"])
                task_id = f"cgt-{len(state['posts'])}"
                state["posts"].append(body)
                state["starts"].append(time.monotonic())
                state["active"].add(task_id)
                state["peak"] = max(state["peak"], len(state["active"]))
                if len(state["active"]) == 5:
                    state["full"].set()
            self._json({"id": task_id})

        def do_GET(self):
            if self.path.startswith("/media/"):
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            task_id = self.path.rsplit("/", 1)[-1]
            if state["hold"] and not state["release"].is_set():
                self._json({"id": task_id, "status": "running"})
                return
            with lock:
                state["active"].discard(task_id)
            port = self.server.server_address[1]
            self._json({"id": task_id, "status": "succeeded",
                        "content": {"video_url": f"http://127.0.0.1:{port}/media/{task_id}.mp4"}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/v3", state
    finally:
        state["release"].set()
        server.shutdown()
        server.server_close()


def test_seedance_processes_share_admission_and_merge_the_ledger(tmp_path, gateway):
    base_url, state = gateway
    state["hold"] = True
    script = '''
import sys
from concurrent.futures import ThreadPoolExecutor
from app.services import material, video_run_config as vrc, volcengine_seedance as seedance
run = vrc.resolve_video_run_config({
    "MPT_SEEDANCE_API_KEY": "local-secret", "MPT_SEEDANCE_BASE_URL": sys.argv[1],
    "MPT_SEEDANCE_MODEL_NAME": "doubao-seedance-2-0-fast-260128",
}).with_runtime(proxies={}, tls_verify=True, poll_interval=0.1, run_timeout=20)
endpoint = seedance.endpoint_from_run_config(run)
def generate(i):
    def record(remote_id):
        material._update_seedance_ledger("shared", i, remote_id=remote_id, status="submitted")
    seedance.generate_coordinated_clip("shot", 4, "9:16", endpoint=endpoint,
        identity=run.credential_fingerprint(), on_task_created=record)
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(generate, range(int(sys.argv[2]), int(sys.argv[2]) + 4)))
'''
    children = []
    try:
        for offset in (0, 4):
            children.append(subprocess.Popen(
                [sys.executable, "-c", script, base_url, str(offset)],
                env={**os.environ, "MPT_HOME": str(tmp_path), "NO_PROXY": "127.0.0.1"},
                cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8",
            ))
        assert state["full"].wait(15), "five accepted tasks should overlap"
        assert len(state["posts"]) == 5
        state["release"].set()
        for child in children:
            out, err = child.communicate(timeout=25)
            assert child.returncode == 0, out + err
        assert state["peak"] == 5 and len(state["posts"]) == 8
        assert all(b - a >= coordinator.SUBMIT_INTERVAL for a, b in zip(state["starts"], state["starts"][1:]))
        ledger = json.loads((Path(utils.task_dir("shared")) / material.SEEDANCE_LEDGER_NAME).read_text())
        assert len(ledger["shots"]) == len(ledger["task_ids"]) == 8
        assert all(record is None for record in _records(tmp_path))
    finally:
        state["release"].set()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


def test_local_gateway_end_to_end_renders_ordered_final_video(monkeypatch, tmp_path, gateway):
    base_url, server_state = gateway
    monkeypatch.setattr(seedance.requests, "post", REAL_POST)
    monkeypatch.setattr(seedance.requests, "get", REAL_GET)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(config, "proxy", {})
    monkeypatch.setattr(tm.sm, "state", MemoryState())
    monkeypatch.setattr(video.VideoAspect, "to_resolution", lambda _self: (96, 160))
    durations = {"第一句。": 1.4, "第二句。": 4.6}

    def tts(**kwargs):
        _pcm(kwargs["voice_file"], durations[kwargs["text"]])
        return SimpleNamespace()

    def create_subtitle(**kwargs):
        Path(kwargs["subtitle_file"]).write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n" + kwargs["text"] + "\n", encoding="utf-8")

    monkeypatch.setattr(storyboard_timing.voice, "tts", tts)
    monkeypatch.setattr(storyboard_timing.voice, "create_subtitle", create_subtitle)
    monkeypatch.setitem(config.app, "subtitle_provider", "edge")
    run_config = vrc.resolve_video_run_config({
        "MPT_SEEDANCE_API_KEY": "local-secret", "MPT_SEEDANCE_BASE_URL": base_url,
        "MPT_SEEDANCE_MODEL_NAME": "doubao-seedance-2-0-mini-pk-260615-custom",
    }).with_runtime(proxies={}, tls_verify=True, poll_interval=0.05, run_timeout=30)
    params = _params(video_script="第一句。第二句。", subtitle_enabled=True, bgm_type="random",
                     bgm_volume=0.2, n_threads=1, video_aspect="9:16")
    result = tm.start("e2e-task", params, storyboard_data=_board("第一句。", "第二句。"),
                      video_run_config=run_config)
    assert result.get("videos"), result
    final = Path(result["videos"][0])
    assert final.name == "final-1.mp4" and final.stat().st_size > 0
    assert server_state["models"] == ["doubao-seedance-2-0-mini-pk-260615-custom"] * 2
    assert sorted(post["duration"] for post in server_state["posts"]) == [4, 5]
    assert all(post["generate_audio"] is False and post["resolution"] == "720p"
               for post in server_state["posts"])
    ledger = json.loads((Path(utils.task_dir("e2e-task")) / material.SEEDANCE_LEDGER_NAME).read_text())
    assert [ledger["shots"][key]["status"] for key in ("1", "2")] == ["downloaded", "downloaded"]
    assert ledger["shots"]["2"]["actual_seconds"] == pytest.approx(5.0, abs=0.1)
    assert (ledger["shots"]["1"]["width"], ledger["shots"]["1"]["height"]) == (96, 160)
    assert Path(result["subtitle_path"]).read_text(encoding="utf-8").count("-->") == 2
    script = json.loads((Path(utils.task_dir("e2e-task")) / "script.json").read_text(encoding="utf-8"))
    shots = script["storyboard_timeline"]["shots"]
    ids_by_prompt = {post["content"][0]["text"]: f"cgt-{i}"
                     for i, post in enumerate(server_state["posts"])}
    assert [shot["remote_task_id"] for shot in shots] == [ids_by_prompt[shot["prompt"]] for shot in shots]
    with VideoFileClipContext(final) as clip:
        assert clip.duration == pytest.approx(6.0, abs=0.2)
    assert "local-secret" not in json.dumps(result, ensure_ascii=False)


class VideoFileClipContext:
    def __init__(self, path):
        from moviepy.video.io.VideoFileClip import VideoFileClip
        self.clip = VideoFileClip(str(path))

    def __enter__(self):
        return self.clip

    def __exit__(self, *_exc):
        self.clip.close()
