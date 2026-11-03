"""Wan3 overlap, ordering and crash recovery, using only simulated paid jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import config
from app.services import material, wan3, wan3_coordinator as coordinator
from app.utils import logging_utils


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    monkeypatch.setenv("MPT_WAN3_API_KEY", "mock-key")
    monkeypatch.setenv("MPT_WAN3_BASE_URL", "https://wan3.invalid")
    monkeypatch.setitem(config.app, "wan3_concurrency", 5)
    monkeypatch.setitem(config.app, "wan3_resolution", "1080P")
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)
    # No test can accidentally submit to the real provider.
    monkeypatch.setattr(wan3.requests, "post", lambda *_a, **_k: pytest.fail("unexpected POST"))
    monkeypatch.setattr(wan3.requests, "get", lambda *_a, **_k: pytest.fail("unexpected GET"))


def _response(body, status=200):
    return SimpleNamespace(status_code=status, json=lambda: body)


def _plans(count):
    return [SimpleNamespace(index=i + 1, prompt=f"prompt-{i}", request_seconds=i + 2,
                            required_source_seconds=1.5, target_playback_seconds=1.5)
            for i in range(count)]


def _download(storyboard=False, count=6, **options):
    common = dict(task_id="local", video_aspect="9:16", material_directory="", **options)
    if storyboard:
        return material._download_storyboard_wan3(shot_plan=_plans(count), **common)
    return material._download_videos_wan3_on_demand(
        search_terms=[f"prompt-{i}" for i in range(count)], audio_duration=count * 2,
        max_clip_duration=5, **common,
    )


@pytest.mark.parametrize("value, expected", [
    (1, 1), ("2", 2), (5, 5), (6, 5), (100, 5), (0, 5), (-1, 5),
    (None, 5), (True, 5), (False, 5), ("bad", 5), ("", 5), (2.5, 5),
    (float("nan"), 5), (float("inf"), 5),
])
def test_concurrency_config(monkeypatch, value, expected):
    monkeypatch.setitem(config.app, "wan3_concurrency", value)
    assert wan3.get_concurrency() == expected


def test_missing_setting_defaults_to_five_without_writing_config(monkeypatch, tmp_path):
    monkeypatch.delitem(config.app, "wan3_concurrency")
    target = tmp_path / "config.toml"
    target.write_text('[app]\nwan3_model_name = "my-model"\n')
    before = target.read_bytes()
    assert wan3.get_concurrency() == 5
    assert "wan3_concurrency" not in config.app and target.read_bytes() == before


@pytest.mark.parametrize("storyboard", [False, True])
def test_five_then_one_sliding_window_and_original_order(monkeypatch, storyboard):
    monkeypatch.delitem(config.app, "wan3_concurrency")  # Exercise the upgraded default.
    started = [threading.Event() for _ in range(6)]
    finish = [threading.Event() for _ in range(6)]
    completed = [threading.Event() for _ in range(6)]
    calls, counts, sources, callback_threads = [], [], [], []
    scheduler_thread = []
    scope_root = []
    log_context = ContextVar("wan3-test-context")

    def generate(**kwargs):
        index = int(kwargs["prompt"].split("-")[-1])
        calls.append((index, kwargs["request_seconds"], kwargs["video_aspect"]))
        assert logging_utils.log_scope_thread_id() == scope_root[0]
        assert log_context.get() == "local-task"
        started[index].set()
        assert finish[index].wait(5)
        kwargs["on_task_created"](f"remote-{index}")
        yield f"clip-{index}", 5.0, {"index": index}, f"remote-{index}"

    def progress(done, total):
        counts.append((done, total))
        callback_threads.append(threading.get_ident())

    def persist(_task, records):
        sources.append(list(records))
        for record in records:
            completed[record["index"]].set()

    def run():
        log_context.set("local-task")
        scheduler_thread.append(threading.get_ident())
        scope_root.append(logging_utils.log_scope_thread_id())
        return _download(storyboard, on_clip_completed=progress)

    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    monkeypatch.setattr(material, "_persist_material_sources", persist)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(run)
        try:
            assert all(event.wait(3) for event in started[:5])
            assert not started[5].is_set()
            finish[3].set()  # Out-of-order completion opens exactly one slot.
            assert started[5].wait(3)
            for index in [5, 1, 4, 2, 0]:
                finish[index].set()
                assert completed[index].wait(3)
            assert result.result(timeout=3) == [f"clip-{i}" for i in range(6)]
        finally:
            for event in finish:
                event.set()
    assert sorted(calls) == [(i, i + 2 if storyboard else 5, "9:16") for i in range(6)]
    assert [record["index"] for record in sources[-1]] == list(range(6))
    assert counts == [(i, 6) for i in range(7)]
    assert set(callback_threads) == set(scheduler_thread)


@pytest.mark.parametrize("storyboard", [False, True])
def test_one_worker_is_serial(monkeypatch, storyboard):
    monkeypatch.setitem(config.app, "wan3_concurrency", 1)
    calls = []
    def generate(**kwargs):
        calls.append(kwargs["prompt"])
        yield kwargs["prompt"], 5.0, {}, "remote"
    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    assert _download(storyboard, count=3) == [f"prompt-{i}" for i in range(3)]
    assert calls == [f"prompt-{i}" for i in range(3)]


@pytest.mark.parametrize("storyboard, short", [(False, False), (True, False), (True, True)])
def test_failure_stops_refill_and_drains_submitted_results(monkeypatch, storyboard, short):
    started = threading.Barrier(5)
    failed = threading.Event()
    recorded, sources, counts, calls = [], [], [], []

    def generate(**kwargs):
        index = int(kwargs["prompt"].split("-")[-1])
        calls.append(index)
        kwargs["on_task_created"](f"remote-{index}")
        started.wait(timeout=3)
        if index == 1:
            failed.set()
            if short:
                yield "short", 0.5, {}, "remote-1"
                return
            raise wan3.Wan3DownloadError("first actual failure", task_id="remote-1")
        assert failed.wait(3)
        assert kwargs["stop_event"].wait(3)
        yield f"clip-{index}", 5.0, {"index": index}, f"remote-{index}"

    # Deliberately unsafe read/modify/write; the scheduler must serialize it.
    def record(remote_id):
        snapshot = list(recorded)
        time.sleep(0.01)
        recorded[:] = [*snapshot, remote_id]

    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    monkeypatch.setattr(material, "_persist_material_sources", lambda _t, r: sources.append(list(r)))
    error_type = wan3.Wan3Error if short else wan3.Wan3DownloadError
    with pytest.raises(error_type, match="too short" if short else "first actual failure") as caught:
        _download(storyboard, on_task_created=record,
                  on_clip_completed=lambda *args: counts.append(args))
    assert caught.value.task_id == "remote-1"
    assert sorted(calls) == list(range(5)) and len(recorded) == 5
    assert [record["index"] for record in sources[-1]] == [0, 2, 3, 4]
    assert counts == [(i, 6) for i in range(5)]


@pytest.mark.parametrize("storyboard", [False, True])
def test_validate_all_inputs_before_start(monkeypatch, storyboard):
    if storyboard:
        plans = _plans(6)
        plans[-1].request_seconds = 31
        with pytest.raises(wan3.Wan3Error):
            material._download_storyboard_wan3(task_id="t", shot_plan=plans,
                                              video_aspect="9:16", material_directory="")
    else:
        with pytest.raises(wan3.Wan3Error):
            material._download_videos_wan3_on_demand(
                task_id="t", search_terms=["ok", ""], video_aspect="9:16",
                audio_duration=5, max_clip_duration=5, material_directory="",
            )


def _records(tmp_path):
    return [json.loads(p.read_text()) for p in (tmp_path / ".agent-logs/wan3").glob("slot-*.json")]


def test_id_is_durable_before_callback_and_terminal_frees_before_download(monkeypatch, tmp_path):
    monkeypatch.setattr(wan3.requests, "post", lambda *_a, **_k: _response({"output": {"task_id": "paid"}}))
    monkeypatch.setattr(wan3.requests, "get", lambda *_a, **_k: _response({
        "output": {"task_status": "SUCCEEDED", "video_url": "https://media.invalid/video.mp4"},
    }))
    def record(remote_id):
        assert remote_id == "paid"
        assert any(r and r["task_id"] == remote_id and r["phase"] == "submitted" for r in _records(tmp_path))
    wan3.generate_videos("prompt", on_task_created=record)
    assert _records(tmp_path) == [None]


def test_throttle_metadata_failure_after_post_cannot_hide_paid_response(monkeypatch, tmp_path):
    original_write = coordinator.atomic_json
    accepted = threading.Event()
    recorded = []

    def write(target, data):
        if target.name == "submit.json" and accepted.is_set():
            raise PermissionError("throttle metadata became unwritable after submission")
        original_write(target, data)

    def post(*_a, **_k):
        accepted.set()
        return _response({"output": {"task_id": "paid"}})

    def record(remote_id):
        assert any(r and r["task_id"] == remote_id for r in _records(tmp_path))
        recorded.append(remote_id)

    monkeypatch.setattr(coordinator, "atomic_json", write)
    monkeypatch.setattr(wan3.requests, "post", post)
    monkeypatch.setattr(wan3.requests, "get", lambda *_a, **_k: _response({
        "output": {"task_status": "SUCCEEDED", "video_url": "https://media.invalid/video.mp4"},
    }))
    wan3.generate_videos("prompt", on_task_created=record)
    assert recorded == ["paid"]
    assert _records(tmp_path) == [None]


@pytest.mark.parametrize("failure", ["network", "timeout", "no-id", "server", "callback"])
def test_unconfirmed_tasks_remain_reserved(monkeypatch, tmp_path, failure):
    def post(*_a, **_k):
        if failure == "network":
            raise requests.exceptions.ConnectionError("interrupted")
        return _response({"output": {} if failure == "no-id" else {"task_id": "paid"}},
                         503 if failure == "server" else 200)
    monkeypatch.setattr(wan3.requests, "post", post)
    monkeypatch.setattr(wan3, "_wait_for_task", lambda **_k: (_ for _ in ()).throw(
        wan3.Wan3UnconfirmedTaskError("timeout", task_id="paid")))
    callback = (lambda _id: (_ for _ in ()).throw(OSError("disk"))) if failure == "callback" else None
    with pytest.raises(wan3.Wan3UnconfirmedTaskError):
        wan3.generate_videos("prompt", on_task_created=callback)
    record = _records(tmp_path)[0]
    assert record["phase"] == ("submitted" if failure in {"timeout", "callback"} else "submitting")
    assert record["task_id"] == ("paid" if failure in {"timeout", "callback"} else "")


@pytest.mark.parametrize("status", ["SUCCEEDED", "FAILED", "CANCELED"])
def test_orphan_recovery_is_read_only_and_terminal_only(tmp_path, status):
    slots = [coordinator.acquire_slot(base_url="base", probe=lambda _id: False) for _ in range(5)]
    for index, slot in enumerate(slots):
        slot.save("submitted", task_id=f"paid-{index}")
        slot.close()
    queried = []
    def probe(remote_id):
        queried.append(remote_id)
        return status in coordinator.TERMINAL_STATES
    recovered = coordinator.acquire_slot(base_url="base", probe=probe)
    try:
        assert queried == ["paid-0"]
        assert recovered.record["phase"] == "reserved"
        assert sum(bool(r) and r["phase"] == "submitted" for r in _records(tmp_path)) == 4
    finally:
        recovered.complete()
        recovered.close()


def test_waiting_worker_checks_stop_before_post(monkeypatch, tmp_path):
    stop, waiting = threading.Event(), threading.Event()
    root = tmp_path / ".agent-logs/wan3"
    root.mkdir(parents=True)
    with coordinator.file_lock(root / "submit.lock"):
        original = coordinator.FileLock.acquire
        def acquire(lock):
            result = original(lock)
            if not result:
                waiting.set()
            return result
        monkeypatch.setattr(coordinator.FileLock, "acquire", acquire)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wan3.generate_videos, "prompt", stop_event=stop)
            assert waiting.wait(3)
            stop.set()
            with pytest.raises(coordinator.SubmissionStopped):
                future.result(timeout=3)
    assert _records(tmp_path) == [None]


def test_worker_waiting_for_capacity_never_posts_after_stop():
    slots = [coordinator.acquire_slot(base_url="base", probe=lambda _id: False) for _ in range(5)]
    stop = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wan3.generate_videos, "prompt", stop_event=stop)
            stop.set()
            with pytest.raises(coordinator.SubmissionStopped):
                future.result(timeout=3)
    finally:
        for slot in slots:
            slot.complete()
            slot.close()


@pytest.mark.parametrize("status", ["FAILED", "CANCELED", "SUCCEEDED"])
def test_confirmed_terminal_state_frees_slot_even_if_output_is_unusable(monkeypatch, tmp_path, status):
    monkeypatch.setattr(wan3.requests, "post", lambda *_a, **_k: _response({"output": {"task_id": "paid"}}))
    monkeypatch.setattr(wan3.requests, "get", lambda *_a, **_k: _response({"output": {"task_status": status}}))
    with pytest.raises(wan3.Wan3Error):
        wan3.generate_videos("prompt")
    assert _records(tmp_path) == [None]


def test_polling_network_failure_retains_id_without_paid_retry(monkeypatch, tmp_path):
    posts = []
    def post(*_a, **_k):
        posts.append(1)
        return _response({"output": {"task_id": "paid"}})
    def get(*_a, **_k):
        raise requests.exceptions.ConnectionError("offline")
    monkeypatch.setattr(wan3.requests, "post", post)
    monkeypatch.setattr(wan3.requests, "get", get)
    monkeypatch.setattr(wan3, "MAX_POLL_RETRIES", 0)
    with pytest.raises(wan3.Wan3UnconfirmedTaskError) as caught:
        wan3.generate_videos("prompt")
    assert caught.value.task_id == "paid" and posts == [1]
    assert _records(tmp_path)[0]["task_id"] == "paid"


def test_local_timeout_never_means_remote_termination(monkeypatch, tmp_path):
    monkeypatch.setattr(wan3.requests, "post", lambda *_a, **_k: _response({"output": {"task_id": "paid"}}))
    ticks = iter([0, 2000])
    monkeypatch.setattr(wan3, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(wan3.Wan3UnconfirmedTaskError, match="timed out"):
        wan3.generate_videos("prompt")
    assert _records(tmp_path)[0]["task_id"] == "paid"


@pytest.mark.parametrize("status", ["RUNNING", "PENDING", "SUCCEEDED", "FAILED", "CANCELED"])
def test_recovery_queries_only_existing_id(monkeypatch, status):
    urls = []
    def get(url, **_k):
        urls.append(url)
        return _response({"output": {"task_status": status}})
    monkeypatch.setattr(wan3.requests, "get", get)
    assert wan3._probe_terminal("existing/id", "https://wan3.invalid", {}) == (status in coordinator.TERMINAL_STATES)
    assert urls == ["https://wan3.invalid/api/v1/tasks/existing%2Fid"]


@pytest.mark.parametrize("body", [
    {}, [], {"output": None}, {"output": {}}, {"output": {"task_status": "UNKNOWN"}},
])
def test_unknown_recovery_status_reports_uncertainty_without_reclaiming_slot(monkeypatch, tmp_path, body):
    slots = [coordinator.acquire_slot(base_url="https://wan3.invalid", probe=lambda _id: False)
             for _ in range(5)]
    for index, slot in enumerate(slots):
        slot.save("submitted", task_id=f"paid-{index}")
        slot.close()
    before = _records(tmp_path)
    monkeypatch.setattr(wan3.requests, "get", lambda *_a, **_k: _response(body))
    with pytest.raises(coordinator.CoordinationError, match="verification"):
        coordinator.acquire_slot(
            base_url="https://wan3.invalid",
            probe=lambda remote_id: wan3._probe_terminal(remote_id, "https://wan3.invalid", {}),
        )
    assert _records(tmp_path) == before


@pytest.mark.parametrize("raw", ["{", "{}", "[]"])
def test_corrupt_records_fail_closed_and_are_not_overwritten(tmp_path, raw):
    root = tmp_path / ".agent-logs/wan3"
    root.mkdir(parents=True)
    target = root / "slot-0.json"
    target.write_text(raw)
    with pytest.raises(coordinator.CoordinationError, match="verification"):
        coordinator.acquire_slot(base_url="base", probe=lambda _id: True)
    assert target.read_text() == raw


def test_parallel_ledger_writes_do_not_lose_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(material.utils, "task_dir", lambda _id: str(tmp_path))
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(lambda i: material._save_wan3_task_id("local", str(i)), range(30)))
    assert set(json.loads((tmp_path / "wan3-tasks.json").read_text())["task_ids"]) == set(map(str, range(30)))


def test_task_callbacks_preserve_every_id_in_state_and_script_on_failure(monkeypatch, tmp_path):
    from app.models.schema import VideoParams
    from app.services import task as tm, task_artifacts
    from app.services.state import MemoryState

    monkeypatch.setattr(material.utils, "task_dir", lambda _id: str(tmp_path))
    state = MemoryState()
    state.update_task("local", progress=40)
    monkeypatch.setattr(tm.sm, "state", state)
    task_artifacts.write_script_data("local", {"script": "unchanged narration"})
    def download(**kwargs):
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(kwargs["on_task_created"], [f"remote-{i}" for i in range(10)]))
        raise wan3.Wan3DownloadError("one failed", task_id="remote-2")
    monkeypatch.setattr(material, "download_videos", download)
    params = VideoParams(video_subject="subject", video_source="wan3")
    assert tm.get_video_materials("local", params, ["prompt"], 5) is None
    expected = {f"remote-{i}" for i in range(10)}
    assert set(state.get_task("local")["wan3_task_ids"]) == expected
    assert set(json.loads((tmp_path / "wan3-tasks.json").read_text())["task_ids"]) == expected
    script = json.loads((tmp_path / "script.json").read_text())
    assert set(script["wan3_task_ids"]) == expected and script["script"] == "unchanged narration"


@pytest.fixture
def mock_service():
    gate, full = threading.Event(), threading.Event()
    lock = threading.Lock()
    tasks, starts, active, peak = {}, [], [0], [0]
    class Handler(BaseHTTPRequestHandler):
        def reply(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                remote_id = str(len(tasks))
                tasks[remote_id] = {"done": False, "payload": data}
                starts.append(time.monotonic())
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                if active[0] == 5:
                    full.set()
            self.reply({"output": {"task_id": remote_id}})
        def do_GET(self):
            remote_id = self.path.rsplit("/", 1)[-1]
            with lock:
                done = gate.is_set()
                if done and not tasks[remote_id]["done"]:
                    tasks[remote_id]["done"] = True
                    active[0] -= 1
            self.reply({"output": {"task_status": "SUCCEEDED" if done else "RUNNING",
                                    "video_url": f"https://media.invalid/{remote_id}.mp4"}})
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", gate=gate,
                              full=full, tasks=tasks, starts=starts, peak=peak)
    finally:
        gate.set()
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def _child(script, tmp_path, url="https://wan3.invalid", *args):
    env = dict(os.environ, MPT_HOME=str(tmp_path), MPT_WAN3_API_KEY="mock-key",
               MPT_WAN3_BASE_URL=url, NO_PROXY="127.0.0.1,localhost")
    return subprocess.Popen([sys.executable, "-c", script, *map(str, args)], env=env,
                            cwd=Path(__file__).resolve().parents[2],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_two_processes_share_five_remote_jobs_and_submission_throttle(tmp_path, mock_service):
    script = '''
import json, sys, time
from pathlib import Path
from app.config import config
from app.services import material
config.app.update(wan3_poll_interval=0.5, wan3_concurrency=5)
material._save_generated_video_with_retry = lambda url, *_a: url
material._get_downloaded_video_duration = lambda _p: 5.0
recorded = []
def record(remote_id):
    snapshot = list(recorded)
    time.sleep(0.02)
    recorded[:] = [*snapshot, remote_id]
    material._save_wan3_task_id(sys.argv[1], remote_id)
paths = material._download_videos_wan3_on_demand(
    task_id=sys.argv[1], search_terms=[f"{sys.argv[1]}-{i}" for i in range(4)],
    video_aspect="9:16", audio_duration=20, max_clip_duration=5,
    material_directory="", on_task_created=record)
Path(sys.argv[2]).write_text(json.dumps(dict(paths=paths, ids=recorded)))
'''
    children = [_child(script, tmp_path, mock_service.url, str(i), tmp_path / f"result-{i}") for i in range(2)]
    try:
        assert mock_service.full.wait(15), "five jobs should overlap before any can finish"
        assert len(mock_service.tasks) == 5
        mock_service.gate.set()
        for child in children:
            out, err = child.communicate(timeout=20)
            assert child.returncode == 0, out + err
        assert mock_service.peak[0] == 5 and len(mock_service.tasks) == 8
        assert all(b - a >= 0.25 for a, b in zip(mock_service.starts, mock_service.starts[1:]))
        for i in range(2):
            result = json.loads((tmp_path / f"result-{i}").read_text())
            assert len(result["ids"]) == 4
            assert [mock_service.tasks[Path(p).stem]["payload"]["input"]["prompt"] for p in result["paths"]] == [f"{i}-{j}" for j in range(4)]
            for remote_id in result["ids"]:
                assert mock_service.tasks[remote_id]["payload"]["parameters"] == {
                    "resolution": "1080P", "ratio": "9:16", "duration": 5, "audio": False,
                }
            ledger = tmp_path / "storage/tasks" / str(i) / "wan3-tasks.json"
            assert set(json.loads(ledger.read_text())["task_ids"]) == set(result["ids"])
        assert all(record is None for record in _records(tmp_path))
    finally:
        mock_service.gate.set()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize("phase", ["submitting", "submitted"])
def test_process_exit_keeps_uncertain_slots_and_confirmed_tasks_can_recover(tmp_path, phase):
    script = '''
import os, sys
from app.services import wan3_coordinator as c
slots = [c.acquire_slot(base_url="base", probe=lambda _id: False) for _ in range(5)]
for i, slot in enumerate(slots):
    slot.save(sys.argv[1], task_id=str(i) if sys.argv[1] == "submitted" else "")
os._exit(0)
'''
    child = _child(script, tmp_path, "https://wan3.invalid", phase)
    out, err = child.communicate(timeout=10)
    assert child.returncode == 0, out + err
    assert len(_records(tmp_path)) == 5
    if phase == "submitting":
        before = _records(tmp_path)
        with pytest.raises(coordinator.CoordinationError, match="verification"):
            coordinator.acquire_slot(base_url="base", probe=lambda _id: pytest.fail("no ID to query"))
        assert _records(tmp_path) == before
    else:
        stop = threading.Event()
        queried = threading.Event()
        def running(_id):
            queried.set()
            return False
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(coordinator.acquire_slot, base_url="base", probe=running, stop_event=stop)
            assert queried.wait(3) and not future.done()
            stop.set()
            with pytest.raises(coordinator.SubmissionStopped):
                future.result(timeout=3)
        assert all(record["phase"] == "submitted" for record in _records(tmp_path))
        slot = coordinator.acquire_slot(base_url="base", probe=lambda _id: True)
        slot.complete()
        slot.close()
