import ast
import dataclasses
from pathlib import Path

import pytest

from app.services import video_run_config as vrc

WAN3_ENV = {
    "MPT_WAN3_API_KEY": "wan3-secret",
    "MPT_WAN3_BASE_URL": "https://ws.cn-beijing.maas.aliyuncs.com/api/v1/",
    "MPT_WAN3_MODEL_NAME": "wan3.0-video",
}
SEEDANCE_ENV = {
    "MPT_SEEDANCE_API_KEY": "gateway-secret",
    "MPT_SEEDANCE_BASE_URL": "https://gateway.example",
    "MPT_SEEDANCE_MODEL_NAME": "Doubao-Seedance-2-0-Fast-260128-Huoshan-New",
}


def test_module_is_standard_library_only():
    tree = ast.parse(Path(vrc.__file__).read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    assert imported <= {"__future__", "hashlib", "re", "collections", "dataclasses", "urllib"}
    assert vrc.VIDEO_RUN_CONFIG_API_VERSION == 1


@pytest.mark.parametrize("environ, reason, options", [
    ({}, vrc.REASON_NO_VIDEO_KEY, ("wan3", "volcengine_seedance")),
    ({"MPT_WAN3_API_KEY": "  ", "MPT_SEEDANCE_API_KEY": ""}, vrc.REASON_NO_VIDEO_KEY,
     ("wan3", "volcengine_seedance")),
    ({"DASHSCOPE_API_KEY": "a", "VOLCENGINE_ARK_API_KEY": "b"}, vrc.REASON_MULTIPLE_VIDEO_KEYS,
     ("wan3", "volcengine_seedance")),
])
def test_zero_or_multiple_candidates_ask(environ, reason, options):
    result = vrc.resolve_video_run_config(environ)
    assert isinstance(result, vrc.VideoInputRequired)
    assert (result.reason, result.options) == (reason, options)
    text = "\n".join(result.lines())
    assert "VIDEO_SOURCE_OPTION=volcengine_seedance|Seedance兼容网关" in text
    assert not any(line.startswith(("MISSING=", "INVALID=")) for line in result.lines())


def test_multiple_candidates_are_asked_even_when_one_is_incomplete():
    environ = {**WAN3_ENV, "MPT_SEEDANCE_API_KEY": "only-a-key"}
    result = vrc.resolve_video_run_config(environ)
    assert result.reason == vrc.REASON_MULTIPLE_VIDEO_KEYS
    assert result.options == ("wan3", "volcengine_seedance")
    assert "only-a-key" not in "\n".join(result.lines())


def test_single_candidate_is_selected_and_alias_counts_once():
    environ = {"MPT_SEEDANCE_API_KEY": "first", "MPT_VOLCENGINE_ARK_API_KEY": "second",
               "VOLCENGINE_ARK_API_KEY": "third", **{k: v for k, v in SEEDANCE_ENV.items()
                                                    if k != "MPT_SEEDANCE_API_KEY"}}
    result = vrc.resolve_video_run_config(environ)
    assert isinstance(result, vrc.VideoRunConfig)
    assert (result.source, result.api_key) == ("volcengine_seedance", "first")
    environ["MPT_SEEDANCE_API_KEY"] = " "
    assert vrc.resolve_video_run_config(environ).api_key == "second"
    environ["MPT_VOLCENGINE_ARK_API_KEY"] = ""
    assert vrc.resolve_video_run_config(environ).api_key == "third"
    wan3 = {**WAN3_ENV, "MPT_WAN3_API_KEY": "", "DASHSCOPE_API_KEY": "dash"}
    assert vrc.resolve_video_run_config(wan3).api_key == "dash"


def test_generic_ark_key_is_not_a_candidate():
    assert vrc.resolve_video_run_config({"ARK_API_KEY": "x"}).reason == vrc.REASON_NO_VIDEO_KEY


def test_explicit_choice_lists_all_missing_and_never_switches():
    result = vrc.resolve_video_run_config(WAN3_ENV, "volcengine_seedance")
    assert result.reason == vrc.REASON_MISSING_VIDEO_SETTINGS
    assert result.source == "volcengine_seedance"
    assert result.missing == ("MPT_SEEDANCE_API_KEY", "MPT_SEEDANCE_BASE_URL",
                              "MPT_SEEDANCE_MODEL_NAME")
    only_key = vrc.resolve_video_run_config({"MPT_WAN3_API_KEY": "k"})
    assert only_key.missing == ("MPT_WAN3_BASE_URL", "MPT_WAN3_MODEL_NAME")


@pytest.mark.parametrize("raw, expected", [
    ("https://gw.example", "https://gw.example/api/v3/contents/generations/tasks"),
    ("https://gw.example/", "https://gw.example/api/v3/contents/generations/tasks"),
    ("https://gw.example/api/v3", "https://gw.example/api/v3/contents/generations/tasks"),
    ("https://gw.example/v1/", "https://gw.example/v1/contents/generations/tasks"),
    ("https://gw.example:8443/api/v3/contents/generations/tasks/",
     "https://gw.example:8443/api/v3/contents/generations/tasks"),
])
def test_seedance_url_is_joined_once(raw, expected):
    assert vrc.normalize_seedance_tasks_url(raw) == expected
    assert vrc.normalize_seedance_tasks_url(expected) == expected


@pytest.mark.parametrize("raw", [
    "ftp://gw.example", "https://user:pw@gw.example", "https://gw.example/?a=1",
    "https://gw.example/#x", "https://gw.example:99999", "gw.example",
    "https://gw.exa mple", "https://gw.example/api\nv3",
])
def test_invalid_url_reports_only_the_field(raw):
    result = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_BASE_URL": raw})
    assert result.reason == vrc.REASON_INVALID_VIDEO_SETTINGS
    assert result.invalid == ("MPT_SEEDANCE_BASE_URL",)
    assert raw not in "\n".join(result.lines())


def test_wan3_uses_native_base_url_normalization():
    assert vrc.resolve_video_run_config(WAN3_ENV).endpoint == "https://ws.cn-beijing.maas.aliyuncs.com"
    result = vrc.resolve_video_run_config(
        {**WAN3_ENV, "MPT_WAN3_BASE_URL": "https://x.example/compatible-mode/v1"})
    assert result.invalid == ("MPT_WAN3_BASE_URL",)


def test_platform_model_name_is_kept_verbatim():
    result = vrc.resolve_video_run_config(
        {**SEEDANCE_ENV, "MPT_SEEDANCE_MODEL_NAME": "  Doubao-Seedance-2-0-Fast-260128-Huoshan-New  "})
    assert result.model == "Doubao-Seedance-2-0-Fast-260128-Huoshan-New"
    assert result.model_family == "seedance-2.0-fast"


@pytest.mark.parametrize("model, family", [
    ("doubao-seedance-2-0-260128", "seedance-2.0"),
    ("doubao-seedance-2-0-260128-huoshan-new", "seedance-2.0"),
    ("doubao-seedance-2-0-fast-260128-huoshan-new", "seedance-2.0-fast"),
    ("doubao-seedance-2-0-mini-pk-260615", "seedance-2.0-mini"),
    ("Doubao-Seedance-2-5-260628", "seedance-2.5"),
    ("doubao-seedance-1-0-pro-250528", "seedance-1.0-pro"),
    ("doubao-seedance-1-0-pro-fast-251015", "seedance-1.0-pro-fast"),
    ("seedance-2.0-mini", "seedance-2.0-mini"),
    ("doubao-seedance-2-0-pk-260615", None),
    ("doubao-seedance-1-0-lite-t2v", None),
    ("my-video-alias", None),
])
def test_family_detection_is_specific_first_and_conservative(model, family):
    assert vrc.detect_seedance_family(model) == family


def test_unknown_alias_needs_family_and_explicit_family_cannot_conflict():
    alias = {**SEEDANCE_ENV, "MPT_SEEDANCE_MODEL_NAME": "team-video-default"}
    assert vrc.resolve_video_run_config(alias).missing == ("MPT_SEEDANCE_MODEL_FAMILY",)
    chosen = vrc.resolve_video_run_config({**alias, "MPT_SEEDANCE_MODEL_FAMILY": "seedance-2.5"})
    assert (chosen.model, chosen.model_family, chosen.max_duration) == ("team-video-default", "seedance-2.5", 30)
    conflict = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_MODEL_FAMILY": "seedance-2.5"})
    assert conflict.invalid == ("MPT_SEEDANCE_MODEL_FAMILY",)
    same = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_MODEL_FAMILY": "seedance-2.0-fast"})
    assert same.model_family == "seedance-2.0-fast"
    bad = vrc.resolve_video_run_config({**alias, "MPT_SEEDANCE_MODEL_FAMILY": "seedance-9"})
    assert bad.invalid == ("MPT_SEEDANCE_MODEL_FAMILY",)


@pytest.mark.parametrize("model, resolution, bounds, audio", [
    ("doubao-seedance-2-0-fast-260128", "720p", (4, 15), True),
    ("doubao-seedance-2-0-mini-pk-260615", "720p", (4, 15), True),
    ("doubao-seedance-2-0-260128", "1080p", (4, 15), True),
    ("doubao-seedance-2-5-260628", "1080p", (4, 30), True),
    ("doubao-seedance-1-0-pro-250528", "1080p", (2, 12), False),
])
def test_capabilities_and_default_resolution(model, resolution, bounds, audio):
    result = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_MODEL_NAME": model})
    assert result.resolution == resolution
    assert (result.min_duration, result.max_duration) == bounds
    assert result.send_generate_audio is audio and result.generate_audio is False


def test_fast_and_mini_reject_1080p_and_concurrency_is_bounded():
    result = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_RESOLUTION": "1080P"})
    assert result.invalid == ("MPT_SEEDANCE_RESOLUTION",)
    for raw in ("0", "6", "two", "-1"):
        assert vrc.resolve_video_run_config(
            {**SEEDANCE_ENV, "MPT_SEEDANCE_CONCURRENCY": raw}).invalid == ("MPT_SEEDANCE_CONCURRENCY",)
    assert vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_CONCURRENCY": "2"}).concurrency == 2
    assert vrc.resolve_video_run_config(SEEDANCE_ENV).concurrency == 3
    for raw in ("", "  "):
        assert vrc.resolve_video_run_config(
            {**SEEDANCE_ENV, "MPT_SEEDANCE_CONCURRENCY": raw}).concurrency == 3
    assert vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_CONCURRENCY": "5"}).concurrency == 5
    wan3 = vrc.resolve_video_run_config({**WAN3_ENV, "MPT_WAN3_RESOLUTION": "720p"})
    assert (wan3.resolution, wan3.min_duration, wan3.max_duration) == ("720P", 2, 30)
    assert wan3.concurrency == 5


def test_secrets_are_hidden_and_child_environment_overrides_aliases():
    environ = {**SEEDANCE_ENV, "MPT_SEEDANCE_API_KEY": "", "MPT_VOLCENGINE_ARK_API_KEY": "chosen",
               "VOLCENGINE_ARK_API_KEY": "stale", "MPT_SEEDANCE_BASE_URL": "https://gw.example/api/v3/"}
    config = vrc.resolve_video_run_config(environ).with_runtime(
        proxies={"https": "http://user:proxy-secret@proxy"}, tls_verify=True,
        poll_interval=5, run_timeout=1800,
    )
    assert "chosen" not in repr(config) and "proxy-secret" not in repr(config)
    assert "chosen" not in str(config.summary())
    assert set(config.secrets()) == {"chosen", "http://user:proxy-secret@proxy"}
    child = config.child_environment(environ)
    assert child["MPT_SEEDANCE_API_KEY"] == child["MPT_VOLCENGINE_ARK_API_KEY"] == child["VOLCENGINE_ARK_API_KEY"] == "chosen"
    assert child["MPT_SEEDANCE_BASE_URL"] == "https://gw.example/api/v3/contents/generations/tasks"
    assert environ["VOLCENGINE_ARK_API_KEY"] == "stale"
    again = vrc.resolve_video_run_config(child, "volcengine_seedance")
    assert dataclasses.replace(config, proxies=None, tls_verify=None, poll_interval=None,
                               run_timeout=None) == again


def test_fingerprint_changes_with_key_or_endpoint():
    first = vrc.resolve_video_run_config(SEEDANCE_ENV)
    other_key = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_API_KEY": "rotated"})
    other_url = vrc.resolve_video_run_config({**SEEDANCE_ENV, "MPT_SEEDANCE_BASE_URL": "https://b.example"})
    assert len({first.credential_fingerprint(), other_key.credential_fingerprint(),
                other_url.credential_fingerprint()}) == 3
    assert "gateway-secret" not in first.credential_fingerprint()


def test_config_is_frozen():
    config = vrc.resolve_video_run_config(WAN3_ENV)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model = "other"


def test_wan3_bounds_match_provider_module():
    from app.services import wan3
    assert (vrc.WAN3_FAMILY.min_duration, vrc.WAN3_FAMILY.max_duration) == (
        wan3.MIN_CLIP_SECONDS, wan3.MAX_CLIP_SECONDS)


def test_unknown_requested_source_is_a_programming_error():
    with pytest.raises(ValueError):
        vrc.resolve_video_run_config({}, "pexels")
