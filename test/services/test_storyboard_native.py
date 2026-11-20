"""Native audio regression tests: simulated providers and real local FFmpeg only."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
import pytest

import cli
from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoParams
from app.services import material, storyboard_checkpoint as cp, storyboard_native as native
from app.services import task, video, video_run_config as vrc, volcengine_seedance as seedance, wan3
from app.services.state import MemoryState
from app.services.storyboard import parse_native_storyboard, parse_storyboard
from app.utils import utils


BOARD = """# Storyboard — Native test

**Aspect:** 16:9

## Beats (timed)

| Time | Shot type | On-screen / B-roll | VO / on-camera line | Music / SFX |
| --- | --- | --- | --- | --- |
| 0–4.01 | Wide | Red visual | Alpha line. | Warm piano at 80 BPM; gentle entrance; rain. |
| 4.01–9.01 | Detail | Green visual | Beta line. | Fast drums at 120 BPM; soft ending; wind. |
| 9.01–13.01 | Close | Blue visual | — | — |

## Voiceover script (clean, ready to paste)
Alpha line.
Beta line.

## Narration voice
English, warm alto, calm and unhurried.

## Captions guidance
English, white text, bottom safe area, two lines.
"""


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    monkeypatch.setattr(task.sm, "state", MemoryState())
    monkeypatch.setattr(wan3.requests, "post", Mock(side_effect=AssertionError("unexpected paid request")))
    monkeypatch.setattr(wan3.requests, "get", Mock(side_effect=AssertionError("unexpected network request")))


@pytest.fixture
def board_path(tmp_path):
    target = tmp_path / "storyboard.md"
    target.write_text(BOARD, encoding="utf-8")
    return target


def run_config(source="wan3", **overrides):
    settings = dict(source=source, model="gateway-model", model_family="wan3" if source == "wan3" else "seedance-2.0",
                    api_key="test-secret", endpoint="https://gateway.invalid", resolution="720P" if source == "wan3" else "720p",
                    min_duration=2 if source == "wan3" else 4, max_duration=30, concurrency=1)
    return vrc.VideoRunConfig(**(settings | overrides))


def native_argv(path, *extra):
    return ["--storyboard", str(path), "--storyboard-audio-mode", "native",
            "--video-source", "wan3", "--video-config-from-env", *extra]


def test_native_fields_only_reach_their_own_shot_and_legacy_shape_is_unchanged(board_path):
    old = parse_storyboard(board_path, require_shot_timing=True)
    assert set(asdict(old)) == {"video_script", "video_terms", "aspect", "title", "shots"}
    assert set(asdict(old.shots[0])) == {"index", "prompt", "narration", "planned_start_seconds", "planned_end_seconds"}
    board = parse_native_storyboard(board_path)
    plans = native.build_generation_plan(board, run_config())
    assert [p.request_seconds for p in plans] == [5, 5, 4]
    assert "Wide: Red visual" in plans[0].prompt
    assert "Alpha line." in plans[0].prompt and "Beta line." not in plans[0].prompt
    assert "Warm piano at 80 BPM" in plans[0].prompt and "Fast drums" not in plans[0].prompt
    assert board.narration_voice in plans[0].prompt and board.captions_guidance in plans[0].prompt
    assert "0.2 seconds" in plans[0].prompt
    assert "No dialogue, narration" in plans[2].prompt
    assert "No background music and no additional sound effects" in plans[2].prompt
    assert "Alpha line." not in plans[2].prompt and "Generate captions" not in plans[2].prompt
    assert board.shots[0].narration == "Alpha line."


def test_full_music_design_and_unchanged_voice_reach_every_segment(board_path):
    music = (
        "Electronic piano, 88 BPM, restrained, no singing.\n"
        "Segment 1: sparse notes and a gentle pulse.\n"
        "Segment 2: light bass and a steady beat, open ending.\n"
        "Segment 3: fewer notes and a gradual ending."
    )
    board_path.write_text(BOARD + "\n## Music mood\n" + music + "\n", encoding="utf-8")
    board = parse_native_storyboard(board_path)
    assert board.music_mood == music
    plans = native.build_generation_plan(board, run_config())
    for position, (shot, plan) in enumerate(zip(board.shots, plans, strict=True), start=1):
        assert plan.prompt.count(music) == 1
        assert plan.prompt.count(board.narration_voice) == 1
        assert "keep the same perceived speaker, accent, age, pitch range, timbre" in plan.prompt
        assert f"Current music segment: {position} of 3." in plan.prompt
        assert (f"Planned position in the whole video: {shot.planned_start_seconds:g}–"
                f"{shot.planned_end_seconds:g} seconds.") in plan.prompt
        assert f"Generate only this {plan.request_seconds}-second segment" in plan.prompt
        assert "do not generate the whole soundtrack or compress all segments" in plan.prompt
        assert "current shot's concrete music / sound-effect instructions take precedence" in plan.prompt
        other_lines = [other.narration for other in board.shots
                       if other.index != shot.index and other.narration]
        assert all(line not in plan.prompt for line in other_lines)
    assert "No background music and no additional sound effects." in plans[2].prompt
    assert "if they specify no music or effects, do not add them" in plans[2].prompt


@pytest.mark.parametrize("music_section", ["", "\n## Music mood\n\n", "\n## Music mood\n—\n"])
def test_old_native_storyboards_without_music_design_remain_readable(board_path, music_section):
    board_path.write_text(BOARD + music_section, encoding="utf-8")
    board = parse_native_storyboard(board_path)
    assert board.music_mood == ""
    plans = native.build_generation_plan(board, run_config())
    assert "Whole-video music design" not in plans[0].prompt
    assert "Warm piano at 80 BPM" in plans[0].prompt
    assert "Current music segment: 1 of 3." in plans[0].prompt


@pytest.mark.parametrize("section", ["Narration voice", "Captions guidance"])
def test_missing_and_empty_guidance_fail_before_generation(board_path, section):
    import re

    text = BOARD
    for replacement in ("", f"## {section}\n\n—\n\n"):
        board_path.write_text(re.sub(r"## " + section + r"\n.*?(?=## |\Z)", replacement, text, flags=re.S))
        with pytest.raises(ValueError, match=section):
            parse_native_storyboard(board_path)


def test_legacy_clean_narration_is_ignored_but_tts_still_checks_it(board_path):
    board_path.write_text(BOARD.replace("Beta line.\n\n##", "Different text.\n\n##"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        parse_storyboard(board_path, require_shot_timing=True)
    assert parse_native_storyboard(board_path).video_script == "Alpha line.\nBeta line."


@pytest.mark.parametrize("clean", [None, "", "—", "Wrong dialogue.\n\n—"])
def test_native_script_is_derived_only_from_shots(board_path, clean):
    replacement = "" if clean is None else f"## Voiceover script (clean, ready to paste)\n{clean}\n\n"
    text = re.sub(r"## Voiceover script[^\n]*\n.*?(?=## |\Z)", replacement, BOARD, flags=re.S)
    text = text.replace("| Alpha line. |", "| Alpha line.<br><br>Another line. |")
    board_path.write_text(text, encoding="utf-8")
    board = parse_native_storyboard(board_path)
    assert board.video_script == "Alpha line.\n\nAnother line.\nBeta line."
    plans = native.build_generation_plan(board, run_config())
    assert "Another line." in plans[0].prompt
    assert "Wrong dialogue." not in "\n".join(plan.prompt for plan in plans)
    assert "No dialogue" in plans[2].prompt


@pytest.mark.parametrize("marker", ["-", "—", "–", " - ", '"-"', '" - "', "“—”", ""])
def test_native_silence_markers_never_become_speech_or_sound(board_path, marker):
    text = re.sub(r"## Voiceover script[^\n]*\n.*?(?=## |\Z)", "", BOARD, flags=re.S)
    text = text.replace("| — | — |", f"| {marker} | {marker} |")
    board_path.write_text(text, encoding="utf-8")
    board = parse_native_storyboard(board_path)
    assert board.shots[2].narration == ""
    assert board.shots[2].music_sfx == ""
    assert board.video_script == "Alpha line.\nBeta line."
    plan = native.build_generation_plan(board, run_config())[2]
    assert "No dialogue" in plan.prompt
    assert "No background music and no additional sound effects" in plan.prompt
    assert "Speak exactly" not in plan.prompt
    assert "Generate captions" not in plan.prompt


def test_native_preserves_hyphens_inside_real_dialogue(board_path):
    board_path.write_text(BOARD.replace("Alpha line.", "A-B — still spoken."), encoding="utf-8")
    board = parse_native_storyboard(board_path)
    assert board.shots[0].narration == "A-B — still spoken."
    assert board.video_script == "A-B — still spoken.\nBeta line."


def test_fully_silent_native_board_needs_no_clean_section_or_voice_guidance(board_path):
    board_path.write_text(BOARD.replace("Alpha line.", "—").replace("Beta line.", "—")
                          .split("## Voiceover script")[0], encoding="utf-8")
    board = parse_native_storyboard(board_path)
    assert board.video_script == ""
    assert all("No dialogue" in p.prompt for p in native.build_generation_plan(board, run_config()))


@pytest.mark.parametrize("duration,minimum,maximum,expected", [(2.001, 2, 30, 3), (30.001, 2, 30, None), (1, 2, 30, None)])
def test_integer_requests_are_never_clamped(board_path, duration, minimum, maximum, expected):
    board = parse_native_storyboard(board_path)
    board = replace(board, shots=(replace(board.shots[0], planned_start_seconds=0, planned_end_seconds=duration),))
    if expected is None:
        with pytest.raises(ValueError, match="outside model range"):
            native.build_generation_plan(board, run_config(min_duration=minimum, max_duration=maximum))
    else:
        assert native.build_generation_plan(board, run_config())[0].request_seconds == expected


@pytest.mark.parametrize("options", [
    ["--voice-name", "no-voice"], ["--voice-volume", "0"], ["--voice-rate", "1"],
    ["--custom-audio-file", "missing.wav"], ["--bgm-type", "none"], ["--bgm-file", "missing.mp3"],
    ["--bgm-volume", "0"], ["--font-name", "missing.ttf"], ["--no-subtitle-enabled"],
    ["--subtitle-position", "top"], ["--stroke-color", "#FFFFFF"], ["--font-size", "40"],
    ["--video-clip-duration", "5"], ["--video-clip-speed", "1.1"], ["--video-transition-mode", "fade-in"],
    ["--video-music-prompt", "piano"], ["--subtitle-background-enabled"], ["--video-count", "2"],
])
def test_explicit_replacement_options_fail_at_cli_parse(board_path, options):
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(native_argv(board_path, *options))
    assert caught.value.code == 2


def test_cli_defaults_stay_tts_and_saved_audio_resources_do_not_apply(board_path, monkeypatch):
    assert cli.parse_args(["--video-subject", "old task"]).storyboard_audio_mode == "tts"
    monkeypatch.setattr(config, "ui", {"voice_name": "invalid", "font_name": "missing.ttf", "video_clip_speed": 1.5,
                                     "voice_rate": 2, "subtitle_enabled": True, "bgm_type": "custom"})
    args = cli.parse_args(native_argv(board_path, "--video-clip-speed", "1"))
    params = cli.build_video_params(args)
    cli.prepare_cli_files(params, "video")
    assert (params.voice_name, params.bgm_type, params.font_name, params.video_clip_speed) == ("no-voice", "", "", 1)
    assert not params.subtitle_enabled and not params.custom_audio_file


@pytest.mark.parametrize("source, prefix, model, expected", [
    ("wan3", "WAN3", "wan3.0-video", "720P"),
    ("volcengine_seedance", "SEEDANCE", "doubao-seedance-2-0-mini", "720p"),
])
def test_storyboard_resolution_reaches_native_payload(board_path, monkeypatch, source, prefix, model, expected):
    for name, value in {"API_KEY": "test", "BASE_URL": "https://gateway.invalid",
                        "MODEL_NAME": model, "RESOLUTION": "480p"}.items():
        monkeypatch.setenv(f"MPT_{prefix}_{name}", value)
    old_board = parse_native_storyboard(board_path)
    assert old_board.resolution is None
    assert cli._video_run_config_from_environment(source, old_board.resolution).resolution.lower() == "480p"
    board_path.write_text(BOARD.replace("**Aspect:** 16:9", "**Aspect:** 16:9\n**Resolution:** 720p"), encoding="utf-8")
    board = parse_native_storyboard(board_path)
    resolved = native.audio_run_config(cli._video_run_config_from_environment(source, board.resolution))
    if source == "wan3":
        created = {"output": {"task_id": "local-id"}}
        finished = {"output": {"task_status": "SUCCEEDED", "video_url": "https://media.invalid/test.mp4"}}
    else:
        created = {"id": "local-id"}
        finished = {"id": "local-id", "status": "succeeded", "content": {"video_url": "https://media.invalid/test.mp4"}}
    post = Mock(return_value=SimpleNamespace(status_code=200, headers={}, json=lambda: created))
    monkeypatch.setattr(wan3.requests, "post", post)
    monkeypatch.setattr(wan3.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200, headers={}, json=lambda: finished))
    if source == "wan3":
        wan3.generate_videos("prompt", 4, "16:9", run_config=resolved)
        params = post.call_args.kwargs["json"]["parameters"]
    else:
        seedance.generate_videos("prompt", 4, "16:9", endpoint=seedance.endpoint_from_run_config(resolved))
        params = post.call_args.kwargs["json"]
    assert params["resolution"] == expected


def test_cli_rejects_mini_1080p_before_submission(board_path, monkeypatch, capsys):
    board_path.write_text(BOARD.replace("**Aspect:** 16:9", "**Aspect:** 16:9\n**Resolution:** 1080p"), encoding="utf-8")
    for name, value in {"API_KEY": "test", "BASE_URL": "https://gateway.invalid",
                        "MODEL_NAME": "doubao-seedance-2-0-mini"}.items():
        monkeypatch.setenv(f"MPT_SEEDANCE_{name}", value)
    args = native_argv(board_path, "--confirm-seedance-charge")
    args[args.index("wan3")] = "volcengine_seedance"
    assert cli.run_cli(args) == 2
    output = capsys.readouterr()
    assert "doubao-seedance-2-0-mini" in output.out + output.err
    assert "480p / 720p" in output.out + output.err
    wan3.requests.post.assert_not_called()


def test_cli_preflight_rejects_model_limits_before_task_start(board_path, monkeypatch):
    monkeypatch.setattr(cli, "_video_run_config_from_environment", lambda source, resolution: run_config(max_duration=4))
    start = Mock(side_effect=AssertionError("must fail before starting a task"))
    monkeypatch.setattr(task, "start", start)
    assert cli.run_cli(native_argv(board_path)) == 2
    start.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_wan3_audio_payload_and_legacy_disabled_default(monkeypatch, enabled):
    slot = SimpleNamespace(record={"phase": "submitted"}, close=lambda: None)
    monkeypatch.setattr(wan3.coordinator, "acquire_slot", lambda **kwargs: slot)
    captured = []
    monkeypatch.setattr(wan3, "_submit_and_wait", lambda *args: captured.append(args[1]) or [])
    resolved = run_config()
    resolved = native.audio_run_config(resolved) if enabled else resolved
    wan3.generate_videos("prompt", 4, run_config=resolved)
    assert captured[0]["parameters"]["audio"] is enabled


@pytest.mark.parametrize("family", ["seedance-2.0", "seedance-2.5", "unknown-gateway-family"])
def test_seedance_native_enables_audio_even_for_unknown_gateway_family(family):
    resolved = run_config("volcengine_seedance", model_family=family)
    legacy = seedance._video_payload("prompt", 5, VideoAspect.portrait, seedance.endpoint_from_run_config(
        replace(resolved, send_generate_audio=True)))
    payload = seedance._video_payload("prompt", 5, VideoAspect.portrait,
                                     seedance.endpoint_from_run_config(native.audio_run_config(resolved)))
    assert legacy["generate_audio"] is False and payload["generate_audio"] is True


@pytest.mark.parametrize("family", ["seedance-1.0-pro", "seedance-1.0-pro-fast"])
def test_known_silent_families_fail_but_wan3_switch_metadata_is_not_audio_capability(board_path, family):
    board = parse_native_storyboard(board_path)
    with pytest.raises(ValueError, match="does not support native audio"):
        native.build_generation_plan(board, run_config("volcengine_seedance", model_family=family))
    assert native.audio_run_config(run_config(send_generate_audio=False)).generate_audio


def test_seedance_native_uses_shared_ordered_scheduler_and_does_not_require_tts_coverage(board_path, tmp_path, monkeypatch):
    plans = native.build_generation_plan(parse_native_storyboard(board_path), run_config("volcengine_seedance"))
    endpoint_audio = []

    def generate(prompt, seconds, aspect, **kwargs):
        endpoint_audio.append(kwargs["endpoint"].generate_audio)
        remote_id = f"paid-{len(endpoint_audio)}"
        kwargs["on_task_created"](remote_id)
        return MaterialInfo(provider="volcengine_seedance", url="https://media.invalid/clip", duration=seconds,
                            source_info={"asset_id": remote_id})

    clip = tmp_path / "short-native.mp4"
    clip.write_bytes(b"downloaded")
    monkeypatch.setattr(seedance, "generate_coordinated_clip", generate)
    monkeypatch.setattr(material, "_save_generated_video_with_retry", lambda *args, **kwargs: str(clip))
    probe = Mock(return_value=(0.5, 160, 90))
    monkeypatch.setattr(material, "_measure_generated_video", probe)
    saved = material._download_storyboard_seedance(
        task_id="seedance-native", shot_plan=plans, video_aspect="16:9", material_directory=str(tmp_path),
        video_run_config=native.audio_run_config(run_config("volcengine_seedance")),
    )
    assert saved == [str(clip)] * 3 and endpoint_audio == [True] * 3
    assert [plan.material_seconds for plan in plans] == [0.5] * 3
    assert [plan.remote_task_id for plan in plans] == ["paid-1", "paid-2", "paid-3"]
    assert all(call.kwargs == {"audio": False} for call in probe.call_args_list)


def test_api_rejection_stops_dispatch_and_keeps_previous_download_and_remote_ids(board_path, tmp_path, monkeypatch):
    board = parse_native_storyboard(board_path)
    paid = []
    downloaded = Path(utils.storage_dir("cache_videos", create=True)) / "first.mp4"

    def generate(**kwargs):
        paid.append(kwargs)
        remote_id = f"paid-{len(paid)}"
        kwargs["on_task_created"](remote_id)
        if len(paid) == 2:
            raise wan3.Wan3Error("gateway rejected audio request", task_id=remote_id)
        downloaded.write_bytes(b"first paid result")
        return [(str(downloaded), 0.5, {}, remote_id)]

    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    result = cp.start(task_id="rejected-native", params=VideoParams(video_subject="native", video_source="wan3"),
                      storyboard_data=board, video_run_config=run_config(), storyboard_audio_mode="native")
    assert result["failed_stage"] == "materials"
    assert result["wan3_task_id"] == "paid-2"
    assert len(paid) == 2 and downloaded.exists()
    checkpoint = json.loads((Path(utils.task_dir("rejected-native")) / "storyboard-checkpoint.json").read_text())
    assert checkpoint["clips"]["1"]["status"] == "completed"
    assert checkpoint["clips"]["2"]["remote_id"] == "paid-2"
    assert checkpoint["clips"]["3"]["status"] == "pending"


def make_clip(target, duration, color="red", frequency=440, audio=True, silence=False, size="160x90"):
    command = [utils.get_ffmpeg_binary(), "-y", "-v", "error", "-f", "lavfi", "-i",
               f"color=c={color}:s={size}:r=30:d={duration}"]
    if audio:
        source = f"anullsrc=r=48000:cl=mono:d={duration}" if silence else f"sine=frequency={frequency}:sample_rate=48000:duration={duration}"
        command += ["-f", "lavfi", "-i", source, "-c:a", "pcm_s16le"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(target)]
    subprocess.run(command, check=True, capture_output=True)
    return str(target)


@pytest.mark.parametrize("provider", ["wan3", "volcengine_seedance"])
def test_five_shot_revision_pins_four_clips_and_resumes_without_submitting_again(tmp_path, monkeypatch, provider):
    from app.services import storyboard_revision as revision

    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (160, 90))
    board_path = tmp_path / "board.md"
    rows = "\n".join(f"| {4*i}–{4*i+4} | Wide | Coffee {i} | — | — |" for i in range(5))
    board_path.write_text(BOARD[:BOARD.index("| 0–")] + rows + "\n", encoding="utf-8")
    board = parse_native_storyboard(board_path)
    saved = Path(utils.storage_dir("cache_videos", create=True))
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["prompt"])
        remote = f"paid-{len(calls)}"
        kwargs["on_task_created"](remote)
        path = make_clip(saved / f"{remote}.mkv", 0.2, frequency=400 + len(calls)*20)
        return [(path, 0.2, {}, remote)]

    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    if provider == "volcengine_seedance":
        def generate_seedance(prompt, _seconds, _aspect, **kwargs):
            path, _, _, remote = generate(prompt=prompt, on_task_created=kwargs["on_task_created"])[0]
            return MaterialInfo(provider=provider, url=path, source_info={"asset_id": remote})
        monkeypatch.setattr(seedance, "generate_coordinated_clip", generate_seedance)
        monkeypatch.setattr(material, "_save_generated_video_with_retry", lambda url, *args, **kwargs: url)
    config = run_config(provider)
    options = dict(params=VideoParams(video_subject="Coffee", video_source=provider, video_aspect="16:9"),
                   storyboard_data=board, video_run_config=config, storyboard_audio_mode="native")
    parent = cp.start(task_id="parent", **options)
    root = Path(utils.task_dir("parent"))
    original = json.loads((root / revision.MANIFEST).read_text())
    source = dict(task_id="parent", final_sha256=cp.digest(parent["videos"][0]))
    assert len(calls) == 5
    assert [r["start_frame"] for r in original["shots"]] == [0, 6, 12, 18, 24]
    board_path.write_text(board_path.read_text(encoding="utf-8").replace("Coffee 2", "Overhead pour"), encoding="utf-8")
    options["storyboard_data"] = board = parse_native_storyboard(board_path)
    plan, _ = revision.preflight(root.parent, source, board, config, "16:9", "partial", [3])
    assert [r["action"] for r in plan["shots"]] == ["reuse", "reuse", "generate", "reuse", "reuse"]
    with pytest.raises(ValueError, match="重新预检"):
        cp.start(task_id="drift", revision_plan=plan, **{**options, "video_run_config": replace(config, model="changed")})
    assert len(calls) == 5
    child = cp.start(task_id="child", revision_plan=plan, **options)
    assert len(calls) == 6 and "Overhead pour" in calls[-1]
    assert native.inspect_clip(child["videos"][0])["duration"] > 0
    updated = json.loads((root.parent / "child" / revision.MANIFEST).read_text())
    assert [a["sha256"] == b["sha256"] for a, b in zip(original["shots"], updated["shots"])] == [True, True, False, True, True]
    assert cp.start(task_id="child", revision_plan=plan, resume=True, **options)["videos"] == child["videos"]
    assert len(calls) == 6 and cp.digest(parent["videos"][0]) == source["final_sha256"]
    child_source = dict(task_id="child", final_sha256=updated["final_sha256"])
    resample, _ = revision.preflight(root.parent, child_source, board, config, "16:9", "partial", [3])
    assert resample["shots"][2]["reason"] == "requested"
    third = cp.start(task_id="third", revision_plan=resample, **options)
    assert third["videos"] and len(calls) == 7
    assert cp.digest(child["videos"][0]) == child_source["final_sha256"]
    missing = root / original["shots"][0]["path"]
    original_bytes = missing.read_bytes()
    missing.unlink()
    lookup = Mock(side_effect=wan3.Wan3DownloadError("expired", task_id="paid-1"))
    monkeypatch.setattr(wan3 if provider == "wan3" else seedance, "recover_video", lookup)
    changed, _ = revision.preflight(root.parent, source, board, config, "16:9", "partial", [3])
    assert changed["shots"][0]["reason"] == "missing_material" and lookup.call_count == 1
    with pytest.raises(ValueError, match="重新预检"):
        cp.start(task_id="stale", revision_plan=plan, **options)
    assert len(calls) == 7
    lookup.side_effect = None
    lookup.return_value = MaterialInfo(provider="wan3", url="https://media.invalid/1")
    def recover(_url, directory, _provider, **_):
        recovered = Path(directory) / "recovered.mkv"
        recovered.write_bytes(original_bytes)
        return str(recovered)
    with monkeypatch.context() as recovery:
        recovery.setattr(material, "_save_generated_video_with_retry", recover)
        recovered, _ = revision.preflight(root.parent, source, board, config, "16:9", "partial", [3])
    assert recovered == plan and not missing.exists()
    (root / revision.MANIFEST).unlink()
    with pytest.raises(ValueError, match="整片重做"):
        revision.preflight(root.parent, source, board, config, "16:9", "partial", [3])
    full, _ = revision.preflight(root.parent, source, board, config, "16:9", "full", [])
    assert cp.start(task_id="legacy-full", revision_plan=full, **options)["videos"]
    assert len(calls) == 12 and cp.digest(parent["videos"][0]) == source["final_sha256"]


def test_native_landscape_720p_material_exports_full_hd(tmp_path):
    clip = make_clip(tmp_path / "720p.mkv", 0.5, size="1280x720")
    output = tmp_path / "final.mp4"
    native.render_native([clip], [native.NativeShotPlan(1, "prompt", 4, 4)],
                         str(output), aspect="16:9")
    with av.open(str(output)) as media:
        stream = media.streams.video[0]
        assert (stream.width, stream.height) == (1920, 1080)


def audio_samples(filename):
    with av.open(str(filename)) as media:
        arrays = [frame.to_ndarray()[0].astype(np.float32) for frame in media.decode(audio=0)]
    return np.concatenate(arrays)


def test_real_three_clip_assembly_keeps_order_audio_fades_and_actual_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (320, 180))
    durations = [0.8, 3.1, 0.6]
    clips = [make_clip(tmp_path / f"{index}.mkv", duration, color, tone)
             for index, (duration, color, tone) in enumerate(zip(durations, ["red", "lime", "blue"], [440, 880, 1320], strict=True))]
    # Planned/requested seconds deliberately disagree with all actual clips.
    plans = [native.NativeShotPlan(i + 1, "prompt", seconds, seconds)
             for i, seconds in enumerate([5, 2, 5])]
    output = tmp_path / "final.mp4"
    result = native.render_native(clips, plans, str(output), aspect="16:9")
    assert result["actual_seconds"] == pytest.approx(durations, abs=0.002)
    assert result["output_seconds"] == pytest.approx(sum(durations), abs=1 / 30 + 1024 / 48000)
    samples = audio_samples(output)
    elapsed = 0
    with av.open(str(output)) as media:
        frames = [(float(frame.time), frame.to_ndarray(format="rgb24").mean(axis=(0, 1))) for frame in media.decode(video=0)]
    for index, (duration, tone) in enumerate(zip(durations, [440, 880, 1320], strict=True)):
        center = elapsed + duration / 2
        color = min(frames, key=lambda item: abs(item[0] - center))[1]
        assert int(color.argmax()) == index
        middle = samples[round((center - 0.08) * 48000):round((center + 0.08) * 48000)]
        frequencies = np.fft.rfftfreq(len(middle), 1 / 48000)
        assert frequencies[np.abs(np.fft.rfft(middle)).argmax()] == pytest.approx(tone, abs=10)
        def rms(window):
            return float(np.sqrt(np.mean(window ** 2)))
        start = samples[round((elapsed + 0.01) * 48000):round((elapsed + 0.04) * 48000)]
        end = samples[round((elapsed + duration - 0.04) * 48000):round((elapsed + duration - 0.01) * 48000)]
        assert rms(start) < rms(middle) * 0.35
        assert rms(end) < rms(middle) * 0.35
        # A color boundary and a new tone occur on the same frame, without
        # accumulating per-clip AAC priming or overlapping the preceding shot.
        if index:
            boundary = next(time for time, value in frames if int(value.argmax()) == index)
            assert boundary == pytest.approx(elapsed, abs=1 / 30)
        elapsed += duration


def test_digital_silence_is_valid_and_short_fades_do_not_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (160, 90))
    clip = make_clip(tmp_path / "short.mkv", 0.1, silence=True)
    assert native.inspect_clip(clip)["audio_samples"] > 0
    result = native.render_native([clip], [native.NativeShotPlan(1, "prompt", 5, 5)],
                                  str(tmp_path / "short.mp4"), aspect="16:9")
    assert result["output_seconds"] == pytest.approx(0.1, abs=0.025)


@pytest.mark.parametrize("missing_audio", [True, False])
def test_missing_or_corrupt_audio_is_a_local_failure_and_keeps_original(tmp_path, missing_audio):
    clip = tmp_path / "original.mkv"
    if missing_audio:
        make_clip(clip, 0.5, audio=False)
    else:
        clip.write_bytes(b"corrupt video and audio")
    with pytest.raises(native.NativeStoryboardError) as error:
        native.render_native([str(clip)], [native.NativeShotPlan(7, "prompt", 5, 5, "remote-7")],
                             str(tmp_path / "final.mp4"), aspect="16:9")
    assert error.value.stage == "audio_validation"
    assert "shot 7" in str(error.value) and "remote-7" in str(error.value)
    assert clip.exists() and not (tmp_path / "final.mp4").exists()


def test_corrupt_audio_packets_fail_even_when_video_is_decodable(tmp_path):
    source = make_clip(tmp_path / "valid.mkv", 0.5)
    encoded = tmp_path / "encoded.mkv"
    subprocess.run([utils.get_ffmpeg_binary(), "-y", "-v", "error", "-i", source,
                    "-c:v", "copy", "-c:a", "aac", str(encoded)], check=True, capture_output=True)
    corrupt = tmp_path / "bad-audio.mkv"
    data = bytearray(encoded.read_bytes())
    with av.open(str(encoded)) as original:
        for packet in original.demux():
            if packet.stream.type == "audio" and packet.size:
                position = data.find(bytes(packet))
                assert position >= 0
                data[position:position + packet.size] = b"\xff" * packet.size
    corrupt.write_bytes(data)
    with av.open(str(corrupt)) as media:
        assert list(media.decode(video=0))  # Only its audio was damaged.
    with pytest.raises(native.NativeStoryboardError, match="shot 1"):
        native.render_native([str(corrupt)], [native.NativeShotPlan(1, "prompt", 5, 5, "paid-1")],
                             str(tmp_path / "final.mp4"), aspect="16:9")
    assert corrupt.exists()


def test_real_local_render_failure_resumes_without_submitting_again(tmp_path, board_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (160, 90))
    board = parse_native_storyboard(board_path)
    sources = [make_clip(tmp_path / f"original-{i}.mkv", duration) for i, duration in enumerate([0.3, 0.4, 0.5])]
    paid = []

    def generate(**kwargs):
        index = len(paid)
        paid.append(kwargs)
        kwargs["on_task_created"](f"paid-{index}")
        return [(sources[index], [0.3, 0.4, 0.5][index], {}, f"paid-{index}")]

    # Store clips within an accepted material artifact root for checkpoint reuse.
    monkeypatch.setattr(utils, "storage_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    executor = video._run_concat_with_heartbeat
    monkeypatch.setattr(video, "_run_concat_with_heartbeat", Mock(side_effect=OSError("local encoder unavailable")))
    options = dict(task_id="render-resume", params=VideoParams(video_subject="native", video_source="wan3", video_aspect="16:9"),
                   storyboard_data=board, video_run_config=run_config(), storyboard_audio_mode="native")
    failed = cp.start(**options)
    assert failed["failed_stage"] == "combine" and len(paid) == 3
    monkeypatch.setattr(video, "_run_concat_with_heartbeat", executor)
    resumed = cp.start(**options, resume=True)
    assert len(paid) == 3 and resumed["videos"]
    assert native.inspect_clip(resumed["videos"][0])["duration"] == pytest.approx(1.2, abs=0.04)


@pytest.mark.parametrize("failure", ["audio", "render", "none"])
def test_native_resume_reuses_materials_and_render_without_tts_or_rebilling(tmp_path, board_path, monkeypatch, failure):
    board = parse_native_storyboard(board_path)
    params = VideoParams(video_subject="native", video_source="wan3", video_aspect="16:9",
                         video_script=board.video_script, match_materials_to_script=True)
    monkeypatch.setattr(utils, "check_ffmpeg_ready", lambda: True)
    forbidden = Mock(side_effect=AssertionError("native path called legacy audio/LLM/subtitle/BGM"))
    for module, name in [(task, "generate_audio"), (task, "generate_script"), (task, "generate_terms"),
                         (task, "generate_subtitle"), (video, "get_bgm_file"), (task, "generate_final_videos")]:
        monkeypatch.setattr(module, name, forbidden)
    paid = []

    def generate(**kwargs):
        paid.append(kwargs)
        saved = Path(utils.storage_dir("cache_videos", create=True)) / f"{len(paid)}.mkv"
        saved.write_bytes(b"downloaded original")
        kwargs["on_task_created"](f"remote-{len(paid)}")
        return [(str(saved), [2, 7, 1][len(paid) - 1], {}, f"remote-{len(paid)}")]

    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    renders = []
    reject = failure != "none"

    def render(clips, plans, output, **kwargs):
        renders.append(1)
        if reject:
            raise native.NativeStoryboardError("local failure", stage="audio_validation" if failure == "audio" else "combine",
                                               shot=2, remote_id=plans[1].remote_task_id)
        Path(output).write_bytes(b"final mp4")
        return {"videos": [output], "combined_videos": [], "actual_seconds": [2.1, 7.1, 1.1], "output_seconds": 10.3}

    monkeypatch.setattr(native, "render_native", render)
    options = dict(task_id="native-task", params=params, storyboard_data=board,
                   video_run_config=run_config(), storyboard_audio_mode="native")
    first = cp.start(**options)
    checkpoint = json.loads((Path(utils.task_dir("native-task")) / "storyboard-checkpoint.json").read_text())
    assert len(paid) == 3
    assert all(record["status"] == "completed" for record in checkpoint["clips"].values())
    if failure != "none":
        assert first["state"] == task.const.TASK_STATE_FAILED
        second = cp.start(**options, resume=True)
        assert second["state"] == task.const.TASK_STATE_FAILED
        assert len(paid) == 3  # Missing audio stays a completed download, never a failed submission.
        reject = False
    final = cp.start(**options, resume=True)
    assert final["videos"] and len(paid) == 3
    before = len(renders)
    cp.start(**options, resume=True)
    assert len(renders) == before  # Existing render checkpoint is serializable and reused.
    forbidden.assert_not_called()
    script = (Path(utils.task_dir("native-task")) / "script.json").read_text()
    assert "test-secret" not in script
    diagnostics = json.loads(script)
    assert diagnostics["storyboard_audio_mode"] == "native"
    assert diagnostics["actual_total_seconds"] == pytest.approx(10.3)
    assert [shot["actual_seconds"] for shot in diagnostics["native_shots"]] == [2.1, 7.1, 1.1]
    assert [shot["request_seconds"] for shot in diagnostics["native_shots"]] == [5, 5, 4]


def test_legacy_checkpoint_identity_is_unchanged_and_native_inputs_are_fenced(tmp_path, board_path, monkeypatch):
    params = VideoParams(video_subject="old", subtitle_enabled=False)
    board = parse_storyboard(board_path, require_shot_timing=True)
    old_identity = cp.fingerprint({"params": params.model_dump(), "storyboard": cp._pack(board), "video": run_config().summary()})
    monkeypatch.setattr(task, "start", lambda **kwargs: {})
    cp.start(task_id="old", params=params, storyboard_data=board, video_run_config=run_config())
    checkpoint = json.loads((Path(utils.task_dir("old")) / "storyboard-checkpoint.json").read_text())
    assert checkpoint["identity"] == old_identity
    options = dict(task_id="new", params=params, storyboard_data=parse_native_storyboard(board_path),
                   video_run_config=run_config(), storyboard_audio_mode="native")
    cp.start(**options)
    old_board = cp._pack(options["storyboard_data"])
    del old_board["fields"]["resolution"]
    plans = native.build_generation_plan(options["storyboard_data"], options["video_run_config"])
    old_identity = cp.fingerprint({"params": params.model_dump(), "storyboard": old_board,
                                   "video": options["video_run_config"].summary(), "storyboard_audio_mode": "native",
                                   "native_requests": [[plan.prompt, plan.request_seconds] for plan in plans]})
    checkpoint = json.loads((Path(utils.task_dir("new")) / "storyboard-checkpoint.json").read_text())
    assert checkpoint["identity"] == old_identity
    cp.start(**options, resume=True)
    changed = replace(options["storyboard_data"], narration_voice="A different voice")
    with pytest.raises(cp.CheckpointError, match="inputs changed"):
        cp.start(**(options | {"storyboard_data": changed}), resume=True)
    changed_music = replace(options["storyboard_data"], music_mood="A new whole-video score")
    with pytest.raises(cp.CheckpointError, match="inputs changed"):
        cp.start(**(options | {"storyboard_data": changed_music}), resume=True)
