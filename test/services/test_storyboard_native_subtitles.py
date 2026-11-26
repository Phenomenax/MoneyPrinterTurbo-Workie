"""Tests for native post-production faster-whisper ASR and FFmpeg ASS subtitle burn-in."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import pytest

from app.models.schema import VideoAspect, VideoParams
from app.services import (
    material,
    storyboard_checkpoint as cp,
    storyboard_native as native,
    storyboard_native_subtitles as native_sub,
    storyboard_revision as revision,
    subtitle as subtitle_service,
    task,
    video_run_config as vrc,
    wan3,
)
from app.services.state import MemoryState
from app.services.storyboard import parse_native_storyboard
from app.utils import utils

SPOKEN_BOARD = """# Storyboard — Native subtitles test

**Aspect:** 9:16

## Beats (timed)

| Time | Shot type | On-screen / B-roll | VO / on-camera line | Music / SFX |
| --- | --- | --- | --- | --- |
| 0–4 | Wide | Morning oak table with white ceramic mug. | 清晨的手冲咖啡，从轻柔注水开始。 | Soft guitar. |
| 4–8 | Close | Steam rising silently from the mug. | — | Soft guitar. |
| 8–12 | Detail | Warm amber coffee dripping into the cup. | 明早也试着用一杯手冲开启一天。 | Gentle fade out. |

## Narration voice
Mandarin female narrator, warm and clear.
"""


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MPT_HOME", str(tmp_path))
    monkeypatch.setattr(task.sm, "state", MemoryState())
    monkeypatch.setattr(subtitle_service, "model", None)
    monkeypatch.setattr(subtitle_service, "_loaded_device", None)
    monkeypatch.setattr(
        wan3.requests,
        "post",
        Mock(side_effect=AssertionError("unexpected paid request")),
    )
    monkeypatch.setattr(
        wan3.requests,
        "get",
        Mock(side_effect=AssertionError("unexpected network request")),
    )


def make_run_config(**overrides):
    settings = dict(
        source="wan3",
        model="wan3.0-video",
        model_family="wan3",
        api_key="test-secret",
        endpoint="https://gateway.invalid",
        resolution="720P",
        min_duration=2,
        max_duration=30,
        concurrency=1,
    )
    return vrc.VideoRunConfig(**(settings | overrides))


def make_clip(target: Path, duration: float, color: str = "black", frequency: int = 440, size: str = "360x640") -> str:
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:r=30:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return str(target)


def install_fake_local_model(tmp_path: Path, model_size: str = "large-v3-turbo") -> Path:
    model_dir = tmp_path / "models" / f"whisper-{model_size}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.bin").write_bytes(b"fake-ct2-weights")
    return model_dir


def test_whisper_auto_runtime_and_cuda_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(subtitle_service, "_has_cuda", lambda: True)
    assert subtitle_service.resolve_whisper_runtime("auto", "auto") == ("cuda", "float16")
    assert subtitle_service.resolve_whisper_runtime("cpu", "auto") == ("cpu", "int8")

    monkeypatch.setattr(subtitle_service, "_has_cuda", lambda: False)
    assert subtitle_service.resolve_whisper_runtime("auto", "auto") == ("cpu", "int8")
    assert subtitle_service.resolve_whisper_runtime("auto", "float16") == ("cpu", "int8")
    assert subtitle_service.resolve_whisper_runtime("auto", "int8_float16") == ("cpu", "int8")

    install_fake_local_model(tmp_path, "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "model_size", "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "device", "auto")
    monkeypatch.setattr(subtitle_service, "compute_type", "auto")
    monkeypatch.setattr(subtitle_service, "_has_cuda", lambda: True)

    loaded_devices = []

    class FakeCudaThenCpuModel:
        def __init__(self, *, model_size_or_path, device, compute_type):
            self.device = device
            loaded_devices.append((device, compute_type))

        def transcribe(self, audio_file, **kwargs):
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            word = SimpleNamespace(start=0.1, end=0.5, word="测试。")
            segment = SimpleNamespace(start=0.1, end=0.5, text="测试。", words=[word])
            return iter([segment]), SimpleNamespace(language="zh", language_probability=0.99)

    monkeypatch.setattr(subtitle_service, "WhisperModel", FakeCudaThenCpuModel)
    segments, info = subtitle_service.transcribe_audio("clip.wav", initial_prompt="提示", local_files_only=True)
    assert loaded_devices == [("cuda", "float16"), ("cpu", "int8")]
    assert len(segments) == 1 and info.language == "zh"
    assert subtitle_service._ensure_model_loaded(local_files_only=True, force_cpu=True) is True
    assert loaded_devices == [("cuda", "float16"), ("cpu", "int8")]


def test_register_cuda_dll_directories_exposes_nvidia_wheel_bins(tmp_path, monkeypatch):
    cublas_bin = tmp_path / "nvidia" / "cublas" / "bin"
    cudnn_bin = tmp_path / "nvidia" / "cudnn" / "bin"
    cublas_bin.mkdir(parents=True)
    cudnn_bin.mkdir(parents=True)

    specs = {
        "nvidia.cublas": SimpleNamespace(submodule_search_locations=[str(cublas_bin.parent)]),
        "nvidia.cudnn": SimpleNamespace(submodule_search_locations=[str(cudnn_bin.parent)]),
    }
    registered_handles = []
    registered_dirs = set()
    added_via_os = []

    monkeypatch.setattr(subtitle_service.os, "name", "nt")
    monkeypatch.setattr(
        subtitle_service.importlib.util,
        "find_spec",
        lambda name: specs.get(name),
    )
    monkeypatch.setattr(subtitle_service, "_CUDA_DLL_DIR_HANDLES", registered_handles)
    monkeypatch.setattr(subtitle_service, "_CUDA_DLL_DIRS_REGISTERED", registered_dirs)
    monkeypatch.setattr(
        subtitle_service.os,
        "add_dll_directory",
        lambda path: added_via_os.append(path) or f"handle:{path}",
        raising=False,
    )
    monkeypatch.setenv("PATH", r"C:\Windows\System32")

    added = subtitle_service._register_cuda_dll_directories()
    assert added == [str(cublas_bin.resolve()), str(cudnn_bin.resolve())]
    assert added_via_os == added
    assert len(registered_handles) == 2
    path_entries = subtitle_service.os.environ["PATH"].split(subtitle_service.os.pathsep)
    assert path_entries[:2] == added
    assert subtitle_service._register_cuda_dll_directories() == []


def test_subtitle_create_supports_per_call_initial_prompt_override(tmp_path, monkeypatch):
    recorded = []

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def transcribe(self, _audio_file, **kwargs):
            recorded.append(kwargs.get("initial_prompt"))
            word = SimpleNamespace(start=0.0, end=0.5, word="你好。")
            segment = SimpleNamespace(start=0.0, end=0.5, text="你好。", words=[word])
            return [segment], SimpleNamespace(language="zh", language_probability=0.98)

    monkeypatch.setattr(subtitle_service, "WhisperModel", FakeModel)
    monkeypatch.setattr(subtitle_service, "initial_prompt", "global-prompt")
    subtitle_service.create("a.wav", str(tmp_path / "one.srt"), initial_prompt="shot-prompt")
    subtitle_service.create("b.wav", str(tmp_path / "two.srt"))
    assert recorded == ["shot-prompt", "global-prompt"]
    assert subtitle_service.initial_prompt == "global-prompt"


def test_preflight_blocks_missing_whisper_model_before_any_paid_call(tmp_path, monkeypatch):
    board_file = tmp_path / "board.md"
    board_file.write_text(SPOKEN_BOARD, encoding="utf-8")
    board = parse_native_storyboard(board_file)
    monkeypatch.setattr(subtitle_service, "find_local_model_path", lambda *_a, **_kw: None)
    paid = Mock(side_effect=AssertionError("paid generation should not run when whisper model is missing"))
    monkeypatch.setattr(material, "_generate_wan3_materials", paid)

    result = cp.start(
        task_id="missing-whisper",
        params=VideoParams(video_subject="coffee", video_source="wan3", video_aspect="9:16"),
        storyboard_data=board,
        video_run_config=make_run_config(),
        storyboard_audio_mode="native",
    )
    assert result["state"] == task.const.TASK_STATE_FAILED
    assert result["failed_stage"] == "preflight"
    assert "whisper-large-v3-turbo" in result["error"]
    paid.assert_not_called()


def test_wrap_subtitle_text_and_ass_generation():
    wrapped_portrait = native_sub.wrap_subtitle_text(
        "清晨的手冲咖啡从轻柔的注水开始唤醒早晨", aspect="9:16", font_size=60
    )
    lines = wrapped_portrait.splitlines()
    assert len(lines) == 2
    assert abs(len(lines[0]) - len(lines[1])) <= 4

    wrapped_landscape = native_sub.wrap_subtitle_text(
        "清晨的手冲咖啡从轻柔注水开始", aspect="16:9", font_size=60
    )
    assert "\n" not in wrapped_landscape

    mixed = native_sub.wrap_subtitle_text(
        "Welcome to MoneyPrinterTurbo studio workflow test", aspect="9:16", font_size=60
    )
    assert "MoneyPrinterTurbo" in mixed
    assert native_sub._clean_cue_text("注意：") == "注意"

    params = VideoParams(
        video_subject="style",
        subtitle_position="custom",
        custom_position=75.0,
        font_size=54,
        text_fore_color="#112233",
        stroke_color="#445566",
        stroke_width=2.0,
        text_background_color="#102030",
    )
    ass_text = native_sub.build_ass_content(
        [native_sub.SubtitleCue(start=1.25, end=3.5, text="第一行\n{测试}")],
        params=params,
        width=1080,
        height=1920,
        font_family="Heiti TC",
        font_bold=True,
    )
    assert "PlayResX: 1080" in ass_text and "PlayResY: 1920" in ass_text
    assert "&H00332211" in ass_text
    assert "&H00302010" in ass_text
    assert ",3," in ass_text
    assert r"{\an5\pos(540,1440)}第一行\N\{测试\}" in ass_text


def test_generate_native_subtitles_skips_silent_shots_clamps_and_preserves_recognized_text(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (360, 640))
    board_file = tmp_path / "board.md"
    board_file.write_text(SPOKEN_BOARD, encoding="utf-8")
    board = parse_native_storyboard(board_file)
    plans = native.build_generation_plan(board, make_run_config())

    clips = [
        make_clip(tmp_path / "shot1.mkv", 0.5, frequency=440),
        make_clip(tmp_path / "shot2.mkv", 0.5, frequency=660),
        make_clip(tmp_path / "shot3.mkv", 0.5, frequency=880),
    ]
    clean_video = tmp_path / "clean.mp4"
    stitched = native.render_native(clips, plans, str(clean_video), aspect="9:16")

    prompts_seen = []

    def fake_transcribe(audio_file, **kwargs):
        assert Path(audio_file).is_file()
        assert kwargs["local_files_only"] is True
        assert kwargs["vad_filter"] is True
        assert kwargs["word_timestamps"] is True
        prompts_seen.append(kwargs["initial_prompt"])
        if len(prompts_seen) == 1:
            words = [
                SimpleNamespace(start=0.05, end=0.35, word="今早的手冲咖啡，"),
                SimpleNamespace(start=0.35, end=0.90, word="从轻柔注水开始。"),
            ]
            return [SimpleNamespace(start=0.05, end=0.90, text="今早的手冲咖啡，从轻柔注水开始。", words=words)], SimpleNamespace(language="zh", language_probability=0.99)
        words = [SimpleNamespace(start=0.10, end=0.80, word="明早也来一杯手冲。")]
        return [SimpleNamespace(start=0.10, end=0.80, text="明早也来一杯手冲。", words=words)], SimpleNamespace(language="zh", language_probability=0.99)

    monkeypatch.setattr(subtitle_service, "transcribe_audio", fake_transcribe)
    srt_path = tmp_path / "subtitle.srt"
    params = VideoParams(video_subject="coffee", video_aspect="9:16", font_size=36)
    result = native_sub.generate_native_subtitles(
        clean_video, plans, stitched["frame_counts"], srt_path, params=params, fps=native.FPS
    )
    assert result["subtitle_path"] == str(srt_path.resolve())
    assert prompts_seen == [plans[0].narration, plans[2].narration]

    cues = native_sub.load_srt_cues(srt_path)
    assert len(cues) == 3
    full_text = "".join(cue.text.replace("\n", "") for cue in cues)
    assert "今早的手冲咖啡" in full_text
    assert "清晨的手冲咖啡" not in full_text
    shot1_end = stitched["frame_counts"][0] / native.FPS
    shot3_start = (stitched["frame_counts"][0] + stitched["frame_counts"][1]) / native.FPS
    shot3_end = sum(stitched["frame_counts"]) / native.FPS
    assert cues[0].end <= shot1_end + 1e-3
    assert cues[1].end <= shot1_end + 1e-3
    assert shot3_start - 1e-3 <= cues[2].start < cues[2].end <= shot3_end + 1e-3


def test_end_to_end_offline_render_burn_in_and_no_subtitle_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (360, 640))
    install_fake_local_model(tmp_path, "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "model_size", "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "WhisperModel", object)

    board_file = tmp_path / "board.md"
    board_file.write_text(SPOKEN_BOARD, encoding="utf-8")
    board = parse_native_storyboard(board_file)
    cache_dir = Path(utils.storage_dir("cache_videos", create=True))

    def fake_generate(**kwargs):
        index = int(kwargs.get("task_id", "0")[-1]) if str(kwargs.get("task_id", "")).endswith(("1", "2", "3")) else len(list(cache_dir.glob("*.mkv"))) + 1
        remote = f"remote-{index}"
        kwargs["on_task_created"](remote)
        clip_path = make_clip(cache_dir / f"{remote}.mkv", 0.5, color="black", frequency=440 + index * 100)
        return [(clip_path, 0.5, {}, remote)]

    monkeypatch.setattr(material, "_generate_wan3_materials", fake_generate)

    def fake_transcribe(_audio_file, **kwargs):
        text = "清晨手冲咖啡测试字幕" if "清晨" in kwargs["initial_prompt"] else "明早试一杯手冲"
        word = SimpleNamespace(start=0.05, end=0.45, word=text)
        return [SimpleNamespace(start=0.05, end=0.45, text=text, words=[word])], SimpleNamespace(language="zh", language_probability=0.99)

    monkeypatch.setattr(subtitle_service, "transcribe_audio", fake_transcribe)
    params = VideoParams(
        video_subject="coffee",
        video_source="wan3",
        video_aspect="9:16",
        subtitle_enabled=True,
        font_name="STHeitiMedium.ttc",
        font_size=36,
    )
    result = cp.start(
        task_id="e2e-subtitled",
        params=params,
        storyboard_data=board,
        video_run_config=make_run_config(),
        storyboard_audio_mode="native",
    )
    assert result["state"] == task.const.TASK_STATE_COMPLETE
    assert Path(result["subtitle_path"]).is_file()
    final_video = Path(result["videos"][0])
    assert native.inspect_clip(final_video)["audio_samples"] > 0
    with av.open(str(final_video)) as container:
        max_brightness = max(int(frame.to_ndarray(format="rgb24").max()) for frame in container.decode(video=0))
    assert max_brightness > 180

    manifest = json.loads((Path(utils.task_dir("e2e-subtitled")) / revision.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["final_sha256"] == cp.digest(final_video)
    saved_raw = Path(utils.task_dir("e2e-subtitled")) / manifest["shots"][0]["path"]
    with av.open(str(saved_raw)) as raw_container:
        raw_max = max(int(frame.to_ndarray(format="rgb24").max()) for frame in raw_container.decode(video=0))
    assert raw_max < 25

    forbidden_asr = Mock(side_effect=AssertionError("ASR must not run when subtitles are disabled"))
    monkeypatch.setattr(subtitle_service, "transcribe_audio", forbidden_asr)
    clean_result = cp.start(
        task_id="e2e-nosub",
        params=VideoParams(video_subject="coffee", video_source="wan3", video_aspect="9:16", subtitle_enabled=False),
        storyboard_data=board,
        video_run_config=make_run_config(),
        storyboard_audio_mode="native",
    )
    assert clean_result["state"] == task.const.TASK_STATE_COMPLETE
    assert clean_result["subtitle_path"] == ""
    assert not (Path(utils.task_dir("e2e-nosub")) / "subtitle.srt").exists()
    with av.open(str(clean_result["videos"][0])) as clean_container:
        clean_max = max(int(frame.to_ndarray(format="rgb24").max()) for frame in clean_container.decode(video=0))
    assert clean_max < 25
    forbidden_asr.assert_not_called()


def test_whisper_failure_preserves_materials_and_resumes_without_rebilling(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (360, 640))
    install_fake_local_model(tmp_path, "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "model_size", "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "WhisperModel", object)

    board_file = tmp_path / "board.md"
    board_file.write_text(SPOKEN_BOARD, encoding="utf-8")
    board = parse_native_storyboard(board_file)
    cache_dir = Path(utils.storage_dir("cache_videos", create=True))
    paid_calls = []

    def fake_generate(**kwargs):
        paid_calls.append(kwargs["prompt"])
        remote = f"paid-{len(paid_calls)}"
        kwargs["on_task_created"](remote)
        clip_path = make_clip(cache_dir / f"{remote}.mkv", 0.4, frequency=400 + len(paid_calls) * 80)
        return [(clip_path, 0.4, {}, remote)]

    monkeypatch.setattr(material, "_generate_wan3_materials", fake_generate)
    should_fail = True
    asr_calls = []
    render_calls = []
    real_render_native = native.render_native

    def tracked_render_native(*args, **kwargs):
        render_calls.append(1)
        return real_render_native(*args, **kwargs)

    monkeypatch.setattr(native, "render_native", tracked_render_native)

    def flaky_transcribe(_audio_file, **kwargs):
        asr_calls.append(kwargs["initial_prompt"])
        if should_fail:
            raise RuntimeError("whisper decoder error")
        word = SimpleNamespace(start=0.05, end=0.35, word="测试台词。")
        return [SimpleNamespace(start=0.05, end=0.35, text="测试台词。", words=[word])], SimpleNamespace(language="zh", language_probability=0.99)

    monkeypatch.setattr(subtitle_service, "transcribe_audio", flaky_transcribe)
    options = dict(
        task_id="whisper-resume",
        params=VideoParams(
            video_subject="coffee",
            video_source="wan3",
            video_aspect="9:16",
            subtitle_enabled=True,
            font_name="STHeitiMedium.ttc",
            font_size=36,
        ),
        storyboard_data=board,
        video_run_config=make_run_config(),
        storyboard_audio_mode="native",
    )
    failed = cp.start(**options)
    assert failed["state"] == task.const.TASK_STATE_FAILED
    assert failed["failed_stage"] == "subtitles"
    assert len(paid_calls) == 3
    assert len(render_calls) == 1
    assert not (Path(utils.task_dir("whisper-resume")) / "final-1.mp4").exists()

    should_fail = False
    resumed = cp.start(**options, resume=True)
    assert resumed["state"] == task.const.TASK_STATE_COMPLETE
    assert len(paid_calls) == 3
    assert len(render_calls) == 1
    assert Path(resumed["subtitle_path"]).is_file()
    assert Path(resumed["videos"][0]).is_file()

    asr_before = len(asr_calls)
    reused = cp.start(**options, resume=True)
    assert reused["state"] == task.const.TASK_STATE_COMPLETE
    assert len(paid_calls) == 3
    assert len(asr_calls) == asr_before


def test_single_shot_revision_reuses_clean_clips_and_rebuilds_subtitled_video(tmp_path, monkeypatch):
    monkeypatch.setattr(VideoAspect, "to_resolution", lambda self: (360, 640))
    install_fake_local_model(tmp_path, "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "model_size", "large-v3-turbo")
    monkeypatch.setattr(subtitle_service, "WhisperModel", object)

    board_file = tmp_path / "board.md"
    board_file.write_text(SPOKEN_BOARD, encoding="utf-8")
    board = parse_native_storyboard(board_file)
    cache_dir = Path(utils.storage_dir("cache_videos", create=True))
    paid_calls = []

    def fake_generate(**kwargs):
        paid_calls.append(kwargs["prompt"])
        remote = f"paid-{len(paid_calls)}"
        kwargs["on_task_created"](remote)
        clip_path = make_clip(cache_dir / f"{remote}.mkv", 0.4, frequency=400 + len(paid_calls) * 60)
        return [(clip_path, 0.4, {}, remote)]

    monkeypatch.setattr(material, "_generate_wan3_materials", fake_generate)

    def fake_transcribe(_audio_file, **kwargs):
        spoken = kwargs["initial_prompt"]
        word = SimpleNamespace(start=0.05, end=0.35, word=spoken)
        return [SimpleNamespace(start=0.05, end=0.35, text=spoken, words=[word])], SimpleNamespace(language="zh", language_probability=0.99)

    monkeypatch.setattr(subtitle_service, "transcribe_audio", fake_transcribe)
    cfg = make_run_config()
    params = VideoParams(
        video_subject="coffee",
        video_source="wan3",
        video_aspect="9:16",
        subtitle_enabled=True,
        font_name="STHeitiMedium.ttc",
        font_size=36,
    )
    parent = cp.start(
        task_id="rev-parent",
        params=params,
        storyboard_data=board,
        video_run_config=cfg,
        storyboard_audio_mode="native",
    )
    assert len(paid_calls) == 3
    parent_root = Path(utils.task_dir("rev-parent"))
    parent_manifest = json.loads((parent_root / revision.MANIFEST).read_text(encoding="utf-8"))
    source = dict(task_id="rev-parent", final_sha256=cp.digest(parent["videos"][0]))

    updated_text = SPOKEN_BOARD.replace(
        "明早也试着用一杯手冲开启一天。",
        "明早用一杯冰手冲开启活力一天。",
    )
    board_file.write_text(updated_text, encoding="utf-8")
    updated_board = parse_native_storyboard(board_file)
    frozen_plan, _ = revision.preflight(
        parent_root.parent, source, updated_board, cfg, "9:16", "partial", [3]
    )
    assert [row["action"] for row in frozen_plan["shots"]] == ["reuse", "reuse", "generate"]

    child = cp.start(
        task_id="rev-child",
        params=params,
        storyboard_data=updated_board,
        video_run_config=cfg,
        storyboard_audio_mode="native",
        revision_plan=frozen_plan,
    )
    assert child["state"] == task.const.TASK_STATE_COMPLETE
    assert len(paid_calls) == 4
    assert "明早用一杯冰手冲开启活力一天" in Path(child["subtitle_path"]).read_text(encoding="utf-8").replace("\n", "")
    child_manifest = json.loads((Path(utils.task_dir("rev-child")) / revision.MANIFEST).read_text(encoding="utf-8"))
    assert child_manifest["final_sha256"] == cp.digest(child["videos"][0])
    assert child_manifest["shots"][0]["sha256"] == parent_manifest["shots"][0]["sha256"]
    assert child_manifest["shots"][1]["sha256"] == parent_manifest["shots"][1]["sha256"]
    assert child_manifest["shots"][2]["sha256"] != parent_manifest["shots"][2]["sha256"]
