from pathlib import Path
import subprocess
from types import SimpleNamespace

import av
import pytest

from app.services import video


def _create_color_clip(binary, destination, color):
    subprocess.run(
        [
            binary,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c={color}:s=64x64:r=30",
            "-t", "0.2",
            "-vf", video._BT709_VIDEO_FILTER,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            str(destination),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )


def test_normalized_storyboard_clips_use_stream_copy(tmp_path, monkeypatch):
    binary = video.utils.get_ffmpeg_binary()
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    output = tmp_path / "combined.mp4"
    _create_color_clip(binary, first, "red")
    _create_color_clip(binary, second, "blue")
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda: "libx264")

    assert video.concat_video_clips_with_ffmpeg(
        [str(first), str(second)], str(output), 1, str(tmp_path), stream_copy=True
    ) == "copy"

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert stream.average_rate == 30
        assert len(frames) == 12
        before_boundary = frames[5].to_ndarray(format="rgb24")[32, 32]
        after_boundary = frames[6].to_ndarray(format="rgb24")[32, 32]
        assert before_boundary[0] > before_boundary[2] * 1.5
        assert after_boundary[2] > after_boundary[0] * 1.5
        assert abs(float(stream.duration * stream.time_base) - 0.4) < 0.02
        context = stream.codec_context
        assert context.color_range == 1
        assert context.color_primaries == 1
        assert context.color_trc == 1
        assert context.colorspace == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "combined.mp4", "first.mp4", "second.mp4"
    ]


def test_stream_copy_command_omits_encoding_filters_and_encoder_options(tmp_path, monkeypatch):
    output = tmp_path / "combined.mp4"
    commands = []

    def run(command, _reported_output):
        commands.append(command)
        Path(command[-1]).write_bytes(b"copied video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video, "_run_concat_with_heartbeat", run)
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda: pytest.fail("copy should not resolve an encoder"))

    assert video.concat_video_clips_with_ffmpeg(
        ["first.mp4", "second.mp4"], str(output), 4, str(tmp_path), stream_copy=True
    ) == "copy"

    command = commands[0]
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "-an" in command
    assert not {"-vf", "-pix_fmt", "-threads"}.intersection(command)
    assert output.read_bytes() == b"copied video"


@pytest.mark.parametrize("failure", ["nonzero", "empty"])
def test_stream_copy_failure_retries_encoding_and_preserves_previous_output(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source.mp4"
    output = tmp_path / "combined.mp4"
    source.write_bytes(b"input")
    output.write_bytes(b"previous complete video")
    commands = []

    def run(command, _reported_output):
        commands.append(command)
        if command[command.index("-c:v") + 1] == "copy":
            if failure == "nonzero":
                Path(command[-1]).write_bytes(b"partial copy")
                return SimpleNamespace(returncode=1, stdout="", stderr="copy unsupported")
            Path(command[-1]).write_bytes(b"")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert output.read_bytes() == b"previous complete video"
        Path(command[-1]).write_bytes(b"encoded video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(video, "_run_concat_with_heartbeat", run)
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda: "libx264")

    assert video.concat_video_clips_with_ffmpeg(
        [str(source)], str(output), 2, str(tmp_path), stream_copy=True
    ) == "libx264"

    assert [command[command.index("-c:v") + 1] for command in commands] == [
        "copy", "libx264"
    ]
    assert output.read_bytes() == b"encoded video"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["combined.mp4", "source.mp4"]


def test_stream_copy_timeout_does_not_retry_or_replace_previous_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "combined.mp4"
    output.write_bytes(b"previous complete video")
    commands = []

    def timeout(command, _reported_output):
        commands.append(command)
        raise TimeoutError("ffmpeg concat exceeded 1 seconds")

    monkeypatch.setattr(video, "_run_concat_with_heartbeat", timeout)
    monkeypatch.setattr(video, "_get_effective_video_codec", lambda: pytest.fail("timeout must not retry"))

    with pytest.raises(TimeoutError, match="exceeded"):
        video.concat_video_clips_with_ffmpeg(
            ["source.mp4"], str(output), 2, str(tmp_path), stream_copy=True
        )

    assert len(commands) == 1
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"
    assert output.read_bytes() == b"previous complete video"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["combined.mp4"]


@pytest.mark.parametrize(
    ("clip_codecs", "expect_stream_copy"),
    [
        (["libx264", "libx264"], True),
        (["libx264", "h264_nvenc"], False),
        (["h264_nvenc", "h264_nvenc"], False),
    ],
)
def test_storyboard_stream_copies_only_when_every_clip_used_libx264(
    tmp_path, monkeypatch, clip_codecs, expect_stream_copy
):
    class FakeClip:
        duration = 1.0
        size = (1080, 1920)

        def subclipped(self, _start, _end):
            return self

        def close(self):
            pass

    write_index = 0

    def write_clip(_clip, output_file, **_kwargs):
        nonlocal write_index
        codec = clip_codecs[write_index]
        write_index += 1
        Path(output_file).write_bytes(b"normalized clip")
        return codec

    concat = SimpleNamespace()

    def capture_concat(**kwargs):
        concat.kwargs = kwargs

    monkeypatch.setattr(video, "_get_configured_video_codec", lambda: "h264_nvenc")
    monkeypatch.setattr(video, "_open_video_clip_quietly", lambda *_args, **_kwargs: FakeClip())
    monkeypatch.setattr(video, "_write_videofile_with_codec_fallback", write_clip)
    monkeypatch.setattr(video, "concat_video_clips_with_ffmpeg", capture_concat)

    plans = [
        SimpleNamespace(index=index, target_playback_seconds=1.0, required_source_seconds=1.0)
        for index in (1, 2)
    ]
    video._combine_storyboard_shots(
        str(tmp_path / "combined.mp4"),
        ["first.mp4", "second.mp4"],
        plans,
        video_aspect=video.VideoAspect.portrait,
        video_transition_mode=None,
        threads=2,
        clip_speed=1.0,
        video_fit_mode=video.VideoFitMode.cover,
        used_video_paths=None,
        progress_callback=None,
    )

    assert concat.kwargs["stream_copy"] is expect_stream_copy
    assert write_index == 2
