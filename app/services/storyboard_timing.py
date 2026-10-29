"""Per-shot narration timing for a storyboard render.

Each spoken shot is synthesized once. Its PCM sample count is the shot length.
Silent shots keep their planned duration. Picture requests are derived from those
lengths and checked against the selected material adapter's limits.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import wave
from dataclasses import dataclass

from loguru import logger

from app.services import voice
from app.utils import utils


PCM_SAMPLE_RATE = 24000
OUTPUT_FPS = 30
FRAME_SECONDS = 1.0 / OUTPUT_FPS
TIMELINE_VERSION = 1
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class StoryboardTimingError(Exception):
    """A storyboard timing failure that must stop before or during one stage."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)


@dataclass
class ShotTimeline:
    index: int
    prompt: str
    narration: str
    planned_start_seconds: float
    planned_end_seconds: float
    planned_duration_seconds: float
    estimated_seconds: float | None
    sample_count: int
    actual_seconds: float
    calibrated_start_seconds: float
    calibrated_end_seconds: float
    target_playback_seconds: float
    audio_path: str
    tts_path: str | None
    sub_maker: object = None
    request_seconds: int | None = None
    required_source_seconds: float | None = None
    material_seconds: float | None = None
    remote_task_id: str | None = None
    material_path: str | None = None


@dataclass
class StoryboardTimeline:
    shots: list[ShotTimeline]
    audio_path: str
    audio_samples: int
    planned_total_seconds: float
    calibrated_total_seconds: float
    duration_mode: str = "automatic"
    explicit_clip_duration: int | None = None
    clip_speed: float = 1.0
    version: int = TIMELINE_VERSION

    def script_updates(self) -> dict:
        return {
            "storyboard_timeline_version": self.version,
            "storyboard_timeline": {
                "version": self.version,
                "duration_mode": self.duration_mode,
                "explicit_clip_duration": self.explicit_clip_duration,
                "clip_speed": self.clip_speed,
                "planned_total_seconds": self.planned_total_seconds,
                "calibrated_total_seconds": self.calibrated_total_seconds,
                "audio_samples": self.audio_samples,
                "audio_file": os.path.basename(self.audio_path) if self.audio_path else "",
                "shots": [_shot_record(shot) for shot in self.shots],
            },
        }


def estimate_narration_seconds(text: str, voice_rate: float) -> float:
    """Rough speech length for diagnostics; measured audio sizes the footage."""
    rate = float(voice_rate) if _positive_finite(voice_rate) else 1.0
    speech_parts: list[str] = []
    pause_seconds = 0.0
    if text and utils.has_pause_tags(text):
        for kind, value in utils.parse_script_with_pauses(text):
            if kind == "pause":
                pause_seconds += float(value)
            else:
                speech_parts.append(str(value))
    else:
        speech_parts.append(text or "")
    speech_seconds = _estimate_speech_seconds("".join(speech_parts))
    return speech_seconds / rate + pause_seconds


def synthesize_timeline(
    *,
    task_dir: str,
    shots,
    voice_name: str,
    voice_rate: float,
    voxcpm_reference_audio: bytes | None = None,
    voxcpm_prompt_audio: bytes | None = None,
    voxcpm_prompt_text: str = "",
) -> StoryboardTimeline:
    """Synthesize each spoken shot once and concatenate exact PCM lengths."""
    if not shots:
        raise StoryboardTimingError("audio", "storyboard has no shots to time")
    storyboard_dir = os.path.join(task_dir, "storyboard")
    os.makedirs(storyboard_dir, exist_ok=True)
    parsed_voice = voice.parse_voice_name(voice_name)
    timed: list[ShotTimeline] = []
    sample_offset = 0
    for shot in shots:
        narration = (shot.narration or "").strip()
        estimate = estimate_narration_seconds(narration, voice_rate) if narration else None
        stem = f"shot-{shot.index:04d}"
        if narration:
            tts_path = os.path.join(storyboard_dir, f"{stem}.mp3")
            tts_kwargs = {
                "text": narration,
                "voice_name": parsed_voice,
                "voice_rate": voice_rate,
                "voice_file": tts_path,
            }
            if voxcpm_reference_audio is not None:
                tts_kwargs["voxcpm_reference_audio"] = voxcpm_reference_audio
            if voxcpm_prompt_audio is not None:
                tts_kwargs["voxcpm_prompt_audio"] = voxcpm_prompt_audio
                tts_kwargs["voxcpm_prompt_text"] = voxcpm_prompt_text
            try:
                sub_maker = voice.tts(**tts_kwargs)
            except Exception as exc:
                raise StoryboardTimingError(
                    "audio",
                    f"shot {shot.index} narration could not be synthesized: {exc}",
                ) from exc
            if sub_maker is None or not os.path.isfile(tts_path) or os.path.getsize(tts_path) == 0:
                raise StoryboardTimingError(
                    "audio",
                    f"shot {shot.index} narration could not be synthesized",
                )
            wav_path = os.path.join(storyboard_dir, f"{stem}.wav")
            try:
                sample_count = decode_to_pcm_wav(tts_path, wav_path)
            except Exception as exc:
                raise StoryboardTimingError(
                    "audio",
                    f"shot {shot.index} narration audio could not be decoded: {exc}",
                ) from exc
        else:
            sub_maker = None
            tts_path = None
            wav_path = os.path.join(storyboard_dir, f"{stem}.wav")
            planned = float(shot.planned_end_seconds) - float(shot.planned_start_seconds)
            try:
                sample_count = write_silence(planned, wav_path)
            except Exception as exc:
                raise StoryboardTimingError(
                    "audio",
                    f"shot {shot.index} silence could not be created: {exc}",
                ) from exc
        if sample_count <= 0:
            raise StoryboardTimingError(
                "audio",
                f"shot {shot.index} narration audio has no samples",
            )
        actual = sample_count / PCM_SAMPLE_RATE
        if actual + 1e-9 < FRAME_SECONDS:
            raise StoryboardTimingError(
                "audio",
                f"shot {shot.index} is shorter than one frame ({actual:.4f}s) "
                "and cannot be an independent shot",
            )
        start = sample_offset / PCM_SAMPLE_RATE
        sample_offset += sample_count
        end = sample_offset / PCM_SAMPLE_RATE
        start_frame = _frame_index(start)
        end_frame = _frame_index(end)
        if end_frame <= start_frame:
            raise StoryboardTimingError(
                "audio",
                f"shot {shot.index} is shorter than one frame ({actual:.4f}s) "
                "and cannot be an independent shot",
            )
        timed.append(
            ShotTimeline(
                index=int(shot.index),
                prompt=shot.prompt,
                narration=narration,
                planned_start_seconds=float(shot.planned_start_seconds),
                planned_end_seconds=float(shot.planned_end_seconds),
                planned_duration_seconds=float(shot.planned_end_seconds) - float(shot.planned_start_seconds),
                estimated_seconds=estimate,
                sample_count=sample_count,
                actual_seconds=actual,
                calibrated_start_seconds=start,
                calibrated_end_seconds=end,
                target_playback_seconds=(end_frame - start_frame) / OUTPUT_FPS,
                audio_path=wav_path,
                tts_path=tts_path,
                sub_maker=sub_maker,
            )
        )
        logger.info(
            "storyboard shot {index}: planned={planned:.3f}s estimated={estimated} "
            "audio={actual:.3f}s samples={samples} start={start:.3f}s end={end:.3f}s".format(
                index=shot.index,
                planned=float(shot.planned_end_seconds) - float(shot.planned_start_seconds),
                estimated="none" if estimate is None else f"{estimate:.3f}s",
                actual=actual,
                samples=sample_count,
                start=start,
                end=end,
            )
        )
    audio_path = os.path.join(task_dir, "audio.wav")
    try:
        total_samples = concat_pcm_wavs([item.audio_path for item in timed], audio_path)
    except Exception as exc:
        raise StoryboardTimingError(
            "audio",
            f"storyboard narration could not be joined: {exc}",
        ) from exc
    if total_samples != sample_offset:
        raise StoryboardTimingError(
            "audio",
            "joined narration sample count does not match the shot timeline",
        )
    planned_total = float(shots[-1].planned_end_seconds) - float(shots[0].planned_start_seconds)
    timeline = StoryboardTimeline(
        shots=timed,
        audio_path=audio_path,
        audio_samples=total_samples,
        planned_total_seconds=planned_total,
        calibrated_total_seconds=total_samples / PCM_SAMPLE_RATE,
    )
    logger.info(
        "storyboard timeline: planned={planned:.3f}s calibrated={actual:.3f}s shots={count}".format(
            planned=timeline.planned_total_seconds,
            actual=timeline.calibrated_total_seconds,
            count=len(timed),
        )
    )
    return timeline


def build_generation_plan(
    timeline: StoryboardTimeline,
    *,
    clip_speed: float,
    explicit_clip_duration: int | None,
    minimum_clip_seconds: int = 1,
    maximum_clip_seconds: int | None = None,
    provider_name: str = "所选视频源",
) -> StoryboardTimeline:
    """Size each source clip using limits supplied by its material adapter."""
    speed = utils.normalize_clip_speed(clip_speed)
    explicit = _explicit_duration(explicit_clip_duration)
    problems: list[str] = []
    for shot in timeline.shots:
        required_source = shot.target_playback_seconds * speed
        shot.required_source_seconds = required_source
        if explicit is None:
            request = max(minimum_clip_seconds, _ceil_seconds(required_source))
        else:
            request = explicit
        shot.request_seconds = request
        if maximum_clip_seconds is not None and request > maximum_clip_seconds:
            problems.append(
                f"shot {shot.index} needs {request}s of {provider_name} video, above the "
                f"{maximum_clip_seconds}s single-clip limit. "
                f"planned={shot.planned_duration_seconds:.3f}s, "
                f"narration={shot.actual_seconds:.3f}s, speed={speed:.2f}x, "
                f"required={request}s. Adjust the shot split or the related settings."
            )
            continue
        if request < minimum_clip_seconds:
            problems.append(
                f"shot {shot.index} {provider_name} request must be at least "
                f"{minimum_clip_seconds} seconds, got {request}s."
            )
            continue
        available_playback = request / speed
        if available_playback + FRAME_SECONDS + 1e-6 < shot.target_playback_seconds:
            minimum = max(minimum_clip_seconds, _ceil_seconds(required_source))
            problems.append(
                f"shot {shot.index} needs at least {minimum}s of source video at "
                f"{speed:.2f}x to cover {shot.target_playback_seconds:.3f}s of playback, "
                f"but --video-clip-duration is {request}s. Raise that value or remove it "
                "to calculate each shot automatically."
            )
        logger.info(
            "storyboard shot {index}: playback={playback:.3f}s source={source:.3f}s "
            "request={request}s mode={mode}".format(
                index=shot.index,
                playback=shot.target_playback_seconds,
                source=required_source,
                request=request,
                mode="explicit" if explicit is not None else "automatic",
            )
        )
    timeline.clip_speed = speed
    timeline.explicit_clip_duration = explicit
    timeline.duration_mode = "explicit" if explicit is not None else "automatic"
    if problems:
        detail = " ".join(problems)
        raise StoryboardTimingError(
            "materials",
            detail + f" 本次尚未提交 {provider_name} 视频生成。",
        )
    return timeline


def shift_subtitle_cues(
    cues: list[tuple[float, float, str]],
    *,
    offset_seconds: float,
    shot_duration_seconds: float,
) -> list[tuple[float, float, str]]:
    """Move one shot's cues onto the PCM timeline and keep them inside that shot."""
    shifted = []
    shot_end = offset_seconds + shot_duration_seconds
    for start, end, text in cues:
        moved_start = start + offset_seconds
        moved_end = end + offset_seconds
        if moved_start >= shot_end - 0.001:
            continue
        moved_end = min(moved_end, shot_end)
        if moved_end <= moved_start:
            continue
        cleaned = (text or "").strip()
        if cleaned:
            shifted.append((moved_start, moved_end, cleaned))
    return shifted


def write_storyboard_subtitles(
    timeline: StoryboardTimeline,
    *,
    subtitle_path: str,
    subtitle_enabled: bool,
    subtitle_provider: str,
    word_level: bool,
) -> str:
    """Build one SRT from per-shot captions. Silent shots contribute nothing."""
    if not subtitle_enabled:
        return ""
    provider = (subtitle_provider or "").strip().lower()
    if not provider:
        logger.info("subtitle provider is empty, skip storyboard subtitles")
        return ""
    cues: list[tuple[float, float, str]] = []
    for shot in timeline.shots:
        if not shot.narration:
            continue
        local_path = os.path.splitext(shot.audio_path)[0] + ".srt"
        _write_local_subtitle(
            shot,
            subtitle_path=local_path,
            provider=provider,
            word_level=word_level,
        )
        local_cues = _read_srt_cues(local_path)
        shot_cues = shift_subtitle_cues(
            local_cues,
            offset_seconds=shot.calibrated_start_seconds,
            shot_duration_seconds=shot.actual_seconds,
        )
        if not shot_cues:
            raise StoryboardTimingError(
                "subtitle",
                f"shot {shot.index} subtitle generation produced no usable cues; "
                "the video was not submitted for generation",
            )
        cues.extend(shot_cues)
    if not cues:
        logger.warning("storyboard subtitle generation produced no cues")
        return ""
    _write_srt(cues, subtitle_path)
    return subtitle_path


def decode_to_pcm_wav(source: str, dest: str) -> int:
    """Decode narration with the same PCM layout used by audio concatenation."""
    existing = _standard_pcm_samples(source)
    if existing is not None:
        if os.path.abspath(source) != os.path.abspath(dest):
            shutil.copyfile(source, dest)
        return existing
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    command = [
        utils.get_ffmpeg_binary(),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        source,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(PCM_SAMPLE_RATE),
        "-codec:a",
        "pcm_s16le",
        dest,
    ]
    result = voice._run_tts_ffmpeg(command, "storyboard narration decode")
    if result is None or result.returncode != 0 or not os.path.isfile(dest):
        detail = ""
        if result is not None:
            detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or "ffmpeg could not decode narration")
    samples = _standard_pcm_samples(dest)
    if samples is None:
        raise RuntimeError("decoded narration is not 24000Hz 16-bit mono PCM")
    return samples


def write_silence(duration_seconds: float, dest: str) -> int:
    samples = int(round(float(duration_seconds) * PCM_SAMPLE_RATE))
    if samples <= 0:
        raise ValueError("silent shot duration must contain at least one sample")
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    with wave.open(dest, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(PCM_SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * samples)
    return samples


def concat_pcm_wavs(paths: list[str], dest: str) -> int:
    """Join PCM shots by sample count, without re-encoding through MP3."""
    if not paths:
        raise ValueError("no narration shots to join")
    if not voice._concat_audio_files(paths, dest):
        raise ValueError("narration shots could not be joined")
    samples = _standard_pcm_samples(dest)
    if samples is None:
        raise ValueError("joined narration is not standard PCM")
    return samples


def _estimate_speech_seconds(speech: str) -> float:
    cjk_count = len(_CJK.findall(speech))
    words = _ENGLISH_WORD.findall(speech)
    other_alpha = [
        char
        for char in speech
        if char.isalpha() and not ("A" <= char <= "Z" or "a" <= char <= "z") and _CJK.fullmatch(char) is None
    ]
    if cjk_count and not other_alpha:
        return cjk_count / 4.0 + len(words) / 2.5
    if words and not cjk_count and not other_alpha:
        return len(words) / 2.5
    compact = re.sub(r"\s+", "", speech)
    if not cjk_count and not words:
        return len(compact) / 4.0
    accounted = cjk_count + sum(len(word) for word in words)
    other_chars = max(len(compact) - accounted, 0)
    return cjk_count / 4.0 + len(words) / 2.5 + other_chars / 4.0


def _positive_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _explicit_duration(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoryboardTimingError(
            "materials",
            "explicit clip duration must be an integer number of seconds. "
            "本次尚未提交视频生成。",
        )
    return value


def _ceil_seconds(value: float) -> int:
    return int(math.ceil(value - 1e-9))


def _frame_index(seconds: float) -> int:
    return int(round(float(seconds) * OUTPUT_FPS))


def _standard_pcm_samples(path: str) -> int | None:
    try:
        with wave.open(path, "rb") as handle:
            if (
                handle.getframerate() != PCM_SAMPLE_RATE
                or handle.getnchannels() != 1
                or handle.getsampwidth() != 2
            ):
                return None
            frames = handle.getnframes()
    except (wave.Error, EOFError, OSError):
        return None
    return frames if frames > 0 else None


def _shot_record(shot: ShotTimeline) -> dict:
    return {
        "index": shot.index,
        "prompt": shot.prompt,
        "narration": shot.narration,
        "planned_start_seconds": shot.planned_start_seconds,
        "planned_end_seconds": shot.planned_end_seconds,
        "planned_duration_seconds": shot.planned_duration_seconds,
        "estimated_seconds": shot.estimated_seconds,
        "sample_count": shot.sample_count,
        "actual_seconds": shot.actual_seconds,
        "calibrated_start_seconds": shot.calibrated_start_seconds,
        "calibrated_end_seconds": shot.calibrated_end_seconds,
        "target_playback_seconds": shot.target_playback_seconds,
        "audio_file": os.path.basename(shot.audio_path) if shot.audio_path else "",
        "tts_file": os.path.basename(shot.tts_path) if shot.tts_path else "",
        "request_seconds": shot.request_seconds,
        "required_source_seconds": shot.required_source_seconds,
        "material_seconds": shot.material_seconds,
        "remote_task_id": shot.remote_task_id or "",
        "material_file": os.path.basename(shot.material_path) if shot.material_path else "",
    }


def _write_local_subtitle(shot: ShotTimeline, *, subtitle_path: str, provider: str, word_level: bool) -> None:
    try:
        if os.path.exists(subtitle_path):
            os.remove(subtitle_path)
    except OSError:
        pass
    if provider == "edge":
        if shot.sub_maker is None:
            logger.warning(f"shot {shot.index} has no subtitle timing, skip its captions")
            return
        voice.create_subtitle(
            text=shot.narration,
            sub_maker=shot.sub_maker,
            subtitle_file=subtitle_path,
            word_level=word_level,
        )
        return
    if provider == "whisper":
        from app.services import subtitle

        subtitle.create(
            audio_file=shot.audio_path,
            subtitle_file=subtitle_path,
            word_level=word_level,
        )
        if not word_level and os.path.isfile(subtitle_path):
            subtitle.correct(subtitle_file=subtitle_path, video_script=shot.narration)
        return
    logger.warning(
        f"storyboard subtitles keep the configured provider {provider!r} "
        f"and skip shot {shot.index}"
    )


_SRT_STAMP = re.compile(r"(\d+):(\d+):(\d+),(\d+)")


def _read_srt_cues(path: str) -> list[tuple[float, float, str]]:
    from app.utils.subtitle_reader import file_to_subtitles

    cues = []
    for _index, timing, text in file_to_subtitles(path):
        stamps = _SRT_STAMP.findall(timing)
        if len(stamps) < 2:
            continue
        start = _stamp_seconds(stamps[0])
        end = _stamp_seconds(stamps[1])
        cues.append((start, end, text))
    return cues


def _stamp_seconds(parts: tuple[str, str, str, str]) -> float:
    hours, minutes, seconds, millis = (int(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(cues: list[tuple[float, float, str]], path: str) -> None:
    blocks = []
    for index, (start, end, text) in enumerate(cues, start=1):
        blocks.append(f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("\n".join(blocks) + "\n")
    os.replace(temporary, path)
