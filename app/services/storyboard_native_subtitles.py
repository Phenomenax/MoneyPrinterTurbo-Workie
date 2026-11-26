"""Local faster-whisper per-shot ASR and FFmpeg ASS burn-in for native storyboards."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import ImageColor, ImageFont

from app.config import config
from app.models import const
from app.models.schema import VideoAspect
from app.services import storyboard_native as native
from app.services import subtitle as subtitle_service
from app.utils import file_security, utils
from app.utils.subtitle_reader import file_to_subtitles
from app.utils.subtitle_writer import write_subtitle_file

_DEFAULT_FONT_NAME = "STHeitiMedium.ttc"
_DEFAULT_FFMPEG_TIMEOUT_SECONDS = 3600.0
_STRIP_TRAILING_PUNCTUATION = ",.，。、；;:："
_KEEP_TRAILING_PUNCTUATION = "?!？！…"
_SPLIT_PUNCTUATIONS = tuple(const.PUNCTUATIONS)
_NO_START_CHARS = set(",.，。、；;：:?!？！…)]】）》”’")
_SRT_RANGE_RE = re.compile(
    r"^\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*$"
)


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


def _ffmpeg_timeout_seconds() -> float:
    configured = config.app.get(
        "ffmpeg_concat_timeout_seconds", _DEFAULT_FFMPEG_TIMEOUT_SECONDS
    )
    try:
        value = float(configured)
    except (TypeError, ValueError):
        return _DEFAULT_FFMPEG_TIMEOUT_SECONDS
    if isinstance(configured, bool) or not math.isfinite(value) or value <= 0:
        return _DEFAULT_FFMPEG_TIMEOUT_SECONDS
    return value


def resolve_subtitle_font_path(font_name: str | None) -> Path:
    selected = (font_name or "").strip() or _DEFAULT_FONT_NAME
    try:
        resolved = Path(
            file_security.resolve_path_within_directory(utils.font_dir(), selected)
        )
    except ValueError as exc:
        raise native.NativeStoryboardError(
            f"invalid subtitle font path: {selected}", stage="preflight"
        ) from exc
    if not resolved.is_file():
        raise native.NativeStoryboardError(
            f"subtitle font not found: {selected}", stage="preflight"
        )
    if resolved.suffix.lower() not in {".ttf", ".ttc"}:
        raise native.NativeStoryboardError(
            "subtitle font must use the .ttf or .ttc extension", stage="preflight"
        )
    return resolved


def extract_font_face(font_path: Path) -> tuple[str, bool]:
    try:
        family, style = ImageFont.truetype(str(font_path), 32).getname()
        family = (family or "").strip() or font_path.stem
        is_bold = "bold" in (style or "").lower()
        return family, is_bold
    except Exception:
        return font_path.stem, False


def has_spoken_shots(plans) -> bool:
    return any((getattr(plan, "narration", "") or "").strip() for plan in (plans or ()))


def ensure_native_whisper_ready(params=None, plans=None) -> str | None:
    """Fail before paid generation if native post-production subtitles cannot run locally."""
    if params is not None and not getattr(params, "subtitle_enabled", True):
        return None
    if plans is not None and not has_spoken_shots(plans):
        return None
    if params is not None:
        from app.services import video

        try:
            video.validate_subtitle_colors(params)
        except ValueError as exc:
            raise native.NativeStoryboardError(str(exc), stage="preflight") from exc
        resolve_subtitle_font_path(getattr(params, "font_name", None))
    if subtitle_service.WhisperModel is None:
        raise native.NativeStoryboardError(
            "faster-whisper is not available in MPT; install faster-whisper "
            "or disable subtitles with --no-subtitle-enabled",
            stage="preflight",
        )
    if subtitle_service.model is not None:
        return "loaded"
    local_path = subtitle_service.find_local_model_path(subtitle_service.model_size)
    if not local_path:
        expected_dir = (
            Path(utils.root_dir()) / "models" / f"whisper-{subtitle_service.model_size}"
        )
        raise native.NativeStoryboardError(
            f"local Whisper model '{subtitle_service.model_size}' is missing; "
            f"place the CTranslate2 model files (including model.bin) in {expected_dir} "
            "or disable subtitles with --no-subtitle-enabled",
            stage="preflight",
        )
    return local_path


def _char_units(char: str) -> float:
    if char.isspace():
        return 0.35
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return 1.0
    if char in ".,:;!?'\"()[]{}-":
        return 0.4
    return 0.55


def _text_units(text: str) -> float:
    return sum(_char_units(char) for char in text)


def _max_line_units(aspect: str, font_size: int) -> float:
    width, _ = VideoAspect(aspect).to_resolution()
    return max(6.0, (width * 0.86) / max(12, int(font_size or 60)))


def _tokenize_for_wrap(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    for char in text:
        if char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            if tokens and not tokens[-1].endswith(" "):
                tokens.append(" ")
        elif unicodedata.east_asian_width(char) in {"W", "F"} or char in _NO_START_CHARS:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
        else:
            buffer.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _clean_line(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def wrap_subtitle_text(
    text: str, *, aspect: str = "9:16", font_size: int = 60
) -> str:
    cleaned = _clean_line(text.replace("\r\n", " ").replace("\n", " "))
    if not cleaned:
        return ""
    max_units = _max_line_units(aspect, font_size)
    total_units = _text_units(cleaned)
    if total_units <= max_units:
        return cleaned

    tokens = _tokenize_for_wrap(cleaned)
    if total_units <= max_units * 1.9:
        target = total_units / 2.0
        best_split = None
        best_score = float("inf")
        for index in range(1, len(tokens)):
            next_token = tokens[index]
            if next_token in _NO_START_CHARS:
                continue
            left = _clean_line("".join(tokens[:index]))
            right = _clean_line("".join(tokens[index:]))
            if not left or not right:
                continue
            left_units = _text_units(left)
            right_units = _text_units(right)
            if left_units > max_units or right_units > max_units:
                continue
            score = abs(left_units - target)
            if left[-1] in _STRIP_TRAILING_PUNCTUATION + _KEEP_TRAILING_PUNCTUATION:
                score -= 1.2
            elif next_token == " " or tokens[index - 1] == " ":
                score -= 0.4
            if score < best_score:
                best_score = score
                best_split = (left, right)
        if best_split is not None:
            return f"{best_split[0]}\n{best_split[1]}"

    lines: list[str] = []
    current = ""
    current_units = 0.0
    for token in tokens:
        token_units = _text_units(token)
        if token == " " and not current:
            continue
        if (
            current
            and current_units + token_units > max_units
            and token not in _NO_START_CHARS
        ):
            lines.append(_clean_line(current))
            current = "" if token == " " else token
            current_units = 0.0 if token == " " else token_units
        else:
            current += token
            current_units += token_units
    if current.strip():
        lines.append(_clean_line(current))
    return "\n".join(line for line in lines if line)


def _clean_cue_text(text: str) -> str:
    cleaned = _clean_line(text)
    if not cleaned:
        return ""
    if cleaned[-1] in _KEEP_TRAILING_PUNCTUATION:
        return cleaned
    return cleaned.rstrip(_STRIP_TRAILING_PUNCTUATION).strip()


def _raw_cues_from_segments(
    segments,
    *,
    aspect: str,
    font_size: int,
    word_level: bool,
) -> list[tuple[float, float, str]]:
    max_cue_units = _max_line_units(aspect, font_size) * 1.9
    raw_cues: list[tuple[float, float, str]] = []

    def emit(start_time: float, end_time: float, text: str) -> None:
        cleaned = _clean_cue_text(text)
        if cleaned:
            raw_cues.append((float(start_time), float(end_time), cleaned))

    for segment in segments:
        words = getattr(segment, "words", None)
        if not words:
            emit(segment.start, segment.end, getattr(segment, "text", ""))
            continue
        if word_level:
            for word in words:
                token = (getattr(word, "word", "") or "").strip()
                if token:
                    raw_cues.append((float(word.start), float(word.end), token))
            continue

        cue_start: float | None = None
        cue_end: float = 0.0
        buffer = ""
        total_words = len(words)
        for word_index, word in enumerate(words):
            token = getattr(word, "word", "") or ""
            if not token:
                continue
            candidate = buffer + token
            if (
                buffer.strip()
                and _text_units(_clean_line(candidate)) > max_cue_units
                and token.strip() not in _NO_START_CHARS
            ):
                emit(cue_start if cue_start is not None else word.start, cue_end, buffer)
                cue_start = None
                buffer = ""
            if cue_start is None:
                cue_start = float(word.start)
            cue_end = float(word.end)
            if word_index == total_words - 1 and getattr(segment, "end", word.end) > word.end:
                cue_end = float(segment.end)
            buffer += token
            if token.rstrip().endswith(_SPLIT_PUNCTUATIONS):
                emit(cue_start, cue_end, buffer)
                cue_start = None
                buffer = ""
        if buffer.strip() and cue_start is not None:
            emit(cue_start, cue_end, buffer)
    return raw_cues


def segments_to_cues(
    segments,
    *,
    shot_start: float,
    shot_end: float,
    aspect: str = "9:16",
    font_size: int = 60,
    word_level: bool = False,
    min_start: float = 0.0,
) -> list[SubtitleCue]:
    raw_cues = _raw_cues_from_segments(
        segments, aspect=aspect, font_size=font_size, word_level=word_level
    )
    cues: list[SubtitleCue] = []
    cursor = max(shot_start, min_start)
    for local_start, local_end, text in raw_cues:
        abs_start = shot_start + max(0.0, local_start)
        abs_end = shot_start + max(0.0, local_end)
        abs_start = min(max(abs_start, cursor), shot_end)
        abs_end = min(max(abs_end, abs_start), shot_end)
        if abs_end - abs_start < 0.02:
            abs_end = min(shot_end, abs_start + 0.15)
        if abs_end <= abs_start:
            continue
        wrapped = (
            text
            if word_level
            else wrap_subtitle_text(text, aspect=aspect, font_size=font_size)
        )
        if not wrapped:
            continue
        cues.append(SubtitleCue(start=abs_start, end=abs_end, text=wrapped))
        cursor = abs_end
    return cues


def extract_shot_audio(
    video_file: str | Path,
    start_seconds: float,
    duration_seconds: float,
    output_wav: str | Path,
    *,
    shot_index: int | None = None,
    remote_id: str | None = None,
) -> str:
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_file),
        "-ss",
        f"{max(0.0, start_seconds):.6f}",
        "-t",
        f"{max(0.001, duration_seconds):.6f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=_ffmpeg_timeout_seconds(),
    )
    if completed.returncode != 0 or not Path(output_wav).is_file():
        raise native.NativeStoryboardError(
            "failed to extract shot audio for transcription: "
            + (completed.stderr or "").strip(),
            stage="subtitles",
            shot=shot_index,
            remote_id=remote_id,
        )
    return str(output_wav)


def format_srt(cues: list[SubtitleCue]) -> str:
    blocks = [
        utils.text_to_srt(index, cue.text, cue.start, cue.end).rstrip()
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def generate_native_subtitles(
    clean_video: str | Path,
    plans,
    frame_counts: list[int],
    subtitle_path: str | Path,
    *,
    params,
    fps: int = native.FPS,
) -> dict:
    """Extract audio per spoken shot, transcribe with local Whisper, and write subtitle.srt."""
    if len(plans) != len(frame_counts):
        raise native.NativeStoryboardError(
            "shot frame counts do not match storyboard shots", stage="subtitles"
        )
    spoken_plans = [
        (plan, frames)
        for plan, frames in zip(plans, frame_counts, strict=True)
        if (getattr(plan, "narration", "") or "").strip()
    ]
    if not spoken_plans:
        return {"subtitle_path": "", "cue_count": 0}

    aspect = getattr(params, "video_aspect", "9:16") or "9:16"
    font_size = int(getattr(params, "font_size", 60) or 60)
    word_level = getattr(params, "subtitle_display_mode", "sentence") in {"word_by_word", "word"}
    target_path = Path(subtitle_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_cues: list[SubtitleCue] = []
    cursor_frames = 0
    previous_end = 0.0

    with tempfile.TemporaryDirectory(
        prefix="native-asr-", dir=target_path.parent
    ) as temporary:
        for plan, frames in zip(plans, frame_counts, strict=True):
            shot_start = cursor_frames / fps
            shot_duration = frames / fps
            shot_end = (cursor_frames + frames) / fps
            cursor_frames += frames

            narration = (getattr(plan, "narration", "") or "").strip()
            if not narration:
                continue

            wav_path = Path(temporary) / f"shot-{plan.index:04d}.wav"
            extract_shot_audio(
                clean_video,
                shot_start,
                shot_duration,
                wav_path,
                shot_index=plan.index,
                remote_id=getattr(plan, "remote_task_id", None),
            )
            try:
                segments, _info = subtitle_service.transcribe_audio(
                    str(wav_path),
                    initial_prompt=narration,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                    local_files_only=True,
                )
            except Exception as exc:
                raise native.NativeStoryboardError(
                    f"speech transcription failed: {exc}",
                    stage="subtitles",
                    shot=plan.index,
                    remote_id=getattr(plan, "remote_task_id", None),
                ) from exc

            shot_cues = segments_to_cues(
                segments,
                shot_start=shot_start,
                shot_end=shot_end,
                aspect=aspect,
                font_size=font_size,
                word_level=word_level,
                min_start=previous_end,
            )
            if shot_cues:
                all_cues.extend(shot_cues)
                previous_end = shot_cues[-1].end

    if not all_cues:
        raise native.NativeStoryboardError(
            "Whisper recognized no speech cues from spoken shots; "
            "check the generated audio or disable subtitles with --no-subtitle-enabled",
            stage="subtitles",
        )

    srt_content = format_srt(all_cues)
    if not write_subtitle_file(str(target_path), srt_content):
        raise native.NativeStoryboardError(
            "failed to publish native subtitle file", stage="subtitles"
        )
    logger.info(
        f"generated native subtitles: cues={len(all_cues)}, path={target_path}"
    )
    return {"subtitle_path": str(target_path), "cue_count": len(all_cues)}


def _parse_srt_time_range(timing: str) -> tuple[float, float] | None:
    matched = _SRT_RANGE_RE.match(timing)
    if not matched:
        return None
    sh, sm, ss, sms, eh, em, es, ems = (int(value) for value in matched.groups())
    start = sh * 3600 + sm * 60 + ss + sms / 1000.0
    end = eh * 3600 + em * 60 + es + ems / 1000.0
    return start, end


def _format_ass_time(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def hex_to_ass_color(color: str, alpha: int = 0) -> str:
    red, green, blue = ImageColor.getrgb(color)
    clamped_alpha = max(0, min(255, int(alpha)))
    return f"&H{clamped_alpha:02X}{blue:02X}{green:02X}{red:02X}"


def _escape_ass_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    escaped = (
        normalized.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )
    return escaped


def build_ass_content(
    cues: list[SubtitleCue],
    *,
    params,
    width: int,
    height: int,
    font_family: str,
    font_bold: bool,
) -> str:
    font_size = int(getattr(params, "font_size", 60) or 60)
    fore_color = getattr(params, "text_fore_color", None) or "#FFFFFF"
    stroke_color = getattr(params, "stroke_color", None) or "#000000"
    stroke_width = float(
        1.5 if getattr(params, "stroke_width", None) is None else params.stroke_width
    )
    bg_setting = getattr(params, "text_background_color", False)

    primary_ass = hex_to_ass_color(fore_color, alpha=0)
    if bg_setting:
        bg_color = bg_setting if isinstance(bg_setting, str) else "#000000"
        border_style = 3
        outline_ass = hex_to_ass_color(bg_color, alpha=0)
        back_ass = "&HFF000000"
        outline_size = max(8.0, round(font_size * 0.2, 2))
    else:
        border_style = 1
        outline_ass = hex_to_ass_color(stroke_color, alpha=0)
        back_ass = "&H80000000"
        outline_size = max(0.0, stroke_width)

    position = (getattr(params, "subtitle_position", None) or "bottom").strip().lower()
    margin_x = max(20, round(width * 0.06))
    margin_v = max(10, round(height * 0.05))
    override_prefix = ""
    if position == "top":
        alignment = 8
    elif position == "center":
        alignment = 5
        margin_v = 0
    elif position in {"two_thirds_bottom", "two_thirds", "2/3_bottom"}:
        alignment = 5
        margin_v = 0
        pos_y = round(height / 3.0)
        override_prefix = f"{{\\an5\\pos({width // 2},{pos_y})}}"
    elif position == "custom":
        alignment = 5
        margin_v = 0
        percent = float(getattr(params, "custom_position", 70.0) or 70.0)
        min_y = max(10, font_size // 2 + 10)
        max_y = max(min_y, height - min_y)
        pos_y = max(min_y, min(max_y, round(height * (percent / 100.0))))
        override_prefix = f"{{\\an5\\pos({width // 2},{pos_y})}}"
    else:
        alignment = 2

    bold_flag = -1 if font_bold else 0
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,{font_family},{font_size},{primary_ass},{primary_ass},"
            f"{outline_ass},{back_ass},{bold_flag},0,0,0,100,100,0,0,"
            f"{border_style},{outline_size:g},0,{alignment},{margin_x},{margin_x},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = [
        f"Dialogue: 0,{_format_ass_time(cue.start)},{_format_ass_time(cue.end)},"
        f"Default,,0,0,0,,{override_prefix}{_escape_ass_text(cue.text)}"
        for cue in cues
    ]
    return "\n".join(header + events) + "\n"


def load_srt_cues(subtitle_path: str | Path) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for _idx, timing, text in file_to_subtitles(str(subtitle_path)):
        parsed = _parse_srt_time_range(timing)
        if parsed is None or not text.strip():
            continue
        cues.append(SubtitleCue(start=parsed[0], end=parsed[1], text=text.strip()))
    return cues


def burn_native_subtitles(
    clean_video: str | Path,
    subtitle_path: str | Path,
    output_file: str | Path,
    *,
    params,
    aspect: str,
    threads: int = 2,
) -> dict:
    """Render ASS subtitles onto the stitched native video while copying its audio stream."""
    cues = load_srt_cues(subtitle_path)
    if not cues:
        raise native.NativeStoryboardError(
            "subtitle file contains no timed cues to burn", stage="subtitles"
        )
    font_path = resolve_subtitle_font_path(getattr(params, "font_name", None))
    font_family, font_bold = extract_font_face(font_path)
    width, height = VideoAspect(aspect).to_resolution()
    ass_content = build_ass_content(
        cues,
        params=params,
        width=width,
        height=height,
        font_family=font_family,
        font_bold=font_bold,
    )

    clean_resolved = Path(clean_video).resolve()
    output_resolved = Path(output_file).resolve()
    output_resolved.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="native-sub-", dir=output_resolved.parent
    ) as temporary:
        temp_dir = Path(temporary)
        fonts_dir = temp_dir / "fonts"
        fonts_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(font_path, fonts_dir / font_path.name)
        ass_file = temp_dir / "sub.ass"
        ass_file.write_text(ass_content, encoding="utf-8-sig")
        staged = temp_dir / "final.mp4"
        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-v",
            "error",
            "-i",
            str(clean_resolved),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            "ass=filename=sub.ass:fontsdir=fonts",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-threads",
            str(threads or 2),
            str(staged),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(temp_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=_ffmpeg_timeout_seconds(),
            )
        except subprocess.TimeoutExpired as exc:
            raise native.NativeStoryboardError(
                "FFmpeg subtitle burn-in timed out", stage="subtitles"
            ) from exc
        if completed.returncode != 0 or not staged.is_file():
            raise native.NativeStoryboardError(
                "FFmpeg subtitle burn-in failed: " + (completed.stderr or "").strip(),
                stage="subtitles",
            )
        final_info = native.inspect_clip(staged)
        os.replace(staged, output_resolved)
    return {
        "videos": [str(output_resolved)],
        "output_seconds": final_info["duration"],
    }
