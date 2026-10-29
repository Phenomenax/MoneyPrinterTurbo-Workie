"""Read the fixed Workie storyboard Markdown format without an LLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


_RANGE_SPLIT = re.compile(r"\s*(?:-|–|—)\s*")
_HMS = re.compile(r"^(\d+):([0-5]\d):([0-5]\d(?:\.\d+)?)$")
_MS = re.compile(r"^(\d+):([0-5]\d(?:\.\d+)?)$")
_SECONDS = re.compile(r"^(\d+(?:\.\d+)?)$")
_QUOTE_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
)
_SILENCE_MARKERS = {"", "—", "-"}


@dataclass(frozen=True)
class StoryboardShot:
    index: int
    prompt: str
    narration: str
    planned_start_seconds: float
    planned_end_seconds: float


@dataclass(frozen=True)
class Storyboard:
    video_script: str
    video_terms: list[str]
    aspect: str | None = None
    title: str = ""
    shots: tuple[StoryboardShot, ...] = field(default_factory=tuple)


def _section_match(text: str, name: str) -> re.Match[str]:
    match = re.search(
        r"^##[ \t]+" + name + r"[^\n]*\n(.*?)(?=^#{1,2}[ \t]|\Z)",
        text,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"storyboard is missing the {name} section")
    return match


def _section(text: str, name: str) -> str:
    return _section_match(text, name).group(1).strip()


def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    cells, current = [], []
    index = 0
    while index < len(row):
        char = row[index]
        if char == "\\" and index + 1 < len(row) and row[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return [re.sub(r"<br\s*/?>", "\n", cell, flags=re.IGNORECASE) for cell in cells]


def _require_filled(text: str, label: str) -> str:
    text = text.strip()
    if _is_unfilled(text):
        raise ValueError(f"storyboard {label} is empty or contains an unfilled placeholder")
    return text


def _is_unfilled(text: str) -> bool:
    text = text.strip()
    return (
        not text
        or text in {"—", "-", "…", "..."}
        or re.search(r"<[^>]+>|\b(?:TODO|TBD|INSERT)\b", text, re.IGNORECASE) is not None
        or text.lower().startswith("just the spoken text")
    )


def _is_silence_marker(text: str) -> bool:
    return text.strip() in _SILENCE_MARKERS


def _parse_clock(token: str) -> float:
    token = token.strip()
    matched = _HMS.fullmatch(token)
    if matched:
        hours, minutes, seconds = matched.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    matched = _MS.fullmatch(token)
    if matched:
        minutes, seconds = matched.groups()
        return int(minutes) * 60 + float(seconds)
    matched = _SECONDS.fullmatch(token)
    if matched:
        return float(matched.group(1))
    raise ValueError(token)


def _parse_time_range(cell: str, line_no: int) -> tuple[float, float]:
    parts = [part for part in _RANGE_SPLIT.split(cell.strip()) if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"storyboard shot timing at line {line_no} must be one start-end range: {cell.strip()}"
        )
    try:
        start = _parse_clock(parts[0])
        end = _parse_clock(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"storyboard shot timing at line {line_no} is not a supported time: {cell.strip()}"
        ) from exc
    return start, end


def _strip_outer_quotes(text: str) -> str:
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in _QUOTE_PAIRS:
            if (
                text.startswith(left)
                and text.endswith(right)
                and len(text) > len(left) + len(right) - 1
            ):
                text = text[len(left):-len(right)].strip()
                changed = True
                break
    return text


def _normalize_spoken(text: str) -> str:
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"^\s*>\s?", "", line)
        lines.append(_strip_outer_quotes(line.strip()))
    compact = re.sub(r"\s+", "", "".join(lines))
    return _strip_outer_quotes(compact)


def _load_text(path: str | Path) -> tuple[str, int]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"storyboard does not exist or is not a file: {path}")
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("storyboard must be UTF-8 encoded") from exc
    if not raw.strip():
        raise ValueError("storyboard is empty")
    prefix = len(raw) - len(raw.lstrip())
    base_line = raw[:prefix].count("\n")
    text = raw.strip()
    if text.startswith("```markdown") and text.endswith("```"):
        newline = text.find("\n")
        if newline < 0:
            raise ValueError("storyboard is empty")
        inner = text[newline + 1:]
        fence = inner.rfind("```")
        if fence >= 0:
            inner = inner[:fence]
        inner_prefix = len(inner) - len(inner.lstrip())
        base_line += text[: newline + 1].count("\n") + inner[:inner_prefix].count("\n")
        text = inner.strip()
        if not text:
            raise ValueError("storyboard is empty")
    return text, base_line


def _line_number(text: str, base_line: int, pos: int) -> int:
    return base_line + text.count("\n", 0, pos) + 1


def _beat_rows(text: str, base_line: int) -> tuple[list[tuple[int, str]], int]:
    match = _section_match(text, r"Beats\s*\(timed\)")
    body = match.group(1)
    body_pos = match.start(1)
    rows: list[tuple[int, str]] = []
    cursor = body_pos
    for line in body.splitlines(keepends=True):
        rows.append((_line_number(text, base_line, cursor), line.rstrip("\r\n")))
        cursor += len(line)
    return rows, _line_number(text, base_line, match.start())


def _parse_shots(
    rows: list[tuple[int, str]],
    *,
    require_shot_timing: bool,
) -> tuple[list[str], tuple[StoryboardShot, ...]]:
    header_index = next(
        (
            index
            for index, (_, line) in enumerate(rows)
            if line.strip().startswith("|")
            and "Shot type" in line
            and "On-screen / B-roll" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("storyboard Beats table is missing the required columns")
    header_line, header_text = rows[header_index]
    headers = _cells(header_text)
    expected = ["Time", "Shot type", "On-screen / B-roll", "VO / on-camera line", "Music / SFX"]
    if headers != expected:
        raise ValueError("storyboard Beats table must use the standard five columns")
    remaining = rows[header_index + 1:]
    if not remaining or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in _cells(remaining[0][1])):
        raise ValueError("storyboard Beats table is missing its header separator")

    prompts: list[str] = []
    shots: list[StoryboardShot] = []
    pending = ""
    pending_line = header_line
    for line_no, line in remaining[1:]:
        if not pending and not line.strip().startswith("|"):
            if prompts:
                break
            continue
        if not pending:
            pending_line = line_no
        pending = pending + "\n" + line if pending else line
        cells = _cells(pending)
        if len(cells) < len(expected):
            continue
        if len(cells) != len(expected):
            raise ValueError(
                "storyboard Beats row has an invalid column count at "
                f"line {pending_line}; escape literal pipes as \\|"
            )
        shot_type = _require_filled(cells[1], f"shot type at line {pending_line}")
        visual = _require_filled(cells[2], f"visual description at line {pending_line}")
        prompt = f"{shot_type}: {visual}"
        prompts.append(prompt)
        if require_shot_timing:
            start, end = _parse_time_range(cells[0], pending_line)
            if _is_silence_marker(cells[3]):
                narration = ""
            elif _is_unfilled(cells[3]):
                raise ValueError(
                    "storyboard narration at line "
                    f"{pending_line} is empty or contains an unfilled placeholder"
                )
            else:
                narration = cells[3].strip()
            _validate_shot_bounds(
                index=len(shots) + 1,
                line_no=pending_line,
                cell=cells[0],
                start=start,
                end=end,
                previous=shots[-1] if shots else None,
            )
            shots.append(
                StoryboardShot(
                    index=len(shots) + 1,
                    prompt=prompt,
                    narration=narration,
                    planned_start_seconds=start,
                    planned_end_seconds=end,
                )
            )
        pending = ""
    if pending or not prompts:
        raise ValueError("storyboard Beats table has no complete shots")
    return prompts, tuple(shots)


def _milliseconds(seconds: float) -> int:
    return int(round(seconds * 1000))


def _validate_shot_bounds(
    *,
    index: int,
    line_no: int,
    cell: str,
    start: float,
    end: float,
    previous: StoryboardShot | None,
) -> None:
    label = cell.strip()
    start_ms = _milliseconds(start)
    end_ms = _milliseconds(end)
    if end_ms < start_ms:
        raise ValueError(f"storyboard shot timing is reversed at line {line_no}: {label}")
    if end_ms <= start_ms:
        raise ValueError(f"storyboard shot timing has zero length at line {line_no}: {label}")
    if index == 1:
        if abs(start_ms) > 1:
            raise ValueError(
                f"storyboard first shot must start at 0 at line {line_no}: {label}"
            )
        return
    assert previous is not None
    previous_end_ms = _milliseconds(previous.planned_end_seconds)
    previous_start_ms = _milliseconds(previous.planned_start_seconds)
    if start_ms + 1 < previous_start_ms:
        raise ValueError(f"storyboard shot timing is reversed at line {line_no}: {label}")
    if start_ms + 1 < previous_end_ms:
        raise ValueError(f"storyboard shot timing overlaps at line {line_no}: {label}")
    if start_ms > previous_end_ms + 1:
        raise ValueError(
            "storyboard shot timing has a gap at line "
            f"{line_no}: previous shot ends at {previous.planned_end_seconds:.3f}s "
            f"and this shot starts at {start:.3f}s"
        )


def _voiceover_script(
    text: str,
    base_line: int,
    shots: tuple[StoryboardShot, ...],
    *,
    require_shot_timing: bool,
) -> str:
    match = _section_match(text, r"Voiceover script\b")
    script_line = _line_number(text, base_line, match.start())
    script = "\n".join(
        re.sub(r"^\s*> ?", "", line).rstrip() for line in match.group(1).splitlines()
    ).strip()
    if not require_shot_timing:
        return _require_filled(script, "voiceover script")
    spoken = [shot.narration for shot in shots if shot.narration.strip()]
    if not spoken:
        if _normalize_spoken(script) == "" or _is_silence_marker(script):
            return ""
        raise ValueError(
            "storyboard voiceover script does not match the shot lines "
            f"near line {script_line}"
        )
    script = _require_filled(script, "voiceover script")
    if _normalize_spoken("\n".join(spoken)) != _normalize_spoken(script):
        raise ValueError(
            "storyboard voiceover script does not match the shot lines "
            f"near line {script_line}"
        )
    return script


def parse_storyboard(path: str | Path, *, require_shot_timing: bool = False) -> Storyboard:
    text, base_line = _load_text(path)
    rows, _header_line = _beat_rows(text, base_line)
    prompts, shots = _parse_shots(rows, require_shot_timing=require_shot_timing)
    script = _voiceover_script(
        text, base_line, shots, require_shot_timing=require_shot_timing
    )
    aspect_match = re.search(r"\*\*Aspect(?::\*\*|\*\*:)\s*([^\s*]+)", text, re.IGNORECASE)
    aspect = aspect_match.group(1).strip() if aspect_match else None
    if aspect is not None and aspect not in {"16:9", "9:16", "1:1"}:
        raise ValueError("storyboard Aspect must be 16:9, 9:16, or 1:1")
    title_match = re.search(r"^# +(?:🎬\s*)?(?:Storyboard\s*[—:-]\s*)?(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    return Storyboard(
        video_script=script,
        video_terms=prompts,
        aspect=aspect,
        title=title,
        shots=shots,
    )
