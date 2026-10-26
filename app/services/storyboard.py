"""Read the fixed Workie storyboard Markdown format without an LLM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Storyboard:
    video_script: str
    video_terms: list[str]
    aspect: str | None = None
    title: str = ""


def _section(text: str, name: str) -> str:
    match = re.search(r"^##[ \t]+" + name + r"[^\n]*\n(.*?)(?=^#{1,2}[ \t]|\Z)", text,
                      re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"storyboard is missing the {name} section")
    return match.group(1).strip()


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
    if (not text or text in {"—", "-", "…", "..."}
            or re.search(r"<[^>]+>|\b(?:TODO|TBD|INSERT)\b", text, re.IGNORECASE)
            or text.lower().startswith("just the spoken text")):
        raise ValueError(f"storyboard {label} is empty or contains an unfilled placeholder")
    return text


def parse_storyboard(path: str | Path) -> Storyboard:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"storyboard does not exist or is not a file: {path}")
    try:
        text = source.read_text(encoding="utf-8-sig").strip()
    except UnicodeError as exc:
        raise ValueError("storyboard must be UTF-8 encoded") from exc
    if not text:
        raise ValueError("storyboard is empty")
    if text.startswith("```markdown") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    beats = _section(text, r"Beats\s*\(timed\)")
    lines = beats.splitlines()
    header_index = next((i for i, line in enumerate(lines)
                         if line.strip().startswith("|") and "Shot type" in line
                         and "On-screen / B-roll" in line), None)
    if header_index is None:
        raise ValueError("storyboard Beats table is missing the required columns")
    headers = _cells(lines[header_index])
    expected = ["Time", "Shot type", "On-screen / B-roll", "VO / on-camera line", "Music / SFX"]
    if headers != expected:
        raise ValueError("storyboard Beats table must use the standard five columns")
    remaining = lines[header_index + 1:]
    if not remaining or not all(re.fullmatch(r":?-{3,}:?", c) for c in _cells(remaining[0])):
        raise ValueError("storyboard Beats table is missing its header separator")

    prompts, pending = [], ""
    for line in remaining[1:]:
        if not pending and not line.strip().startswith("|"):
            if prompts:
                break
            continue
        pending = pending + "\n" + line if pending else line
        cells = _cells(pending)
        if len(cells) < len(expected):
            continue
        if len(cells) != len(expected):
            raise ValueError("storyboard Beats row has an invalid column count; escape literal pipes as \\|")
        shot = _require_filled(cells[1], "shot type")
        visual = _require_filled(cells[2], "visual description")
        prompts.append(f"{shot}: {visual}")
        pending = ""
    if pending or not prompts:
        raise ValueError("storyboard Beats table has no complete shots")

    script_section = _section(text, r"Voiceover script\b")
    script = "\n".join(re.sub(r"^\s*> ?", "", line).rstrip()
                       for line in script_section.splitlines()).strip()
    script = _require_filled(script, "voiceover script")
    aspect_match = re.search(r"\*\*Aspect(?::\*\*|\*\*:)\s*([^\s*]+)", text, re.IGNORECASE)
    aspect = aspect_match.group(1).strip() if aspect_match else None
    if aspect is not None and aspect not in {"16:9", "9:16", "1:1"}:
        raise ValueError("storyboard Aspect must be 16:9, 9:16, or 1:1")
    title_match = re.search(r"^# +(?:🎬\s*)?(?:Storyboard\s*[—:-]\s*)?(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    return Storyboard(video_script=script, video_terms=prompts, aspect=aspect, title=title)
