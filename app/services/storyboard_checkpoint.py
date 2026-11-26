"""Opt-in, local storyboard checkpoints. Never serialize credentials or Python code."""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

from app.services.wan3_coordinator import FileLock, atomic_json

CURRENT = contextvars.ContextVar("storyboard_checkpoint", default=None)
STAGES = ("audio", "subtitles", "materials", "combine", "native_subtitles", "render")


class CheckpointError(RuntimeError):
    stage = "checkpoint"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _pack(value):
    if is_dataclass(value):
        return {
            "__kind": type(value).__name__,
            "fields": {
                field.name: _pack(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"__kind": "tuple", "items": [_pack(item) for item in value]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _pack(item) for key, item in value.items()}
    if hasattr(value, "get_srt"):
        return {
            "__kind": "SubMaker",
            "cues": [
                [
                    cue.index,
                    cue.start.total_seconds(),
                    cue.end.total_seconds(),
                    cue.content,
                ]
                for cue in getattr(value, "cues", [])
            ],
            "type": getattr(value, "type", None),
            "subs": list(getattr(value, "subs", [])),
            "offset": list(getattr(value, "offset", [])),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise CheckpointError(f"unsupported checkpoint value: {type(value).__name__}")


def _unpack(value):
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__kind")
    if kind == "tuple":
        return tuple(_unpack(item) for item in value["items"])
    if kind == "SubMaker":
        from datetime import timedelta
        from edge_tts import SubMaker
        from edge_tts.srt_composer import Subtitle

        result = SubMaker()
        result.cues = [
            Subtitle(
                index=index,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=content,
            )
            for index, start, end, content in value["cues"]
        ]
        result.type = value.get("type")
        result.subs, result.offset = value["subs"], value["offset"]
        return result
    if kind in {"ShotTimeline", "StoryboardTimeline"}:
        from app.services.storyboard_timing import ShotTimeline, StoryboardTimeline

        cls = ShotTimeline if kind == "ShotTimeline" else StoryboardTimeline
        return cls(**{key: _unpack(item) for key, item in value["fields"].items()})
    if kind:
        raise CheckpointError("unsupported checkpoint type")
    return {key: _unpack(item) for key, item in value.items()}


def _files(value):
    if isinstance(value, str) and Path(value).is_absolute() and Path(value).is_file():
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _files(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _files(item)


class Checkpoint:
    def __init__(self, task_dir, identity, *, resume=False, allow_rebuild=False):
        self.root = Path(task_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "storyboard-checkpoint.json"
        self.guard = threading.RLock()
        self.allow_rebuild = allow_rebuild
        self.owner = FileLock(self.root / "storyboard-checkpoint.lock")
        if not self.owner.acquire():
            self.owner.close()
            raise CheckpointError("original storyboard task is still running")
        try:
            if self.path.exists():
                if not resume:
                    raise CheckpointError("existing task requires explicit resume")
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if (
                    self.data.get("version") != 1
                    or self.data.get("identity") != identity
                ):
                    raise CheckpointError(
                        "checkpoint inputs changed; original task was not resubmitted"
                    )
            elif resume:
                raise CheckpointError(
                    "checkpoint missing; cannot safely resume the original task"
                )
            else:
                self.data = {
                    "version": 1,
                    "identity": identity,
                    "stages": {},
                    "clips": {},
                }
                self.save()
        except BaseException:
            self.owner.close()
            raise

    def close(self):
        self.owner.close()

    def save(self):
        with self.guard:
            atomic_json(self.path, self.data)

    def valid(self, record, *, artifact_roots=()):
        roots = (self.root, *(Path(root).resolve() for root in artifact_roots))
        for name, expected in record.get("files", {}).items():
            target = Path(name).resolve()
            if (
                not any(target.is_relative_to(root) for root in roots)
                or not target.is_file()
                or digest(target) != expected
            ):
                return False
        return True

    def call(self, stage, operation):
        record = self.data["stages"].get(stage)
        if record is not None:
            if self.valid(record):
                return _unpack(record["value"])
            if not self.allow_rebuild:
                raise CheckpointError(
                    f"saved {stage} files are missing or damaged; explicit rebuild approval required"
                )
            for dependent in STAGES[STAGES.index(stage) :]:
                self.data["stages"].pop(dependent, None)
            self.save()
        result = operation()
        packed = _pack(result)
        owned = sorted(set(_files(packed)))
        if any(not Path(name).resolve().is_relative_to(self.root) for name in owned):
            raise CheckpointError("checkpoint artifact is outside the task directory")
        self.data["stages"][stage] = {
            "value": packed,
            "files": {name: digest(name) for name in owned},
        }
        self.save()
        return result

    def clip(self, index, *, persist=True, **changes):
        with self.guard:
            key = str(index)
            entry = self.data["clips"].setdefault(key, {})
            if changes:
                entry.update(changes)
                if persist:
                    self.save()
            return dict(entry)


@contextmanager
def scope(task_dir, identity, *, enabled=False, resume=False, allow_rebuild=False):
    if not enabled:
        yield None
        return
    checkpoint = Checkpoint(
        task_dir, identity, resume=resume, allow_rebuild=allow_rebuild
    )
    token = CURRENT.set(checkpoint)
    try:
        yield checkpoint
    finally:
        CURRENT.reset(token)
        checkpoint.close()


def run(stage, operation):
    checkpoint = CURRENT.get()
    return operation() if checkpoint is None else checkpoint.call(stage, operation)


def start(
    *, task_id, params, storyboard_data, resume=False, allow_rebuild=False, revision_plan=None, **kwargs
):
    from app.services import task
    from app.utils import utils

    inputs = {
        "params": params.model_dump(),
        "storyboard": _pack(storyboard_data),
        "video": kwargs.get("video_run_config").summary()
        if kwargs.get("video_run_config") is not None
        else None,
    }
    if kwargs.get("storyboard_audio_mode", "tts") == "native":
        from app.services.storyboard_native import build_generation_plan

        # Preserve the fingerprint written before Resolution was an optional header.
        if getattr(storyboard_data, "resolution", None) is None:
            inputs["storyboard"]["fields"].pop("resolution", None)
        inputs["storyboard_audio_mode"] = "native"
        inputs["native_requests"] = [
            [plan.prompt, plan.request_seconds]
            for plan in build_generation_plan(storyboard_data, kwargs.get("video_run_config"))
        ]
    if revision_plan is not None:
        if Path(task_id) == Path(revision_plan["source"]["task_id"]):
            raise ValueError("修改任务需要单独的任务编号。")
        inputs["revision_plan"] = revision_plan
    identity = fingerprint(inputs)
    with scope(
        utils.task_dir(task_id),
        identity,
        enabled=True,
        resume=resume,
        allow_rebuild=allow_rebuild,
    ) as checkpoint:
        if revision_plan is not None:
            from app.services.storyboard_revision import stage_revision

            stage_revision(checkpoint, revision_plan, storyboard_data,
                           kwargs["video_run_config"], params.video_aspect)
        return task.start(
            task_id=task_id, params=params, storyboard_data=storyboard_data, **kwargs
        )
