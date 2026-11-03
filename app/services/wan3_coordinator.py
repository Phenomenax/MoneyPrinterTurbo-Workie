"""Local-disk, cross-process Wan3 admission control (no paid-submit retries).

Owner locks are ephemeral; slot records survive crashes. Only a confirmed remote
terminal state or a request known not to have been sent/rejected frees a slot.
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from app import paths

CAPACITY = 5
SUBMIT_INTERVAL = 0.25
TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"})


class CoordinationError(RuntimeError):
    """An unresolved record needs verification before capacity can be reused."""


class SubmissionStopped(RuntimeError):
    """A sibling failed before this worker submitted a paid request."""


def check_stopped(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise SubmissionStopped("Wan3 submission stopped after another clip failed")


def atomic_json(target: Path, data) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.stem}-", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FileLock:
    """Separate open handles lock threads as well as processes on local disks."""

    def __init__(self, target: Path):
        self.handle = target.open("a+b")
        # msvcrt locks a byte range, including on a freshly created file.
        if os.name == "nt" and target.stat().st_size == 0:
            self.handle.write(b"\0")
            self.handle.flush()

    def acquire(self) -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            return False
        return True

    def close(self):
        # Closing the handle also releases the OS lock, including on Windows.
        self.handle.close()


@contextmanager
def file_lock(target: Path, stop_event=None):
    lock = FileLock(target)
    try:
        while not lock.acquire():
            check_stopped(stop_event)
            time.sleep(0.05)
        check_stopped(stop_event)
        yield
    finally:
        lock.close()


class Slot:
    def __init__(self, root: Path, index: int, owner: FileLock):
        self.root = root
        self.target = root / f"slot-{index}.json"
        self.owner = owner
        self.record = {}

    def save(self, phase: str, **fields):
        self.record.update(phase=phase, **fields)
        atomic_json(self.target, self.record)

    def complete(self):
        atomic_json(self.target, None)
        self.record = {}

    def close(self):
        self.owner.close()

    @contextmanager
    def submission(self, stop_event=None):
        """Hold the shared lock only through POST, never during remote polling."""
        with file_lock(self.root / "submit.lock", stop_event):
            target = self.root / "submit.json"
            last = json.loads(target.read_text()) if target.exists() else 0
            delay = max(0.0, SUBMIT_INTERVAL - (time.time() - float(last)))
            if stop_event is None:
                time.sleep(delay)
            elif stop_event.wait(delay):
                check_stopped(stop_event)
            check_stopped(stop_event)
            # Persist uncertainty BEFORE entering requests.post. A killed process
            # cannot turn a possibly accepted request into an empty slot.
            self.save("submitting")
            atomic_json(target, time.time())
            if stop_event is not None and stop_event.is_set():
                self.complete()  # Known not to have entered POST yet.
                check_stopped(stop_event)
            # The durable start timestamp and the lock held through POST already
            # space submissions. Do not let advisory bookkeeping after POST hide
            # an accepted response before its paid task ID can be recorded.
            yield


def _read_record(target):
    try:
        record = json.loads(target.read_text()) if target.exists() else None
        if record is not None and (
            not isinstance(record, dict)
            or record.get("phase") not in {"reserved", "submitting", "submitted"}
            or not isinstance(record.get("task_id"), str)
            or not isinstance(record.get("base_url"), str)
        ):
            raise ValueError("invalid slot record")
        return record
    except (ValueError, OSError) as exc:
        raise CoordinationError(f"Wan3 record requires verification: {target}") from exc


def acquire_slot(*, base_url: str, probe, stop_event: threading.Event | None = None) -> Slot:
    root = paths.runtime_root() / ".agent-logs" / "wan3"
    root.mkdir(parents=True, exist_ok=True)
    while True:
        check_stopped(stop_event)
        unresolved = []
        for index in range(CAPACITY):
            owner = FileLock(root / f"slot-{index}.lock")
            slot = Slot(root, index, owner)
            keep = False
            try:
                if not owner.acquire():
                    continue
                record = _read_record(slot.target)
                if record and record.get("phase") != "reserved":
                    remote_id = record.get("task_id")
                    if not remote_id or record.get("base_url") != base_url:
                        unresolved.append(f"{slot.target}: submission requires verification")
                        continue
                    try:
                        terminal = probe(remote_id)
                    except Exception as exc:
                        unresolved.append(f"{slot.target}: status requires verification ({type(exc).__name__})")
                        continue
                    if not terminal:
                        continue
                check_stopped(stop_event)
                slot.save("reserved", base_url=base_url, pid=os.getpid(), task_id="")
                keep = True
                return slot
            finally:
                if not keep:
                    slot.close()
        if unresolved:
            raise CoordinationError("Wan3 capacity cannot be safely reclaimed; " + "; ".join(unresolved))
        if stop_event is None:
            time.sleep(0.5)
        else:
            stop_event.wait(0.5)
