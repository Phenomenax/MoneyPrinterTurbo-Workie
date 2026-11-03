"""Optional CLI progress snapshots; never participates in generation decisions."""

import json
import logging
import math
import os
import re
import threading
from datetime import datetime, timezone
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

ACTIVITY_PHASES = frozenset({
    "submitting", "remote_queued", "generating", "checking", "generated",
    "downloading", "verifying",
})
_IDENTIFIER = re.compile(r"^[\w.:-]{1,200}$")


def public_activity(activity):
    """Whitelist display fields; drop URLs, provider text and malformed values."""
    if (not isinstance(activity, dict) or not isinstance(activity.get("phase"), str)
            or activity["phase"] not in ACTIVITY_PHASES):
        return None
    result = {"phase": activity["phase"]}
    for key in ("provider", "remote_id"):
        value = activity.get(key)
        if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
            result[key] = value
    confirmed = activity.get("confirmed_at")
    if type(confirmed) in (int, float) and math.isfinite(confirmed) and confirmed >= 0:
        result["confirmed_at"] = confirmed
    if isinstance(activity.get("retrying"), bool):
        result["retrying"] = activity["retrying"]
    for key in ("downloaded_bytes", "total_bytes"):
        value = activity.get(key)
        if type(value) is int and value >= 0:
            result[key] = value
    return result


class TaskProgress:
    def __init__(self, path, task_id, state, interval=1.0):
        self.path = Path(path)
        self.task_id = task_id
        self.state = state
        self.interval = interval
        self._stop = threading.Event()
        self._warned = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def finish(self, succeeded):
        self._succeeded = succeeded
        self._stop.set()
        # State backends (including Redis) may block. Advisory progress must not
        # hold up the CLI result, and only this worker may write snapshots.
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            logging.getLogger(__name__).warning("Progress export is busy; returning the CLI result")

    def _run(self):
        while not self._stop.is_set():
            self._write()
            self._stop.wait(self.interval)
        self._write("generated" if self._succeeded else "failed")

    def _write(self, status=None):
        temp = None
        try:
            task = self.state.get_task(self.task_id) or {}
            payload = {
                "version": 1,
                "task_id": self.task_id,
                "status": status or ("failed" if task.get("state") == -1 else "running"),
                "stage": task.get("stage", "script"),
                "percent": max(0, min(100, int(task.get("progress", 0)))),
                "clips_completed": task.get("clips_completed"),
                "clips_total": task.get("clips_total"),
                "generation_activity": public_activity(task.get("generation_activity")),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp")
            temp.write_text(json.dumps(payload), encoding="utf-8")
            temp.replace(self.path)
        except Exception as exc:
            if not self._warned:
                logging.getLogger(__name__).warning("Progress export unavailable: %s", type(exc).__name__)
                self._warned = True
        finally:
            if temp is not None:
                with suppress(OSError):
                    temp.unlink(missing_ok=True)
