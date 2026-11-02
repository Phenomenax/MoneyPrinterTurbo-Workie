"""Optional CLI progress snapshots; never participates in generation decisions."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from contextlib import suppress
from pathlib import Path
from uuid import uuid4


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
