"""Progress is observational: no credentials, paid calls, or generation retries."""
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cli
from app.services.progress import TaskProgress
from app.services import material, task as tm
from app.services.state import MemoryState


def test_cli_exports_live_state_and_joins_before_final_snapshot(tmp_path, monkeypatch, capsys):
    state = MemoryState()
    monkeypatch.setattr(tm.sm, "state", state)
    monkeypatch.setattr(cli, "build_video_params", lambda _: object())
    monkeypatch.setattr(cli, "prepare_cli_files", lambda *_a, **_k: None)
    original = TaskProgress.__init__
    def fast_reporter(self, *args, **kwargs):
        original(self, *args, **kwargs, interval=0.01)
    monkeypatch.setattr(TaskProgress, "__init__", fast_reporter)
    target = tmp_path / "进度.json"
    observed = []
    def generate(task_id, **_kwargs):
        state.update_task(task_id, progress=44, stage="materials", clips_completed=2, clips_total=5,
                          api_key="never-export", remote_url="secret-url")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                snapshot = json.loads(target.read_text())
            except FileNotFoundError:
                time.sleep(0.01)
                continue
            if snapshot.get("clips_completed") == 2:
                observed.append(snapshot)
                break
            time.sleep(0.01)
        state.update_task(task_id, state=1, progress=100, stage="render")
        return {"videos": ["final.mp4"]}
    monkeypatch.setattr(tm, "start", generate)
    assert cli.run_cli(["--video-subject", "test", "--progress-file", str(target)]) == 0
    assert observed and observed[0]["percent"] == 44
    final = json.loads(target.read_text())
    assert final["status"] == "generated"
    assert final["percent"] == 100
    assert "secret" not in json.dumps(final) and "api_key" not in final
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(capsys.readouterr().out)["result"]["videos"] == ["final.mp4"]


@pytest.mark.parametrize("raises", [False, True])
def test_cli_failure_flushes_partial_progress(tmp_path, monkeypatch, raises):
    state = MemoryState()
    monkeypatch.setattr(tm.sm, "state", state)
    monkeypatch.setattr(cli, "build_video_params", lambda _: object())
    monkeypatch.setattr(cli, "prepare_cli_files", lambda *_a, **_k: None)
    def generate(task_id, **_kwargs):
        state.update_task(task_id, progress=42, stage="materials", clips_completed=1, clips_total=4)
        if raises:
            raise RuntimeError("simulated error")
        return {"state": -1, "error": "failed"}
    monkeypatch.setattr(tm, "start", generate)
    target = tmp_path / "progress.json"
    assert cli.run_cli(["--video-subject", "test", "--progress-file", str(target)]) == 1
    snapshot = json.loads(target.read_text())
    assert (snapshot["status"], snapshot["clips_completed"], snapshot["percent"]) == ("failed", 1, 42)


def test_unwritable_progress_and_state_outage_do_not_escape(tmp_path, monkeypatch):
    state = MemoryState()
    target = tmp_path / "file-not-directory"
    target.write_text("existing")
    reporter = TaskProgress(target / "progress.json", "task", state, interval=0.01)
    reporter.start()
    reporter.finish(True)
    assert target.read_text() == "existing"
    monkeypatch.setattr(state, "get_task", lambda _: (_ for _ in ()).throw(OSError("offline")))
    reporter = TaskProgress(tmp_path / "new.json", "task", state)
    reporter.start()
    reporter.finish(False)


def test_progress_option_rejects_batch():
    with pytest.raises(SystemExit):
        cli.parse_args(["--batch-file", "tasks.json", "--progress-file", "progress.json"])


@pytest.mark.parametrize("block_on", [1, 2])
@pytest.mark.parametrize("succeeded", [False, True])
def test_blocked_state_read_does_not_block_start_or_finish(tmp_path, block_on, succeeded):
    entered, release, started, finished = (threading.Event() for _ in range(4))

    class BlockingState:
        calls = 0

        def get_task(self, _task_id):
            self.calls += 1
            if self.calls == block_on:
                entered.set()
                release.wait()
            return {"progress": 75, "stage": "render"}

    target = tmp_path / "progress.json"
    reporter = TaskProgress(target, "task", BlockingState(), interval=0.01)

    def start():
        reporter.start()
        started.set()

    def finish():
        reporter.finish(succeeded)
        finished.set()

    starter = threading.Thread(target=start)
    finisher = threading.Thread(target=finish)
    starter.start()
    try:
        assert started.wait(0.5), "progress startup blocked generation"
        assert entered.wait(2)
        finisher.start()
        assert finished.wait(1.5), "progress shutdown blocked the CLI result"
    finally:
        release.set()
        starter.join(timeout=2)
        if finisher.ident is not None:
            finisher.join(timeout=2)
        else:
            reporter.finish(succeeded)
        reporter._thread.join(timeout=2)
    snapshot = json.loads(target.read_text())
    assert snapshot["status"] == ("generated" if succeeded else "failed")
    assert not reporter._thread.is_alive()


def test_task_failure_preserves_progress_in_state_and_return_value(monkeypatch):
    state = MemoryState()
    state.update_task("task", progress=43, stage="materials", clips_completed=1, clips_total=3)
    monkeypatch.setattr(tm.sm, "state", state)
    result = tm._mark_task_failed("task", "materials", "download failed")
    assert result == state.get_task("task")
    assert result["clips_completed"] == 1 and result["clips_total"] == 3


@pytest.mark.parametrize("storyboard", [False, True])
def test_wan3_counts_only_validated_downloads_and_preserves_partial_count(monkeypatch, storyboard):
    calls, counts = [], []
    def generate(**kwargs):
        calls.append(kwargs["prompt"])
        if kwargs["prompt"] == "second":
            raise material.wan3.Wan3DownloadError("download exhausted", task_id="remote-2")
        yield "clip-1.mp4", 5.0, {}, "remote-1"
    monkeypatch.setattr(material, "_generate_wan3_materials", generate)
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)
    common = dict(task_id="local", video_aspect="9:16", material_directory="",
                  on_clip_completed=lambda done, total: counts.append((done, total)))
    with pytest.raises(material.wan3.Wan3DownloadError):
        if storyboard:
            plans = [SimpleNamespace(index=i+1, prompt=p, request_seconds=5,
                                     required_source_seconds=4, target_playback_seconds=4)
                     for i, p in enumerate(["first", "second"])]
            material._download_storyboard_wan3(shot_plan=plans, **common)
        else:
            material._download_videos_wan3_on_demand(search_terms=["first", "second"],
                                                    audio_duration=8, max_clip_duration=5, **common)
    assert counts == [(0, 2), (1, 2)]
    assert calls == ["first", "second"]


def test_short_storyboard_clip_is_not_counted_and_callback_errors_are_ignored(monkeypatch):
    monkeypatch.setattr(material, "_generate_wan3_materials", lambda **_k: iter([("clip", 1, {}, "remote")]))
    monkeypatch.setattr(material, "_persist_material_sources", lambda *_a: None)
    counts = []
    plan = SimpleNamespace(index=1, prompt="first", request_seconds=5,
                           required_source_seconds=4, target_playback_seconds=4)
    with pytest.raises(material.wan3.Wan3Error, match="too short"):
        material._download_storyboard_wan3(task_id="t", shot_plan=[plan], video_aspect="9:16",
                                         material_directory="", on_clip_completed=lambda *c: counts.append(c))
    assert counts == [(0, 1)]
    material._report_wan3_count(lambda *_a: (_ for _ in ()).throw(OSError("offline")), 1, 2)
