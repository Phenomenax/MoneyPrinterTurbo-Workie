from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.services import material, storyboard_checkpoint as cp, wan3


def test_checkpoint_reuses_completed_stage_and_requires_approval_for_damage(tmp_path):
    artifact = tmp_path / "audio.wav"
    calls = []

    def make():
        calls.append(1)
        artifact.write_bytes(b"audio")
        return str(artifact)

    with cp.scope(tmp_path, "same", enabled=True):
        assert cp.run("audio", make) == str(artifact)
    with cp.scope(tmp_path, "same", enabled=True, resume=True):
        assert cp.run("audio", make) == str(artifact)
    assert len(calls) == 1
    artifact.write_bytes(b"damaged")
    with cp.scope(tmp_path, "same", enabled=True, resume=True):
        with pytest.raises(cp.CheckpointError, match="rebuild approval"):
            cp.run("audio", make)
    with cp.scope(tmp_path, "same", enabled=True, resume=True, allow_rebuild=True):
        cp.run("audio", make)
    assert len(calls) == 2


def test_existing_inputs_and_concurrent_owner_are_protected(tmp_path):
    with cp.scope(tmp_path, "same", enabled=True):
        with pytest.raises(cp.CheckpointError, match="still running"):
            cp.Checkpoint(tmp_path, "same", resume=True)
    with pytest.raises(cp.CheckpointError, match="inputs changed"):
        cp.Checkpoint(tmp_path, "different", resume=True)
    with pytest.raises(cp.CheckpointError, match="explicit resume"):
        cp.Checkpoint(tmp_path, "same")


def test_submaker_roundtrip_keeps_word_timing():
    from datetime import timedelta
    from edge_tts import SubMaker
    from edge_tts.srt_composer import Subtitle

    maker = SubMaker()
    maker.cues = [Subtitle(1, timedelta(0), timedelta(seconds=1), "coffee")]
    restored = cp._unpack(cp._pack(maker))
    assert restored.get_srt() == maker.get_srt()


@dataclass
class Plan:
    index: int
    required_source_seconds: float = 2
    target_playback_seconds: float = 2
    remote_task_id: str | None = None
    material_seconds: float | None = None
    material_path: str | None = None


def schedule(tmp_path, generate):
    config = SimpleNamespace(source="wan3", summary=lambda: {"source": "wan3"})
    return material._schedule_generated_clips(
        task_id="local",
        requests=[("coffee", 2, Plan(1)), ("milk", 2, Plan(2))],
        generate=generate,
        provider_error=wan3.Wan3Error,
        provider_label="Wan3",
        concurrency=1,
        thread_name_prefix="checkpoint-test",
        progress_callback=None,
        on_clip_completed=None,
        checkpoint_config=config,
        material_directory=str(tmp_path),
    )


def test_resume_keeps_successful_shot_and_only_submits_failed_shot(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(material, "_persist_material_sources", lambda *args: None)
    calls = []
    fail_second = True

    def generate(prompt, seconds, plan, stop):
        calls.append(plan.index)
        if plan.index == 2 and fail_second:
            raise wan3.Wan3Error("explicit provider rejection")
        target = tmp_path / f"shot-{plan.index}.mp4"
        target.write_bytes(b"video")
        return [(str(target), 2.0, {}, f"remote-{plan.index}")]

    with cp.scope(tmp_path, "same", enabled=True):
        with pytest.raises(wan3.Wan3Error):
            schedule(tmp_path, generate)
    fail_second = False
    with cp.scope(tmp_path, "same", enabled=True, resume=True):
        assert len(schedule(tmp_path, generate)) == 2
    assert calls == [1, 2, 2]


def test_unknown_original_is_queried_before_any_new_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(material, "_persist_material_sources", lambda *args: None)
    calls = []

    def generate(prompt, seconds, plan, stop):
        calls.append(plan.index)
        cp.CURRENT.get().clip(plan.index, remote_id="original", status="submitted")
        raise wan3.Wan3UnconfirmedTaskError("unknown", task_id="original")

    with cp.scope(tmp_path, "same", enabled=True):
        with pytest.raises(wan3.Wan3UnconfirmedTaskError):
            schedule(tmp_path, generate)
    queried = []

    def recover(remote_id, *args):
        queried.append(remote_id)
        raise wan3.Wan3UnconfirmedTaskError("still unknown", task_id=remote_id)

    monkeypatch.setattr(wan3, "recover_video", recover)
    with cp.scope(tmp_path, "same", enabled=True, resume=True):
        with pytest.raises(wan3.Wan3UnconfirmedTaskError):
            schedule(tmp_path, generate)
    assert queried == ["original"]
    assert calls == [1]


def test_unconfirmed_submission_without_id_never_resubmits(tmp_path):
    with cp.scope(tmp_path, "same", enabled=True) as checkpoint:
        checkpoint.clip(1, status="submitting")
    with cp.scope(tmp_path, "same", enabled=True, resume=True):
        with pytest.raises(cp.CheckpointError, match="no remote ID"):
            schedule(tmp_path, lambda *args: pytest.fail("must not submit"))


def test_pause_file_fences_paid_submission(tmp_path, monkeypatch):
    from app.services.wan3_coordinator import SubmissionStopped, check_stopped

    marker = tmp_path / "pause"
    marker.touch()
    monkeypatch.setenv("MPT_BATCH_PAUSE_FILE", str(marker))
    with pytest.raises(SubmissionStopped):
        check_stopped(None)


@pytest.mark.parametrize("location", ["default_cache", "configured_cache"])
def test_resume_reuses_intact_cached_clip_outside_task_directory(tmp_path, monkeypatch, location):
    from app.services import checkpoint_materials
    from app.utils import utils

    task_dir = tmp_path / "tasks/local"
    cache_dir = tmp_path / location
    cache_dir.mkdir()
    video = cache_dir / "shot-1.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(utils, "storage_dir", lambda *args: str(tmp_path / "default_cache"))
    monkeypatch.setattr(material, "_persist_material_sources", lambda *args: None)
    config = SimpleNamespace(source="wan3", summary=lambda: {"source": "wan3"})
    saved = [(str(video), 2.0, {}, "original")]
    with cp.scope(task_dir, "same", enabled=True):
        checkpoint_materials.completed(Plan(1), saved)
    calls = []

    def generate(prompt, seconds, plan, stop):
        calls.append(plan.index)
        other = cache_dir / "shot-2.mp4"
        other.write_bytes(b"video")
        return [(str(other), 2.0, {}, "new-second")]

    with cp.scope(task_dir, "same", enabled=True, resume=True):
        result = material._schedule_generated_clips(
            task_id="local", requests=[("coffee", 2, Plan(1)), ("milk", 2, Plan(2))],
            generate=generate, provider_error=wan3.Wan3Error, provider_label="Wan3",
            concurrency=1, thread_name_prefix="checkpoint-cache", progress_callback=None,
            on_clip_completed=None, checkpoint_config=config,
            material_directory="" if location == "default_cache" else str(cache_dir),
        )
    assert len(result) == 2
    assert calls == [2]


@pytest.mark.parametrize("damage", ["corrupt", "missing", "outside_cache"])
def test_cached_clip_checks_still_reject_damage_and_unrelated_paths(tmp_path, monkeypatch, damage):
    from app.services import checkpoint_materials
    from app.utils import utils

    task_dir = tmp_path / "tasks/local"
    cache_dir = tmp_path / "cache_videos"
    directory = tmp_path / "unrelated" if damage == "outside_cache" else cache_dir
    directory.mkdir()
    video = directory / "shot.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(utils, "storage_dir", lambda *args: str(cache_dir))
    monkeypatch.setattr(wan3, "recover_video", lambda *args: pytest.fail("must require approval first"))
    with cp.scope(task_dir, "same", enabled=True):
        checkpoint_materials.completed(Plan(1), [(str(video), 2.0, {}, "original")])
    if damage == "corrupt":
        video.write_bytes(b"wrong")
    elif damage == "missing":
        video.unlink()
    config = SimpleNamespace(source="wan3", summary=lambda: {"source": "wan3"})
    with cp.scope(task_dir, "same", enabled=True, resume=True):
        with pytest.raises(cp.CheckpointError, match="rebuild approval"):
            checkpoint_materials.prepare([("coffee", 2, Plan(1))], config, "")


def test_stage_artifacts_remain_confined_to_task_directory(tmp_path):
    cached = tmp_path / "cache_videos/shot.mp4"
    cached.parent.mkdir()
    cached.write_bytes(b"video")
    with cp.scope(tmp_path / "tasks/local", "same", enabled=True):
        with pytest.raises(cp.CheckpointError, match="outside the task directory"):
            cp.run("render", lambda: str(cached))


def test_prepare_batches_initial_pending_clips_into_single_save(tmp_path, monkeypatch):
    from app.services import checkpoint_materials

    saves = []
    real_atomic_json = cp.atomic_json

    def counted_atomic_json(target, data):
        saves.append(len(data["clips"]))
        real_atomic_json(target, data)

    monkeypatch.setattr(cp, "atomic_json", counted_atomic_json)
    config = SimpleNamespace(source="wan3", summary=lambda: {"source": "wan3"})
    requests = [(f"prompt-{idx}", 2, Plan(idx)) for idx in range(1, 7)]
    with cp.scope(tmp_path / "tasks/local", "same", enabled=True):
        assert checkpoint_materials.prepare(requests, config, str(tmp_path)) == {}
        saved = cp.CURRENT.get().data["clips"]
        assert sorted(saved) == ["1", "2", "3", "4", "5", "6"]
        assert all(item["status"] == "pending" for item in saved.values())
    assert saves == [0, 6]
