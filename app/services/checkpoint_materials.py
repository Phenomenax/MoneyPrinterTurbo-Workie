"""Checkpoint adapters for the two paid storyboard material providers."""

from pathlib import Path

from app.services import storyboard_checkpoint as checkpoints


def submitted(plan, remote_id):
    checkpoint = checkpoints.CURRENT.get()
    if checkpoint is not None and plan is not None:
        checkpoint.clip(plan.index, remote_id=remote_id, status="submitted")


def completed(plan, items):
    checkpoint = checkpoints.CURRENT.get()
    if checkpoint is not None and plan is not None:
        checkpoint.clip(
            plan.index,
            status="completed",
            items=checkpoints._pack(items),
            files={item[0]: checkpoints.digest(item[0]) for item in items},
        )


def prepare(requests, run_config, material_directory):
    """Resolve ALL previous submissions before dispatching any new work."""
    checkpoint = checkpoints.CURRENT.get()
    if checkpoint is None:
        return {}
    if run_config is None:
        raise checkpoints.CheckpointError(
            "checkpointed clips require environment video configuration"
        )
    from app.services import material, volcengine_seedance, wan3
    from app.utils import utils

    provider = wan3 if run_config.source == "wan3" else volcengine_seedance
    artifact_roots = (material_directory or utils.storage_dir("cache_videos"),)
    cached = {}
    for index, (prompt, seconds, plan) in enumerate(requests):
        if plan is None:
            raise checkpoints.CheckpointError(
                "only timed storyboard clips can be resumed"
            )
        identity = checkpoints.fingerprint([prompt, seconds, run_config.summary()])
        record = checkpoint.clip(plan.index)
        if record.get("identity") and record["identity"] != identity:
            raise checkpoints.CheckpointError(
                "shot inputs changed; refusing to resubmit original tasks"
            )
        if record.get("status") == "completed":
            if checkpoint.valid(record, artifact_roots=artifact_roots):
                cached[index] = checkpoints._unpack(record["items"])
                continue
            if not checkpoint.allow_rebuild:
                raise checkpoints.CheckpointError(
                    f"saved shot {plan.index} is missing or damaged; explicit rebuild approval required"
                )
        if record.get("status") in {"submitting", "submitted", "completed"}:
            remote_id = record.get("remote_id")
            if not remote_id:
                raise checkpoints.CheckpointError(
                    f"shot {plan.index} submission is unconfirmed and has no remote ID; do not resubmit"
                )
            try:
                item = provider.recover_video(remote_id, prompt, seconds, run_config)
            except (
                wan3.Wan3UnconfirmedTaskError,
                wan3.Wan3DownloadError,
                volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError,
                volcengine_seedance.VolcEngineSeedanceDownloadError,
            ):
                raise
            except (wan3.Wan3Error, volcengine_seedance.VolcEngineSeedanceError):
                checkpoint.clip(plan.index, status="failed")
                continue
            saved = material._save_generated_video_with_retry(
                item.url,
                material_directory,
                item.provider,
                network_options={
                    "proxies": run_config.proxy_map() or {},
                    "tls_verify": True
                    if run_config.tls_verify is None
                    else run_config.tls_verify,
                },
            )
            if not saved:
                raise checkpoints.CheckpointError(
                    f"original shot {plan.index} result could not be downloaded"
                )
            duration, _width, _height = material._measure_generated_video(saved)
            info = material._material_source_record(item, saved)
            info.update(duration=duration, storyboard_shot=plan.index)
            items = [(str(Path(saved).resolve()), duration, info, remote_id)]
            completed(plan, items)
            cached[index] = items
        elif not record:
            checkpoint.clip(plan.index, identity=identity, status="pending")
    return cached


def begin(plan):
    checkpoint = checkpoints.CURRENT.get()
    if checkpoint is not None and plan is not None:
        record = checkpoint.clip(plan.index)
        history = list(record.get("history", []))
        if record.get("remote_id"):
            history.append(record["remote_id"])
        checkpoint.clip(
            plan.index, status="submitting", remote_id=None, history=history
        )


def failed(plan, error):
    checkpoint = checkpoints.CURRENT.get()
    if checkpoint is None or plan is None:
        return
    from app.services import volcengine_seedance, wan3

    # A positively identified rejection/terminal failure can be retried only on
    # a later explicit batch continue. Other errors retain the submission fence.
    known = isinstance(
        error, (wan3.Wan3Error, volcengine_seedance.VolcEngineSeedanceError)
    )
    uncertain = isinstance(
        error,
        (
            wan3.Wan3UnconfirmedTaskError,
            wan3.Wan3DownloadError,
            volcengine_seedance.VolcEngineSeedanceUnconfirmedTaskError,
            volcengine_seedance.VolcEngineSeedanceDownloadError,
        ),
    )
    if known and not uncertain:
        checkpoint.clip(plan.index, status="failed")
