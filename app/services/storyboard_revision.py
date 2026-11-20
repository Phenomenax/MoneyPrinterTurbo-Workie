"""Native shot provenance and read-only preflight for the Workie revision protocol."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from app.models.schema import VideoAspect
from app.services import storyboard_checkpoint as cp
from app.services import storyboard_native as native
from app.services.wan3_coordinator import atomic_json

REVISION_PROTOCOL = 1
MANIFEST = "native-shots-v1.json"


def request_hash(plan, config, aspect):
    from app.services import volcengine_seedance, wan3

    aspect = VideoAspect(aspect)
    if config.source == "wan3":
        payload = wan3.video_payload(
            config.model,
            config.resolution,
            aspect,
            plan.prompt.strip(),
            plan.request_seconds,
            config.generate_audio,
        )
    else:
        payload = volcengine_seedance._video_payload(
            plan.prompt.strip(),
            plan.request_seconds,
            aspect,
            volcengine_seedance.endpoint_from_run_config(config),
        )
    return cp.fingerprint(
        [config.source, config.endpoint, config.model_family, payload]
    )


def copy_clip(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and cp.digest(target) == expected:
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        if cp.digest(temporary) != expected:
            raise ValueError("镜头素材在复制时发生变化，请重新预检。")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_manifest(root, rendered, clips, plans, config, aspect):
    shots = []
    start = 0
    for path, plan, frames in zip(clips, plans, rendered["frame_counts"], strict=True):
        relative = f"native-clips/{plan.index}.mp4"
        sha = cp.digest(path)
        copy_clip(path, root / relative, sha)
        shots.append(
            dict(
                index=plan.index,
                start_frame=start,
                end_frame=start + frames,
                path=relative,
                sha256=sha,
                remote_id=plan.remote_task_id,
                request=request_hash(plan, config, aspect),
            )
        )
        start += frames
    atomic_json(
        root / MANIFEST,
        dict(
            version=REVISION_PROTOCOL,
            fps=native.FPS,
            final_sha256=cp.digest(rendered["videos"][0]),
            shots=shots,
        ),
    )


def read_source(tasks, source):
    ident = source["task_id"]
    if not isinstance(ident, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", ident):
        raise ValueError("源任务编号无效。")
    root = (tasks / ident).resolve()
    if root.parent != tasks.resolve():
        raise ValueError("源任务位置无效。")
    manifest = root / MANIFEST
    if not manifest.is_file() or (
        source.get("manifest_sha256")
        and cp.digest(manifest) != source["manifest_sha256"]
    ):
        return root, None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            data["version"] == REVISION_PROTOCOL
            and data["final_sha256"] == source["final_sha256"]
        ):
            return root, data
    except (KeyError, ValueError, TypeError):
        pass
    return root, None


def checked_clip(path, expected):
    try:
        if path.is_file() and cp.digest(path) == expected:
            return native.inspect_clip(path)
    except Exception:
        pass
    return None


def available_clip(root, shot, plan, config):
    relative = shot["path"]
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("源镜头路径无效。")
    info = checked_clip(path, shot["sha256"])
    if info:
        return path, info
    # Recovery cache is outside the immutable parent task, keyed by content.
    if not re.fullmatch(r"[a-f0-9]{64}", shot["sha256"]):
        raise ValueError("源镜头摘要无效。")
    cache = root.parent.parent / "revision-recovery" / shot["sha256"]
    recovered = cache / "clip.mp4"
    info = checked_clip(recovered, shot["sha256"])
    if info:
        return recovered, info
    if shot.get("remote_id"):
        from app.services import material, volcengine_seedance, wan3

        provider = wan3 if config.source == "wan3" else volcengine_seedance
        try:
            item = provider.recover_video(
                shot["remote_id"], plan.prompt, plan.request_seconds, config
            )
            cache.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                saved = material._save_generated_video_with_retry(
                    item.url,
                    temporary,
                    item.provider,
                    network_options={
                        "proxies": config.proxy_map() or {},
                        "tls_verify": config.tls_verify is not False,
                    },
                )
                info = checked_clip(Path(saved), shot["sha256"]) if saved else None
                if info:
                    copy_clip(saved, recovered, shot["sha256"])
                    return recovered, info
        except Exception:
            pass
    return None


def preflight(tasks, source, board, config, aspect, mode, targets):
    root, manifest = read_source(tasks, source)
    if mode not in {"partial", "full"}:
        raise ValueError("修改范围无效。")
    if mode == "partial" and manifest is None:
        raise ValueError("这条历史成片没有可信镜头清单，只能整片重做。")
    config = native.audio_run_config(config)
    plans = native.build_generation_plan(board, config)
    indices = [plan.index for plan in plans]
    if mode == "partial" and (
        not targets
        or not set(targets) <= set(indices)
        or [shot["index"] for shot in manifest["shots"]] != indices
    ):
        raise ValueError("局部修改需要有效的镜头编号并保持镜头数量和顺序。")
    rows, materials = [], {}
    for plan in plans:
        request = request_hash(plan, config, aspect)
        row = dict(index=plan.index, request=request, action="generate")
        if mode == "full" or plan.index in targets:
            row["reason"] = "requested"
        else:
            shot = manifest["shots"][plan.index - 1]
            if shot["request"] != request:
                row["reason"] = "request_changed"
            else:
                material = available_clip(root, shot, plan, config)
                if material:
                    materials[plan.index] = (*material, shot.get("remote_id"))
                    row.update(
                        reason="unchanged", action="reuse", sha256=shot["sha256"]
                    )
                else:
                    row["reason"] = "missing_material"
        rows.append(row)
    frozen = dict(
        version=REVISION_PROTOCOL,
        installation=cp.fingerprint(str(tasks.resolve())),
        configuration=cp.fingerprint(config.summary()),
        source=source,
        manifest=cp.fingerprint(manifest),
        mode=mode,
        targets=sorted(set(targets)),
        shots=rows,
    )
    return frozen, materials


def stage_revision(checkpoint, frozen, board, config, aspect):
    tasks = checkpoint.root.parent
    plans = native.build_generation_plan(board, config)
    config = native.audio_run_config(config)
    if (
        frozen["installation"] != cp.fingerprint(str(tasks.resolve()))
        or frozen.get("configuration") != cp.fingerprint(config.summary())
        or [request_hash(plan, config, aspect) for plan in plans]
        != [row["request"] for row in frozen["shots"]]
    ):
        raise ValueError("生成配置或分镜已变化，请重新预检。")
    if checkpoint.data.get("revision_staged"):
        for row in frozen["shots"]:
            if row["action"] != "reuse":
                continue
            saved = checkpoint.clip(row["index"])
            expected = checkpoint.root / "revision-clips" / f"{row['index']}.mp4"
            if (
                saved.get("status") != "completed"
                or not saved.get("items")
                or saved.get("files") != {str(expected): row["sha256"]}
                or not checkpoint.valid(saved)
            ):
                raise ValueError("已固定的复用素材损坏，请重新预检。")
        return
    current, materials = preflight(
        tasks,
        frozen["source"],
        board,
        config,
        aspect,
        frozen["mode"],
        frozen["targets"],
    )
    if current != frozen:
        raise ValueError("预检后的素材或生成配置已变化，请重新预检并确认范围。")
    for row in current["shots"]:
        index = row["index"]
        if index not in materials:
            continue
        source, info, remote = materials[index]
        path = checkpoint.root / "revision-clips" / f"{index}.mp4"
        copy_clip(source, path, row["sha256"])
        items = [(str(path), info["duration"], {"storyboard_shot": index}, remote)]
        checkpoint.clip(
            index,
            status="completed",
            remote_id=remote,
            reused=True,
            files={str(path): row["sha256"]},
            items=cp._pack(items),
        )
    checkpoint.data["revision_staged"] = True
    checkpoint.data["revision_shots"] = current["shots"]
    checkpoint.save()


def main():
    """stdin/stdout JSON. Runs in the actual MPT installation with its media dependencies."""
    from app.services.storyboard import parse_native_storyboard
    from cli import _video_run_config_from_environment

    payload = json.load(sys.stdin)
    board = parse_native_storyboard(payload["board"])
    config = _video_run_config_from_environment(payload["provider"], board.resolution)
    from app.utils import utils

    original = payload.get("provider_summary", {})
    if any(
        original.get(key, getattr(config, key)) != getattr(config, key)
        for key in ("source", "model", "model_family", "endpoint")
    ):
        raise ValueError("供应商或模型配置已变化，请恢复原设置后重新预检。")
    frozen, _ = preflight(
        Path(utils.task_dir()).resolve(),
        payload["source"],
        board,
        config,
        payload["aspect"],
        payload["mode"],
        payload["targets"],
    )
    print("MPT_REVISION=" + json.dumps(frozen, ensure_ascii=True))


if __name__ == "__main__":
    main()
