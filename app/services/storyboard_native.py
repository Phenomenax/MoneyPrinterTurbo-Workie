"""Opt-in storyboard audio: deterministic requests and local assembly with audio.

Keep the planning API standard-library-only so Workie's standalone helper can
preflight every item without importing MPT startup or installing media packages.
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

NATIVE_STORYBOARD_API_VERSION = 1
FADE_SECONDS = 0.2
FPS = 30
SAMPLE_RATE = 48000
_NO_AUDIO_FAMILIES = {"seedance-1.0-pro", "seedance-1.0-pro-fast"}


class NativeStoryboardError(ValueError):
    def __init__(self, message, *, stage="preflight", shot=None, remote_id=None):
        self.stage, self.shot, self.remote_id = stage, shot, remote_id
        prefix = f"shot {shot}: " if shot is not None else ""
        suffix = f" (remote task {remote_id})" if remote_id else ""
        super().__init__(prefix + message + suffix)


@dataclass
class NativeShotPlan:
    index: int
    prompt: str
    planned_seconds: float
    request_seconds: int
    remote_task_id: str | None = None
    material_seconds: float = 0
    material_path: str = ""


def validate_native_arguments(arguments):
    """Reject explicit replacement audio/subtitles before any paid submission."""
    supplied = {item.split("=", 1)[0] for item in arguments if item.startswith("--")}
    conflicts = supplied & {
        "--custom-audio-file", "--video-music-prompt", "--sonilo-bgm-prompt",
        "--text-fore-color", "--custom-position", "--rounded-subtitle-background",
        "--no-rounded-subtitle-background", "--video-clip-duration",
    }
    conflicts.update(key for key in supplied if key.startswith((
        "--voice-", "--bgm-", "--subtitle-", "--no-subtitle-", "--font-", "--stroke-",
    )))
    if conflicts:
        raise NativeStoryboardError(
            "native storyboard audio conflicts with " + ", ".join(sorted(conflicts))
            + "; describe speech, music and captions in the storyboard instead"
        )

    def value(option):
        for index, token in enumerate(arguments):
            if token.startswith(option + "="):
                return token.split("=", 1)[1]
            if token == option and index + 1 < len(arguments):
                return arguments[index + 1]
        return None

    speed = value("--video-clip-speed")
    if speed is not None and (not math.isfinite(float(speed)) or float(speed) != 1):
        raise NativeStoryboardError("native storyboard audio requires --video-clip-speed 1")
    transition = value("--video-transition-mode")
    if transition is not None and transition != "none":
        raise NativeStoryboardError("native storyboard audio requires no overlapping transitions")
    if value("--video-count") not in {None, "1"}:
        raise NativeStoryboardError("native storyboard audio produces one video per storyboard")


def audio_run_config(run_config):
    if run_config is None or run_config.source not in {"wan3", "volcengine_seedance"}:
        raise NativeStoryboardError("native storyboard audio requires environment video configuration")
    # sends_generate_audio describes a Seedance payload switch, not Wan3 audio
    # capability. Unknown gateway names are tried with audio enabled.
    if run_config.source == "volcengine_seedance" and run_config.model_family in _NO_AUDIO_FAMILIES:
        raise NativeStoryboardError(f"{run_config.model_family} does not support native audio")
    return replace(run_config, generate_audio=True,
                   send_generate_audio=run_config.source == "volcengine_seedance")


def build_generation_plan(storyboard, run_config):
    audio_run_config(run_config)
    if not getattr(storyboard, "shots", None):
        raise NativeStoryboardError("native storyboard has no timed shots")
    if not hasattr(storyboard, "narration_voice") or not hasattr(storyboard, "captions_guidance"):
        raise NativeStoryboardError("native storyboard must use parse_native_storyboard")
    plans = []
    for position, shot in enumerate(storyboard.shots, start=1):
        milliseconds = round(shot.planned_end_seconds * 1000) - round(shot.planned_start_seconds * 1000)
        if milliseconds <= 0:
            raise NativeStoryboardError("planned duration must be positive", shot=shot.index)
        seconds = (milliseconds + 999) // 1000
        if not run_config.min_duration <= seconds <= run_config.max_duration:
            raise NativeStoryboardError(
                f"request {seconds}s is outside model range {run_config.min_duration}–"
                f"{run_config.max_duration}s; revise and approve the storyboard", shot=shot.index,
            )
        if shot.narration and (not storyboard.narration_voice.strip() or not storyboard.captions_guidance.strip()):
            raise NativeStoryboardError("spoken shots require Narration voice and Captions guidance", shot=shot.index)
        lines = [
            "Generate this single self-contained shot with synchronized video and native audio.",
            "Shot type and visual description:\n" + shot.prompt,
            f"Requested duration: {seconds} seconds. Keep the first and last 0.2 seconds free "
            "of narration or dialogue; finish the spoken line between these margins. "
            "Do not cut spoken words or captions at either boundary.",
            "Whole-video narration voice (description only, do not read aloud):\n"
            + (storyboard.narration_voice or "No narration."),
            "Apply this exact narrator profile to every spoken shot: keep the same perceived "
            "speaker, accent, age, pitch range, timbre, delivery, pace and recording character. "
            "Do not reinterpret or vary the narrator between segments.",
        ]
        if shot.narration:
            lines.append("Speak exactly the following line, without additions or omissions. "
                         "Do not speak instructions or headings:\n" + shot.narration)
        else:
            lines.append("No dialogue, narration, extra voices or narration captions in this shot.")
        if storyboard.music_mood:
            lines.append("Whole-video music design (context only, do not read aloud):\n"
                         + storyboard.music_mood)
        lines.append(
            f"Current music segment: {position} of {len(storyboard.shots)}. "
            f"Planned position in the whole video: {shot.planned_start_seconds:g}–"
            f"{shot.planned_end_seconds:g} seconds. Generate only this {seconds}-second segment; "
            "do not generate the whole soundtrack or compress all segments into this clip. "
            "Use the whole-video design to place this segment, not to add sounds from other shots. "
            "The current shot's concrete music / sound-effect instructions take precedence; "
            "if they specify no music or effects, do not add them from the whole-video design."
        )
        lines.append("Music / sound effects for this shot (do not read aloud):\n"
                     + (shot.music_sfx or "No background music and no additional sound effects."))
        if shot.narration:
            lines.append("Generate captions for only the exact spoken line above, following this guidance:\n"
                         + storyboard.captions_guidance)
        plans.append(NativeShotPlan(shot.index, "\n\n".join(lines), milliseconds / 1000, seconds))
    return plans


def inspect_clip(filename):
    """Decode both tracks completely. Digital silence is valid; missing samples are not."""
    import av

    starts, ends = {"video": [], "audio": []}, {"video": [], "audio": []}
    counts = {"video": 0, "audio": 0}
    elapsed = {"video": 0.0, "audio": 0.0}
    regular = {"video": True, "audio": True}
    with av.open(str(filename), options={"err_detect": "explode"}) as media:
        if not media.streams.video or not media.streams.audio:
            raise NativeStoryboardError("video has no usable video/audio track", stage="audio_validation")
        stream = media.streams.video[0]
        width, height = stream.width, stream.height
        rate = float(stream.average_rate or FPS)
        for frame in media.decode(video=0, audio=0):
            kind = "audio" if isinstance(frame, av.AudioFrame) else "video"
            if getattr(frame, "is_corrupt", False):
                raise NativeStoryboardError("corrupt media frame", stage="audio_validation")
            count = frame.samples if kind == "audio" else 1
            if count <= 0:
                continue
            length = frame.samples / frame.sample_rate if kind == "audio" else float(
                (getattr(frame, "duration", 0) or 0) * (frame.time_base or 0)
            ) or 1 / rate
            start = float(frame.time) if frame.time is not None else (ends[kind][-1] if ends[kind] else 0)
            if not math.isfinite(start) or not math.isfinite(length) or length <= 0:
                raise NativeStoryboardError("invalid media timestamps", stage="audio_validation")
            # Matroska, for example, rounds 30 fps timestamps to milliseconds.
            # Do not turn that timestamp quantization into one extra frame per
            # clip. Use decoded frame/sample counts for regular timelines, and
            # actual timestamps for variable-rate or discontinuous timelines.
            step = frame.samples / frame.sample_rate if kind == "audio" else 1 / rate
            tolerance = float(frame.time_base or 0) + 1e-7
            if starts[kind] and abs(start - starts[kind][0] - elapsed[kind]) > tolerance:
                regular[kind] = False
            if abs(length - step) > tolerance:
                regular[kind] = False
            elapsed[kind] += step
            starts[kind].append(start)
            ends[kind].append(start + length)
            counts[kind] += count
    if not all(counts.values()) or width <= 0 or height <= 0:
        raise NativeStoryboardError("video/audio track contains no valid frames or samples", stage="audio_validation")
    origin = min(min(starts["video"]), min(starts["audio"]))
    extent = {kind: starts[kind][0] + elapsed[kind] if regular[kind] else max(ends[kind])
              for kind in counts}
    duration = max(extent.values()) - origin
    return {"duration": duration, "width": width, "height": height,
            "video_offset": min(starts["video"]) - origin,
            "audio_offset": min(starts["audio"]) - origin, "audio_samples": counts["audio"]}


def render_native(clips, plans, output_file, *, aspect, threads=2, fit_mode="cover"):
    """Normalize to frame-aligned PCM intermediates; encode AAC once after concatenation."""
    from app.models.schema import VideoAspect
    from app.services import video
    from app.utils import utils

    if len(clips) != len(plans) or not clips:
        raise NativeStoryboardError("native material count does not match the shots", stage="combine")
    # Validate ALL downloaded outputs before writing a final video. These files
    # are already completed checkpoints: local failures never authorize resubmission.
    media_info = []
    for clip, plan in zip(clips, plans, strict=True):
        try:
            media_info.append(inspect_clip(clip))
        except Exception as exc:
            raise NativeStoryboardError(str(exc), stage="audio_validation", shot=plan.index,
                                        remote_id=plan.remote_task_id) from exc
    width, height = VideoAspect(aspect).to_resolution()
    root = Path(output_file).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-render-", dir=root) as temporary:
        normalized = []
        frame_counts = []
        for index, (clip, info) in enumerate(zip(clips, media_info, strict=True)):
            duration = info["duration"]
            frames = max(1, math.ceil(duration * FPS - 1e-7))
            frame_counts.append(frames)
            padded_seconds = frames / FPS
            samples = frames * (SAMPLE_RATE // FPS)
            fade = min(FADE_SECONDS, duration / 2)
            scale = (f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
                     if fit_mode == "cover" else
                     f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            visual_filter = (
                f"setpts=PTS-STARTPTS,{scale},setsar=1,"
                f"tpad=start_duration={info['video_offset']:.9f}:start_mode=clone,"
                f"fps={FPS},tpad=stop_mode=clone:stop_duration={padded_seconds:.9f},"
                f"trim=end_frame={frames},setpts=PTS-STARTPTS"
            )
            audio_filter = (
                f"asetpts=PTS-STARTPTS,aresample={SAMPLE_RATE}:async=1:first_pts=0,"
                f"aformat=sample_fmts=s16:channel_layouts=stereo,"
                f"adelay={round(info['audio_offset'] * SAMPLE_RATE)}S:all=1,"
                f"afade=t=in:st=0:d={fade:.9f},"
                f"afade=t=out:st={duration - fade:.9f}:d={fade:.9f},"
                f"apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB"
            )
            target = str(Path(temporary) / f"shot-{index:04d}.mkv")
            command = [utils.get_ffmpeg_binary(), "-y", "-v", "error", "-i", str(clip),
                       "-map", "0:v:0", "-map", "0:a:0", "-vf", visual_filter,
                       "-af", audio_filter, "-c:v", "libx264", "-preset", "fast",
                       "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE),
                       "-ac", "2", "-threads", str(threads or 2), target]
            result = video._run_concat_with_heartbeat(command, target)
            if result.returncode:
                raise NativeStoryboardError("native normalization failed: " + (result.stderr or ""),
                                            stage="combine", shot=plans[index].index,
                                            remote_id=plans[index].remote_task_id)
            normalized.append(target)
        staged = str(Path(temporary) / "final.mp4")
        video.concat_video_clips_with_ffmpeg(normalized, staged, threads, temporary,
                                            stream_copy=True, preserve_audio=True)
        final_info = inspect_clip(staged)
        os.replace(staged, output_file)
    return {"videos": [str(Path(output_file).resolve())], "combined_videos": [],
            "actual_seconds": [info["duration"] for info in media_info],
            "frame_counts": frame_counts, "output_seconds": final_info["duration"]}


def run(task_id, params, storyboard, run_config, *, stop_at="video", explicit_clip_duration=None):
    from app.models import const
    from app.services import generation_progress, state, storyboard_checkpoint, task, task_artifacts
    from app.utils import utils

    try:
        if stop_at != "video" or explicit_clip_duration is not None or params.video_count != 1:
            raise NativeStoryboardError("native audio requires a complete single storyboard video without fixed clip duration")
        if params.video_clip_speed != 1:
            raise NativeStoryboardError("native audio requires playback speed 1")
        audio_config = audio_run_config(run_config)
        if audio_config.source != params.video_source:
            raise NativeStoryboardError("environment video source does not match the selected source")
        plans = build_generation_plan(storyboard, audio_config)
        if not utils.check_ffmpeg_ready():
            raise NativeStoryboardError("FFmpeg is unavailable")
    except (ValueError, OSError) as exc:
        return task._mark_task_failed(task_id, "preflight", str(exc))

    prompts = [plan.prompt for plan in plans]
    task.save_script_data(task_id, storyboard.video_script, prompts, params)

    def diagnostics():
        task_artifacts.patch_script_data(
            task_id, storyboard_audio_mode="native",
            planned_total_seconds=sum(plan.planned_seconds for plan in plans),
            request_total_seconds=sum(plan.request_seconds for plan in plans),
            native_shots=[dict(index=plan.index, prompt=plan.prompt,
                               planned_seconds=plan.planned_seconds, request_seconds=plan.request_seconds,
                               actual_seconds=plan.material_seconds, material_path=plan.material_path,
                               remote_task_id=plan.remote_task_id) for plan in plans],
        )

    diagnostics()
    generation_progress.stage("materials")
    state.state.update_task(task_id, stage="materials", progress=40,
                            clips_completed=0, clips_total=len(plans), storyboard_audio_mode="native")
    try:
        clips = task.get_video_materials(task_id, params, prompts, 0, shot_plan=plans,
                                         video_run_config=audio_config)
        diagnostics()
        if not clips:
            return task._mark_task_failed(task_id, "materials", "no native materials")
        generation_progress.stage("rendering")
        state.state.update_task(task_id, stage="combine", progress=50, materials=clips)
        rendered = storyboard_checkpoint.run("render", lambda: render_native(
            clips, plans, str(Path(utils.task_dir(task_id)) / "final-1.mp4"),
            aspect=params.video_aspect, threads=params.n_threads, fit_mode=params.video_fit_mode,
        ))
        if rendered.get("frame_counts"):
            from app.services.storyboard_revision import publish_manifest

            publish_manifest(Path(utils.task_dir(task_id)), rendered, clips, plans,
                             audio_config, params.video_aspect)
        for plan, actual in zip(plans, rendered["actual_seconds"], strict=True):
            plan.material_seconds = actual
        diagnostics()
        task_artifacts.patch_script_data(task_id, actual_total_seconds=sum(rendered["actual_seconds"]),
                                         output_seconds=rendered["output_seconds"])
        result = dict(state=const.TASK_STATE_COMPLETE, videos=rendered["videos"], combined_videos=[],
                      script=storyboard.video_script, terms=prompts, materials=clips,
                      storyboard_audio_mode="native", audio_file="", subtitle_path="",
                      audio_duration=rendered["output_seconds"])
        state.state.update_task(task_id, **result, progress=100, stage="complete")
        return result
    except Exception as exc:
        diagnostics()
        return task._mark_task_failed(task_id, getattr(exc, "stage", "combine"), str(exc),
                                      details={"materials": [plan.material_path for plan in plans if plan.material_path],
                                               "storyboard_audio_mode": "native"})
