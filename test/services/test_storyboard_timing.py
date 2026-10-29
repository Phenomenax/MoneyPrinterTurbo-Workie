"""Per-shot storyboard timing: parse, measure, request, and local render."""
from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import MaterialInfo, VideoAspect, VideoParams
from app.services import material, storyboard_timing, video
from app.services import task as tm
from app.services.storyboard import Storyboard, StoryboardShot, parse_storyboard
from app.services.storyboard_timing import (
    FRAME_SECONDS,
    PCM_SAMPLE_RATE,
    ShotTimeline,
    StoryboardTimeline,
    build_generation_plan,
    concat_pcm_wavs,
    estimate_narration_seconds,
    shift_subtitle_cues,
    synthesize_timeline,
    write_silence,
    write_storyboard_subtitles,
)
from app.services.wan3 import Wan3DownloadError, Wan3UnconfirmedTaskError
from app.utils import utils


def _board(beats: str, script: str, *, title: str = "示例") -> str:
    return (
        f"# 🎬 Storyboard — {title}\n\n"
        "**Audience:** 大家  **Duration:** 10s  **Tone:** 平静  **Aspect:** 9:16\n\n"
        "## Beats (timed)\n\n"
        "| Time | Shot type | On-screen / B-roll | VO / on-camera line | Music / SFX |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{beats.rstrip()}\n\n"
        "## Voiceover script (clean, ready to paste)\n\n"
        f"{script.strip()}\n"
    )


def _write_board(directory: str, text: str, name: str = "storyboard.md") -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _shot(
    index: int,
    playback: float,
    *,
    planned: float = 5.0,
    narration: str = "台词",
    prompt: str = "Wide: 画面",
) -> ShotTimeline:
    samples = int(round(playback * PCM_SAMPLE_RATE))
    return ShotTimeline(
        index=index,
        prompt=f"{prompt} {index}",
        narration=narration,
        planned_start_seconds=(index - 1) * planned,
        planned_end_seconds=index * planned,
        planned_duration_seconds=planned,
        estimated_seconds=None,
        sample_count=samples,
        actual_seconds=samples / PCM_SAMPLE_RATE,
        calibrated_start_seconds=(index - 1) * playback,
        calibrated_end_seconds=index * playback,
        target_playback_seconds=playback,
        audio_path="",
        tts_path=None,
    )


def _timeline(*shots: ShotTimeline) -> StoryboardTimeline:
    total = shots[-1].calibrated_end_seconds if shots else 0.0
    return StoryboardTimeline(
        shots=list(shots),
        audio_path="audio.wav",
        audio_samples=int(round(total * PCM_SAMPLE_RATE)),
        planned_total_seconds=sum(shot.planned_duration_seconds for shot in shots),
        calibrated_total_seconds=total,
    )


def _pcm(path: str, samples: int) -> None:
    write_silence(samples / PCM_SAMPLE_RATE, path)


class StoryboardParseTests(unittest.TestCase):
    def test_times_decimals_escaped_pipe_and_multiline_line_are_kept(self):
        text = _board(
            "\n".join([
                "| 0:00-0:05.5 | Wide | 房间里有一根 \\| 管子 | 你好<br>世界 | bed |",
                "| 0:05.5–0:08.25 | Close | 杯子 | “下一句。” | bed |",
                "| 0:08.25—0:00:10 | End | 结尾 | - | outro |",
            ]),
            "你好\n世界\n“下一句。”",
        )
        with tempfile.TemporaryDirectory() as tmp:
            parsed = parse_storyboard(_write_board(tmp, text), require_shot_timing=True)
        self.assertEqual(
            [shot.narration for shot in parsed.shots],
            ["你好\n世界", "“下一句。”", ""],
        )
        self.assertEqual(parsed.video_terms[0], "Wide: 房间里有一根 | 管子")
        self.assertAlmostEqual(parsed.shots[0].planned_end_seconds, 5.5)
        self.assertAlmostEqual(parsed.shots[1].planned_start_seconds, 5.5)
        self.assertAlmostEqual(parsed.shots[1].planned_end_seconds, 8.25)
        self.assertAlmostEqual(parsed.shots[2].planned_end_seconds, 10.0)
        self.assertEqual(parsed.video_script.replace(" ", ""), "你好\n世界\n“下一句。”".replace(" ", ""))
        self.assertEqual([shot.prompt for shot in parsed.shots], parsed.video_terms)

    def test_one_millisecond_boundary_is_continuous(self):
        text = _board(
            "| 0-5.000 | Wide | 房间 | 你好 | bed |\n| 5.001-6 | Close | 杯子 | 世界 | bed |",
            "你好世界",
        )
        with tempfile.TemporaryDirectory() as tmp:
            parsed = parse_storyboard(_write_board(tmp, text), require_shot_timing=True)
        self.assertEqual(len(parsed.shots), 2)

    def test_overlap_gap_reverse_zero_and_placeholder_include_line_numbers(self):
        cases = {
            "overlap": (
                "| 0:00-0:05 | Wide | 房间 | 你好 | bed |\n| 0:04-0:08 | Close | 杯子 | 世界 | bed |",
                "overlaps",
            ),
            "gap": (
                "| 0:00-0:05 | Wide | 房间 | 你好 | bed |\n| 0:06-0:08 | Close | 杯子 | 世界 | bed |",
                "gap",
            ),
            "reverse": (
                "| 0:05-0:00 | Wide | 房间 | 你好 | bed |",
                "reversed",
            ),
            "zero": (
                "| 0:00-0:00 | Wide | 房间 | 你好 | bed |",
                "zero length",
            ),
            "placeholder": (
                "| 0:00-0:05 | Wide | TODO | 你好 | bed |",
                "placeholder",
            ),
        }
        for name, (beats, needle) in cases.items():
            text = _board(beats, "你好世界" if name != "reverse" and name != "zero" else "你好")
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError) as caught:
                    parse_storyboard(_write_board(tmp, text), require_shot_timing=True)
            self.assertIn("line", str(caught.exception).lower(), name)
            self.assertIn(needle, str(caught.exception).lower(), name)

    def test_voiceover_must_match_shot_lines_but_ignores_quotes_and_whitespace(self):
        beats = "| 0:00-0:05 | Wide | 房间 | \"你好。\" | bed |"
        with tempfile.TemporaryDirectory() as tmp:
            matched = parse_storyboard(
                _write_board(tmp, _board(beats, "> 你好。")),
                require_shot_timing=True,
            )
            self.assertEqual(matched.shots[0].narration, '"你好。"')
            with self.assertRaises(ValueError) as caught:
                parse_storyboard(
                    _write_board(tmp, _board(beats, "你好")),
                    require_shot_timing=True,
                )
        self.assertIn("does not match", str(caught.exception))

    def test_silent_shots_and_a_fully_silent_board(self):
        mixed = _board(
            "| 0:00-0:02 | Title | 标题 | — | bed |\n| 0:02-0:05 | Wide | 房间 | 你好 | bed |",
            "你好",
        )
        silent = _board(
            "| 0:00-0:02 | Title | 标题 | — | bed |\n| 0:02-0:04 | End | 结尾 | - | outro |",
            "—",
        )
        with tempfile.TemporaryDirectory() as tmp:
            mixed_board = parse_storyboard(_write_board(tmp, mixed), require_shot_timing=True)
            silent_board = parse_storyboard(_write_board(tmp, silent), require_shot_timing=True)
        self.assertEqual(mixed_board.shots[0].narration, "")
        self.assertEqual(mixed_board.shots[1].narration, "你好")
        self.assertEqual(silent_board.video_script, "")
        self.assertTrue(all(shot.narration == "" for shot in silent_board.shots))

    def test_loose_parse_keeps_prompts_when_timing_is_invalid(self):
        text = _board("| later-earlier | Wide | 房间 | 你好 | bed |", "你好")
        with tempfile.TemporaryDirectory() as tmp:
            parsed = parse_storyboard(_write_board(tmp, text), require_shot_timing=False)
        self.assertEqual(parsed.video_terms, ["Wide: 房间"])
        self.assertEqual(parsed.shots, ())
        self.assertEqual(Storyboard("旁白", ["画面"]).shots, ())


class StoryboardTimingTests(unittest.TestCase):
    def test_estimate_is_deterministic_and_does_not_reject_a_long_line(self):
        self.assertAlmostEqual(estimate_narration_seconds("你好世界", 1.0), 1.0)
        self.assertAlmostEqual(estimate_narration_seconds("one two three four five", 1.0), 2.0)
        self.assertAlmostEqual(estimate_narration_seconds("你好世界", 2.0), 0.5)
        self.assertAlmostEqual(estimate_narration_seconds("你好[pause: 1s]世界", 1.0), 2.0)
        self.assertGreater(estimate_narration_seconds("字" * 400, 1.0), 30)

    def test_measured_audio_sets_playback_and_wan3_request_lengths(self):
        cases = (
            (7.2, 1.0, None, 8, 7.2),
            (3.0, 1.0, None, 3, 3.0),
            (1.2, 1.0, None, 2, 1.2),
            (7.0, 2.0, None, 14, 14.0),
            (7.0, 0.5, None, 4, 3.5),
        )
        for playback, speed, explicit, request, source in cases:
            timeline = _timeline(_shot(1, playback))
            build_generation_plan(
                timeline, clip_speed=speed, explicit_clip_duration=explicit,
                **material.storyboard_generation_options("wan3"),
            )
            shot = timeline.shots[0]
            self.assertEqual(shot.request_seconds, request)
            self.assertAlmostEqual(shot.required_source_seconds, source)
            self.assertAlmostEqual(shot.target_playback_seconds, playback)

    def test_explicit_duration_forms_and_insufficient_length(self):
        timeline = _timeline(_shot(1, 7.2, narration="这句比较长"))
        with self.assertRaises(storyboard_timing.StoryboardTimingError) as caught:
            build_generation_plan(
                timeline, clip_speed=1.0, explicit_clip_duration=5,
                **material.storyboard_generation_options("wan3"),
            )
        message = str(caught.exception)
        self.assertEqual(caught.exception.stage, "materials")
        self.assertIn("shot 1", message)
        self.assertIn("--video-clip-duration", message)
        self.assertIn("本次尚未提交 Wan3 视频生成", message)
        self.assertNotIn("没有任何费用", message)

    def test_any_shot_over_thirty_seconds_is_rejected_with_measured_details(self):
        timeline = _timeline(_shot(1, 4.0), _shot(2, 31.0, narration="很长"))
        with self.assertRaises(storyboard_timing.StoryboardTimingError) as caught:
            build_generation_plan(
                timeline, clip_speed=1.0, explicit_clip_duration=None,
                **material.storyboard_generation_options("wan3"),
            )
        message = str(caught.exception)
        self.assertEqual(caught.exception.stage, "materials")
        self.assertIn("shot 2", message)
        self.assertIn("31", message)
        self.assertIn("本次尚未提交 Wan3 视频生成", message)
        self.assertEqual(timeline.shots[0].request_seconds, 4)
        self.assertEqual(timeline.shots[1].request_seconds, 31)

    def test_timeline_uses_replacement_source_limits(self):
        timeline = _timeline(_shot(1, 31.0), _shot(2, 0.8))
        build_generation_plan(
            timeline, clip_speed=1.0, explicit_clip_duration=None,
            minimum_clip_seconds=1, maximum_clip_seconds=60, provider_name="Replacement",
        )
        self.assertEqual([shot.request_seconds for shot in timeline.shots], [31, 1])

    def test_shot_order_and_pcm_offsets_do_not_follow_mp3_durations(self):
        durations = {"第一句。": 0.5, "第二句。": 0.2}
        calls = []

        def fake_tts(**kwargs):
            calls.append(kwargs["text"])
            samples = int(round(durations[kwargs["text"]] * PCM_SAMPLE_RATE))
            _pcm(kwargs["voice_file"], samples)
            return SimpleNamespace(cues=())

        shots = (
            StoryboardShot(1, "Wide: 甲", "第一句。", 0.0, 5.0),
            StoryboardShot(2, "Close: 乙", "第二句。", 5.0, 10.0),
            StoryboardShot(3, "End: 丙", "", 10.0, 10.4),
        )
        with tempfile.TemporaryDirectory() as tmp, patch("app.services.storyboard_timing.voice.tts", side_effect=fake_tts):
            timeline = synthesize_timeline(
                task_dir=tmp,
                shots=shots,
                voice_name="zh-CN-XiaoxiaoNeural-Female",
                voice_rate=1.0,
            )
            with wave.open(timeline.audio_path, "rb") as handle:
                joined_samples = handle.getnframes()
        self.assertEqual(joined_samples, 12000 + 4800 + int(round(0.4 * PCM_SAMPLE_RATE)))
        self.assertEqual(calls, ["第一句。", "第二句。"])
        self.assertEqual([shot.prompt for shot in timeline.shots], ["Wide: 甲", "Close: 乙", "End: 丙"])
        self.assertEqual([shot.narration for shot in timeline.shots], ["第一句。", "第二句。", ""])
        self.assertAlmostEqual(timeline.shots[1].calibrated_start_seconds, 12000 / PCM_SAMPLE_RATE)
        self.assertEqual(timeline.shots[1].sample_count, 4800)
        self.assertAlmostEqual(timeline.shots[0].target_playback_seconds, 0.5)
        self.assertAlmostEqual(timeline.shots[1].target_playback_seconds, 0.2)
        self.assertAlmostEqual(timeline.shots[2].actual_seconds, 0.4)
        misleading_mp3_duration = 1.0
        shifted = shift_subtitle_cues(
            [(0.0, 0.2, "第二句。")],
            offset_seconds=timeline.shots[1].sample_count and timeline.shots[1].calibrated_start_seconds,
            shot_duration_seconds=timeline.shots[1].actual_seconds,
        )
        self.assertNotAlmostEqual(shifted[0][0], misleading_mp3_duration)
        self.assertAlmostEqual(shifted[0][0], 0.5)
        self.assertAlmostEqual(shifted[0][1], 0.7)

    def test_subtitle_cues_stay_inside_each_shot_and_are_renumbered(self):
        first = _shot(1, 0.5, narration="第一句")
        second = _shot(2, 0.2, narration="第二句")
        second.calibrated_start_seconds = 0.5
        second.calibrated_end_seconds = 0.7
        silent = _shot(3, 0.4, narration="")
        silent.calibrated_start_seconds = 0.7
        silent.calibrated_end_seconds = 1.1
        timeline = _timeline(first, second, silent)
        written = []

        def fake_create_subtitle(**kwargs):
            written.append(kwargs["text"])
            with open(kwargs["subtitle_file"], "w", encoding="utf-8") as handle:
                handle.write("1\n00:00:00,000 --> 00:00:05,000\n" + kwargs["text"] + "\n")

        with tempfile.TemporaryDirectory() as tmp:
            for shot in timeline.shots:
                shot.audio_path = os.path.join(tmp, f"shot-{shot.index}.wav")
                shot.sub_maker = object()
            output = os.path.join(tmp, "subtitle.srt")
            with patch("app.services.storyboard_timing.voice.create_subtitle", side_effect=fake_create_subtitle):
                path = write_storyboard_subtitles(
                    timeline,
                    subtitle_path=output,
                    subtitle_enabled=True,
                    subtitle_provider="edge",
                    word_level=False,
                )
            text = Path(path).read_text(encoding="utf-8")
        self.assertEqual(written, ["第一句", "第二句"])
        self.assertIn("1\n00:00:00,000 --> 00:00:00,500\n第一句", text)
        self.assertIn("2\n00:00:00,500 --> 00:00:00,700\n第二句", text)
        self.assertNotIn("00:00:05", text)

    def test_missing_or_out_of_range_shot_captions_do_not_publish_partial_subtitles(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                first = _shot(1, 1.0, narration="第一句")
                second = _shot(2, 1.0, narration="第二句")
                for shot in (first, second):
                    shot.audio_path = os.path.join(tmp, f"shot-{shot.index}.wav")
                    shot.sub_maker = object()
                output = os.path.join(tmp, "subtitle.srt")

                def fake_create_subtitle(**kwargs):
                    if kwargs["text"] == "第二句" and missing:
                        return
                    start, end = ("05", "06") if kwargs["text"] == "第二句" else ("00", "01")
                    Path(kwargs["subtitle_file"]).write_text(
                        f"1\n00:00:{start},000 --> 00:00:{end},000\n{kwargs['text']}\n",
                        encoding="utf-8",
                    )

                with patch("app.services.storyboard_timing.voice.create_subtitle", side_effect=fake_create_subtitle):
                    with self.assertRaises(storyboard_timing.StoryboardTimingError) as caught:
                        write_storyboard_subtitles(
                            _timeline(first, second), subtitle_path=output,
                            subtitle_enabled=True, subtitle_provider="edge", word_level=False,
                        )
                self.assertEqual(caught.exception.stage, "subtitle")
                self.assertIn("shot 2", str(caught.exception))
                self.assertFalse(os.path.exists(output))

    def test_whisper_uses_the_shot_audio_and_edge_does_not_switch_provider(self):
        shot = _shot(1, 0.5, narration="第一句")
        timeline = _timeline(shot)
        with tempfile.TemporaryDirectory() as tmp:
            shot.audio_path = os.path.join(tmp, "shot.wav")
            shot.sub_maker = object()
            output = os.path.join(tmp, "subtitle.srt")

            def fake_whisper(**kwargs):
                self.assertEqual(kwargs["audio_file"], shot.audio_path)
                with open(kwargs["subtitle_file"], "w", encoding="utf-8") as handle:
                    handle.write("1\n00:00:00,000 --> 00:00:00,200\n第一句\n")

            with patch("app.services.subtitle.create", side_effect=fake_whisper), patch(
                "app.services.subtitle.correct", return_value=None
            ):
                write_storyboard_subtitles(
                    timeline,
                    subtitle_path=output,
                    subtitle_enabled=True,
                    subtitle_provider="whisper",
                    word_level=False,
                )
            with patch("app.services.storyboard_timing.voice.create_subtitle"), patch(
                "app.services.subtitle.create"
            ) as whisper:
                with self.assertRaises(storyboard_timing.StoryboardTimingError):
                    write_storyboard_subtitles(
                        timeline,
                        subtitle_path=output,
                        subtitle_enabled=True,
                        subtitle_provider="edge",
                        word_level=False,
                    )
            whisper.assert_not_called()

    def test_joined_pcm_uses_sample_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a.wav")
            second = os.path.join(tmp, "b.wav")
            dest = os.path.join(tmp, "audio.wav")
            _pcm(first, 100)
            _pcm(second, 50)
            self.assertEqual(concat_pcm_wavs([first, second], dest), 150)
            with wave.open(dest, "rb") as handle:
                self.assertEqual(handle.getnframes(), 150)
                self.assertEqual(handle.getframerate(), PCM_SAMPLE_RATE)

    def test_clip_shorter_than_one_frame_is_reported(self):
        shots = (StoryboardShot(1, "Wide: 画面", "", 0.0, FRAME_SECONDS / 2),)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(storyboard_timing.StoryboardTimingError) as caught:
                synthesize_timeline(
                    task_dir=tmp,
                    shots=shots,
                    voice_name="zh-CN-XiaoxiaoNeural-Female",
                    voice_rate=1.0,
                )
        self.assertIn("one frame", str(caught.exception))
        self.assertEqual(caught.exception.stage, "audio")


class StoryboardMaterialTests(unittest.TestCase):
    def _plan(self, index: int, request: int, required: float, playback: float):
        shot = _shot(index, playback)
        shot.request_seconds = request
        shot.required_source_seconds = required
        shot.prompt = f"画面 {index}"
        return shot

    def _item(self, remote_id: str) -> MaterialInfo:
        item = MaterialInfo()
        item.provider = "wan3"
        item.url = f"https://example.test/{remote_id}.mp4"
        item.duration = 5
        item.source_info = {"provider": "wan3", "asset_id": remote_id, "search_term": "画面"}
        return item

    def test_short_unknown_and_download_failures_do_not_submit_another_shot(self):
        plans = [
            self._plan(1, 8, 7.2, 7.2),
            self._plan(2, 3, 3.0, 3.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("app.services.material.wan3.generate_videos", side_effect=[
                [self._item("remote-1")],
                AssertionError("second paid request"),
            ]) as generate, patch(
                "app.services.material._save_generated_video_with_retry", return_value=os.path.join(tmp, "a.mp4")
            ), patch("app.services.material._get_downloaded_video_duration", return_value=5.0), patch(
                "app.services.material._persist_material_sources"
            ):
                with self.assertRaises(Exception) as caught:
                    material._download_storyboard_wan3(
                        task_id="task",
                        shot_plan=plans,
                        video_aspect=VideoAspect.portrait,
                        material_directory=tmp,
                    )
            self.assertIn("too short", str(caught.exception))
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(plans[0].remote_task_id, "remote-1")

            with patch(
                "app.services.material.wan3.generate_videos",
                side_effect=Wan3UnconfirmedTaskError("unknown", task_id="remote-unknown"),
            ) as generate, patch("app.services.material._persist_material_sources"):
                with self.assertRaises(Wan3UnconfirmedTaskError):
                    material._download_storyboard_wan3(
                        task_id="task",
                        shot_plan=plans,
                        video_aspect=VideoAspect.portrait,
                        material_directory=tmp,
                    )
            self.assertEqual(generate.call_count, 1)

            with patch(
                "app.services.material.wan3.generate_videos", return_value=[self._item("remote-dl")]
            ) as generate, patch(
                "app.services.material._save_generated_video_with_retry", return_value=""
            ), patch("app.services.material._persist_material_sources"):
                with self.assertRaises(Wan3DownloadError):
                    material._download_storyboard_wan3(
                        task_id="task",
                        shot_plan=plans,
                        video_aspect=VideoAspect.portrait,
                        material_directory=tmp,
                    )
            self.assertEqual(generate.call_count, 1)

    def test_over_limit_plan_is_rejected_before_the_downloader_runs(self):
        timeline = _timeline(_shot(1, 31.0))
        with patch("app.services.material.wan3.generate_videos") as generate:
            with self.assertRaises(storyboard_timing.StoryboardTimingError):
                build_generation_plan(
                    timeline, clip_speed=1.0, explicit_clip_duration=None,
                    **material.storyboard_generation_options("wan3"),
                )
            generate.assert_not_called()

    def test_video_count_reuses_one_storyboard_plan(self):
        plans = [self._plan(1, 3, 3.0, 3.0), self._plan(2, 2, 1.2, 1.2)]
        params = VideoParams(
            video_subject="示例",
            video_script="你好",
            video_terms=["画面 1", "画面 2"],
            video_source="wan3",
            video_count=3,
            subtitle_enabled=False,
        )
        with patch("app.services.material.download_videos", return_value=["a.mp4", "b.mp4"]) as download:
            tm.get_video_materials("task", params, params.video_terms, 4.2, shot_plan=plans)
        self.assertEqual(download.call_count, 1)
        self.assertIs(download.call_args.kwargs["shot_plan"], plans)
        self.assertEqual(download.call_args.kwargs["audio_duration"], 4.2)

    def test_existing_wan3_and_topic_paths_ignore_shot_plans(self):
        with patch("app.services.material._download_videos_wan3_on_demand", return_value=["old.mp4"]) as old, patch(
            "app.services.material._download_storyboard_wan3"
        ) as storyboard_download:
            result = material.download_videos(
                task_id="task",
                search_terms=["城市"],
                source="wan3",
                audio_duration=8,
                max_clip_duration=5,
            )
        self.assertEqual(result, ["old.mp4"])
        old.assert_called_once()
        storyboard_download.assert_not_called()

    def test_wan3_callers_preserve_uniform_and_per_shot_request_lengths(self):
        for plans, expected in (
            (None, [5, 5]),
            ([self._plan(1, 6, 6.0, 6.0), self._plan(2, 3, 3.0, 3.0)], [6, 3]),
        ):
            with self.subTest(per_shot=plans is not None), tempfile.TemporaryDirectory() as tmp:
                with patch.object(material.wan3, "generate_videos", side_effect=[
                    [self._item("remote-1")], [self._item("remote-2")],
                ]) as generate, patch.object(
                    material, "_save_generated_video_with_retry", side_effect=["a.mp4", "b.mp4"]
                ), patch.object(material, "_get_downloaded_video_duration", side_effect=[6.0, 3.0]), patch.object(
                    material, "_persist_material_sources"
                ):
                    result = material.download_videos(
                        task_id="task", search_terms=["画面 1", "画面 2"], source="wan3",
                        audio_duration=8, max_clip_duration=5, shot_plan=plans,
                    )
                self.assertEqual(result, ["a.mp4", "b.mp4"])
                self.assertEqual([call.kwargs["minimum_duration"] for call in generate.call_args_list], expected)

    def test_external_audio_and_full_preview_fail_before_generation(self):
        params = VideoParams(
            video_subject="示例",
            video_script="你好",
            video_source="wan3",
            custom_audio_file="narration.wav",
        )
        with self.assertRaises(storyboard_timing.StoryboardTimingError) as caught:
            tm._reject_storyboard_external_audio(params, None)
        self.assertIn("per-shot audio", str(caught.exception))
        preview_params = VideoParams(video_subject="示例", video_script="你好", video_source="wan3")
        with self.assertRaises(storyboard_timing.StoryboardTimingError) as preview_caught:
            tm._reject_storyboard_external_audio(preview_params, {"audio_file": "full.mp3", "duration": 8})
        self.assertIn("voice preview", str(preview_caught.exception))

    def test_task_start_appends_storyboard_arguments(self):
        parameters = list(inspect.signature(tm.start).parameters)
        self.assertEqual(parameters[-2:], ["storyboard_data", "storyboard_clip_duration"])


class StoryboardPipelineTests(unittest.TestCase):
    def _params(self, **overrides) -> VideoParams:
        values = {
            "video_subject": "示例",
            "video_script": "第一句。第二句。",
            "video_terms": ["Wide: 甲", "Close: 乙"],
            "video_source": "wan3",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
            "voice_rate": 1.0,
            "subtitle_enabled": False,
            "bgm_volume": 0,
            "video_clip_duration": 5,
        }
        values.update(overrides)
        return VideoParams(**values)

    def _run(self, params, storyboard, *, clip_duration, tts, stop_at="video"):
        state = {}

        def update_task(_task_id, **kwargs):
            state.update(kwargs)
            return state

        def get_task(_task_id):
            return dict(state)

        with tempfile.TemporaryDirectory() as tmp, patch.object(tm.sm.state, "update_task", side_effect=update_task), patch.object(
            tm.sm.state, "get_task", side_effect=get_task
        ), patch.object(tm.sm.state, "patch_task", return_value=True), patch.object(
            tm.utils, "check_ffmpeg_ready", return_value=True
        ), patch.object(tm.wan3, "validate_config"), patch.object(
            tm.utils, "task_dir", return_value=tmp
        ), patch.object(tm.voice, "tts", side_effect=tts), patch.object(
            tm.wan3, "generate_videos", side_effect=AssertionError("wan3 was submitted")
        ) as generate:
            result = tm._run_pipeline(
                "storyboard-task",
                params,
                stop_at=stop_at,
                storyboard_data=storyboard,
                storyboard_clip_duration=clip_duration,
            )
        return result, generate, state

    def test_unadapted_source_does_not_fall_back_or_start_tts(self):
        storyboard = Storyboard(
            "第一句。", ["Wide: 甲"],
            shots=(StoryboardShot(1, "Wide: 甲", "第一句。", 0.0, 5.0),),
        )
        result, generate, _state = self._run(
            self._params(video_source="pexels"), storyboard,
            clip_duration=None, tts=AssertionError("tts started"),
        )
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("per-shot storyboard material adapter", result["error"])
        generate.assert_not_called()

    def test_replacement_adapter_uses_the_same_storyboard_pipeline(self):
        storyboard = Storyboard(
            "第一句。", ["Wide: 甲"],
            shots=(StoryboardShot(1, "Wide: 甲", "第一句。", 0.0, 5.0),),
        )

        def tts(**kwargs):
            _pcm(kwargs["voice_file"], 31 * PCM_SAMPLE_RATE)
            return SimpleNamespace(cues=())

        from unittest.mock import Mock
        download = Mock(return_value=["replacement.mp4"])
        options = {"minimum_clip_seconds": 1, "maximum_clip_seconds": 60, "provider_name": "Replacement"}
        with patch.object(material, "_storyboard_adapter", return_value=(download, options)):
            result, generate, _state = self._run(
                self._params(video_source="pexels", video_script="第一句。"), storyboard,
                clip_duration=None, tts=tts, stop_at="materials",
            )
        self.assertEqual(result, {"materials": ["replacement.mp4"]})
        self.assertEqual(download.call_args.kwargs["shot_plan"][0].request_seconds, 31)
        generate.assert_not_called()

    def test_empty_storyboard_does_not_use_whole_narration(self):
        result, generate, _state = self._run(
            self._params(), Storyboard("旁白", ["画面"]),
            clip_duration=None, tts=AssertionError("tts started"),
        )
        self.assertEqual(result["failed_stage"], "preflight")
        self.assertIn("no timed shots", result["error"])
        generate.assert_not_called()

    def test_tts_failure_submits_no_wan3_task(self):
        storyboard = Storyboard(
            "第一句。",
            ["Wide: 甲"],
            shots=(StoryboardShot(1, "Wide: 甲", "第一句。", 0.0, 5.0),),
        )
        result, generate, _state = self._run(
            self._params(video_script="第一句。", video_terms=["Wide: 甲"]),
            storyboard,
            clip_duration=None,
            tts=lambda **_kwargs: None,
        )
        self.assertEqual(result["failed_stage"], "audio")
        generate.assert_not_called()

    def test_over_limit_after_tts_submits_no_wan3_task(self):
        storyboard = Storyboard(
            "很长的一句。",
            ["Wide: 甲"],
            shots=(StoryboardShot(1, "Wide: 甲", "很长的一句。", 0.0, 5.0),),
        )

        def tts(**kwargs):
            _pcm(kwargs["voice_file"], int(round(31 * PCM_SAMPLE_RATE)))
            return SimpleNamespace(cues=())

        result, generate, _state = self._run(
            self._params(video_script="很长的一句。", video_terms=["Wide: 甲"]),
            storyboard,
            clip_duration=None,
            tts=tts,
        )
        self.assertEqual(result["failed_stage"], "materials")
        self.assertIn("本次尚未提交 Wan3 视频生成", result["error"])
        self.assertNotIn("没有任何费用", result["error"])
        generate.assert_not_called()

    def test_custom_audio_fails_before_tts_and_wan3(self):
        storyboard = Storyboard(
            "第一句。",
            ["Wide: 甲"],
            shots=(StoryboardShot(1, "Wide: 甲", "第一句。", 0.0, 5.0),),
        )
        result, generate, _state = self._run(
            self._params(video_script="第一句。", video_terms=["Wide: 甲"], custom_audio_file="voice.wav"),
            storyboard,
            clip_duration=None,
            tts=AssertionError("tts started"),
        )
        self.assertEqual(result["failed_stage"], "audio")
        self.assertIn("per-shot audio", result["error"])
        generate.assert_not_called()

    def test_missing_subtitles_report_their_stage_before_wan3(self):
        storyboard = Storyboard(
            "第一句。", ["Wide: 甲"],
            shots=(StoryboardShot(1, "Wide: 甲", "第一句。", 0.0, 5.0),),
        )

        def tts(**kwargs):
            _pcm(kwargs["voice_file"], PCM_SAMPLE_RATE)
            return SimpleNamespace(cues=())

        with patch.dict(tm.config.app, {"subtitle_provider": "edge"}), patch.object(tm.voice, "create_subtitle"):
            result, generate, _state = self._run(
                self._params(video_script="第一句。", subtitle_enabled=True),
                storyboard, clip_duration=None, tts=tts,
            )
        self.assertEqual(result["failed_stage"], "subtitle")
        self.assertIn("shot 1", result["error"])
        generate.assert_not_called()


class StoryboardRenderTests(unittest.TestCase):
    def test_local_color_timeline_extends_shortens_and_holds_silence(self):
        ffmpeg = utils.get_ffmpeg_binary()
        # Seven frames is a fractional duration that MoviePy may round down.
        durations = {"第一句。": 0.5, "第二句。": 7 / 30}

        def fake_tts(**kwargs):
            _pcm(kwargs["voice_file"], int(round(durations[kwargs["text"]] * PCM_SAMPLE_RATE)))
            return SimpleNamespace(cues=())

        shots = (
            StoryboardShot(1, "Wide: 红", "第一句。", 0.0, 0.2),
            StoryboardShot(2, "Close: 绿", "第二句。", 0.2, 0.7),
            StoryboardShot(3, "End: 蓝", "", 0.7, 1.1),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "app.services.storyboard_timing.voice.tts", side_effect=fake_tts
        ), patch.object(video.VideoAspect, "to_resolution", return_value=(64, 64)):
            timeline = synthesize_timeline(
                task_dir=tmp,
                shots=shots,
                voice_name="zh-CN-XiaoxiaoNeural-Female",
                voice_rate=1.0,
            )
            build_generation_plan(
                timeline, clip_speed=1.0, explicit_clip_duration=None,
                **material.storyboard_generation_options("wan3"),
            )
            colors = {1: "red", 2: "green", 3: "blue"}
            sources = []
            for shot in timeline.shots:
                path = os.path.join(tmp, f"color-{shot.index}.mp4")
                subprocess.run(
                    [
                        ffmpeg, "-y", "-f", "lavfi",
                        "-i", f"color=c={colors[shot.index]}:s=64x64:d=2:r=30",
                        "-pix_fmt", "yuv420p", path,
                    ],
                    check=True,
                    capture_output=True,
                )
                sources.append(path)
            combined = os.path.join(tmp, "combined.mp4")
            video.combine_videos(
                combined_video_path=combined,
                video_paths=sources,
                audio_file=timeline.audio_path,
                video_aspect=VideoAspect.square,
                max_clip_duration=5,
                threads=1,
                storyboard_shot_plan=timeline.shots,
            )
            finished = os.path.join(tmp, "finished.mp4")
            video.generate_video(
                video_path=combined,
                audio_path=timeline.audio_path,
                subtitle_path="",
                output_file=finished,
                params=VideoParams(
                    video_subject="local render", video_aspect=VideoAspect.square,
                    subtitle_enabled=False, bgm_volume=0, n_threads=1,
                ),
            )
            with wave.open(timeline.audio_path, "rb") as handle:
                audio_seconds = handle.getnframes() / handle.getframerate()
            container_seconds = _media_duration(ffmpeg, finished)
            self.assertLessEqual(abs(container_seconds - audio_seconds), 2 * FRAME_SECONDS)
            # Planned cuts were 0.2s and 0.7s. Measured cuts are 0.5s and 22/30s.
            # Samples stay more than one frame away from those measured cuts.
            self.assertEqual(_dominant_color(ffmpeg, finished, 0.35), "red")
            self.assertEqual(_dominant_color(ffmpeg, finished, 0.45), "red")
            self.assertEqual(_dominant_color(ffmpeg, finished, 0.55), "green")
            self.assertEqual(_dominant_color(ffmpeg, finished, 0.62), "green")
            self.assertEqual(_dominant_color(ffmpeg, finished, 0.699), "green")
            self.assertEqual(_dominant_color(ffmpeg, finished, 0.78), "blue")
            # Stay inside an encoded frame; final render may differ by two frames.
            self.assertEqual(_dominant_color(ffmpeg, finished, max(audio_seconds - 3 * FRAME_SECONDS, 0.78)), "blue")
            self.assertEqual(timeline.shots[0].request_seconds, 2)
            self.assertEqual(timeline.shots[1].request_seconds, 2)
            self.assertAlmostEqual(timeline.shots[0].target_playback_seconds, 0.5)
            self.assertAlmostEqual(timeline.shots[1].target_playback_seconds, 7 / 30)


def _media_duration(ffmpeg: str, path: str) -> float:
    result = subprocess.run(
        [ffmpeg, "-i", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    matched = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not matched:
        raise AssertionError(result.stderr[-500:])
    hours, minutes, seconds = matched.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _dominant_color(ffmpeg: str, path: str, timestamp: float) -> str:
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", path, "-ss", f"{timestamp:.3f}",
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    pixels = result.stdout
    count = len(pixels) // 3
    if count <= 0:
        return "empty"
    red = sum(pixels[index] for index in range(0, len(pixels), 3)) / count
    green = sum(pixels[index] for index in range(1, len(pixels), 3)) / count
    blue = sum(pixels[index] for index in range(2, len(pixels), 3)) / count
    if max(red, green, blue) < 20:
        return "black"
    if red > green + 30 and red > blue + 30:
        return "red"
    if green > red + 30 and green > blue + 30:
        return "green"
    if blue > red + 30 and blue > green + 30:
        return "blue"
    return f"{red:.0f},{green:.0f},{blue:.0f}"


class StoryboardCliTests(unittest.TestCase):
    def test_both_clip_duration_spellings_are_explicit_and_omission_is_automatic(self):
        import cli

        text = _board("| 0:00-0:05 | Wide | 房间 | 你好 | bed |", "你好")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_board(tmp, text)
            spaced = cli.parse_args([
                "--video-subject", "示例", "--storyboard", path, "--video-clip-duration", "5",
            ])
            equals = cli.parse_args([
                "--video-subject", "示例", "--storyboard", path, "--video-clip-duration=5",
            ])
            automatic = cli.parse_args(["--video-subject", "示例", "--storyboard", path])
            replacement = cli.parse_args([
                "--video-subject", "示例",
                "--video-source", "pexels",
                "--storyboard", path,
            ])
            with self.assertRaises(SystemExit):
                cli.parse_args([
                    "--video-source", "pexels",
                    "--storyboard", _write_board(tmp, _board("| later | Wide | 房间 | 你好 | bed |", "你好"), "invalid.md"),
                ])
        self.assertEqual(spaced.video_clip_duration, 5)
        self.assertEqual(equals.video_clip_duration, 5)
        self.assertIsNone(automatic.video_clip_duration)
        self.assertEqual(len(automatic.storyboard_data.shots), 1)
        self.assertEqual(replacement.storyboard_data.shots, automatic.storyboard_data.shots)
        self.assertEqual(replacement.video_source, "pexels")


if __name__ == "__main__":
    unittest.main()
