"""No external APIs: verify the task-result contract at the CLI/helper boundary."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import cli

PATH = Path(__file__).parents[2] / "docs/skill/mpt_agent.py"
SPEC = importlib.util.spec_from_file_location("feedback_helper", PATH)
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)


@pytest.mark.parametrize("result", [None, {"state": -1, "failed_stage": "subtitles", "error": "caption failed"}])
def test_cli_failure_is_machine_readable(result, capsys):
    with patch("app.services.task.start", return_value=result):
        assert cli.run_cli(["--video-subject", "coffee", "--task-id", "12345678-1234-1234-1234-123456789abc"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_id"] == "12345678-1234-1234-1234-123456789abc"
    assert payload["result"]["state"] == -1
    assert payload["result"]["error"] == (result["error"] if result else "empty result")


def test_cli_exception_is_machine_readable(capsys):
    with patch("app.services.task.start", side_effect=RuntimeError("unexpected")):
        assert cli.run_cli(["--video-subject", "coffee", "--task-id", "12345678-1234-1234-1234-123456789abc"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["failed_stage"] == "runtime"
    assert payload["result"]["error"] == "unexpected"


@pytest.mark.parametrize("matching", [True, False])
def test_helper_uses_only_submitted_task(tmp_path, matching):
    output = json.dumps({"task_id": "12345678-1234-1234-1234-123456789abc" if matching else "other-task", "result": {
        "state": -1, "failed_stage": "subtitles", "error": "caption failed",
    }})
    with patch.object(agent.shutil, "which", return_value="uv"), patch.object(agent, "run_checked"), \
         patch.object(agent.uuid, "uuid4", return_value="12345678-1234-1234-1234-123456789abc"), \
         patch.object(agent.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=output)):
        with pytest.raises(agent.SkillError) as error:
            agent.generate_video(tmp_path, "coffee", [])
    if matching:
        assert error.value.stage == "subtitles"
        assert str(error.value) == "caption failed"
    else:
        assert error.value.stage == "generation"
        assert str(error.value).startswith("video generation failed with exit code 1")


def test_old_files_and_shared_latest_cannot_prove_success(tmp_path):
    task = tmp_path / "storage/tasks/old"
    task.mkdir(parents=True)
    (task / "final-1.mp4").write_bytes(b"old")
    agent.write_result_manifest(tmp_path, {"task_id": "old", "status": "completed"})
    with patch.object(agent.shutil, "which", return_value="uv"), patch.object(agent, "run_checked"), \
         patch.object(agent.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="")):
        with pytest.raises(agent.SkillError) as error:
            agent.generate_video(tmp_path, "coffee", [])
    assert "without a valid final MP4" in str(error.value)


def test_feedback_is_bounded_and_redacts_credentials(capsys):
    agent.emit_feedback("failed", 'api_key="super-secret" Bearer abcdef https://host/?sig=secret',
                        stage="materials")
    line = capsys.readouterr().out.strip()
    payload = json.loads(line.removeprefix("MPT_FEEDBACK="))
    assert payload["execution"]["stage"] == "materials"
    assert "super-secret" not in line and "abcdef" not in line and "sig=secret" not in line
    assert payload["error"]["safe_to_resubmit"] is False
    assert len(agent.safe_feedback_text("x" * 1100)) == 1000

