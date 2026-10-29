---
name: moneyprinterturbo-video
description: Create a finished MP4 from a storyboard, topic, or script with MoneyPrinterTurbo. Use the bundled helper for configuration, execution, waiting, and result delivery.
compatibility: Requires terminal, filesystem, network, uv, and long-running command support on Windows or macOS.
metadata:
  author: "harry0703@hotmail.com"
  version: "1.3.2"
  upstream: "https://github.com/harry0703/MoneyPrinterTurbo"
---

# MoneyPrinterTurbo Video Generation

默认使用 Wan3 生成视频素材。

Other defaults are one Chinese portrait (`9:16`) video, Chinese Edge TTS, subtitles, and background music. Preserve explicit user choices.

## Execute

Resolve the actual directory of this `SKILL.md`; the helper is adjacent. Set the terminal working directory to that directory (`workdir=SKILL_DIR`) and use the relative filename:

```text
uv run --no-project --python 3.11 python mpt_agent.py --subject "<topic>" -- --storyboard "<absolute storyboard.md path>"
```

On a machine without MoneyPrinterTurbo, the helper downloads the public project archive into `~/MoneyPrinterTurbo`. Pass `--root` only to select a different installation directory. An existing valid project is reused and is not overwritten.

Use the storyboard skill's fixed Markdown table and clean narration section. MPT extracts ordered `video_terms` and `video_script` from that file without rewriting them through a text model. The table's times guide planning; exact shot timing is not executed. Generic topic-only and explicit CLI inputs remain available when requested; forward other options after `--`.

Run one foreground job with a timeout of at least 20 minutes. Resume the same terminal session if it yields. Do not start another job while waiting. The helper handles uv dependencies and final video validation; do not add a separate runner or start Docker, WebUI, or API services.

## Configuration

Set credentials in the process that runs the helper, or reuse `config.toml`:

- `MPT_WAN3_API_KEY`, falling back to `DASHSCOPE_API_KEY`.
- `MPT_WAN3_BASE_URL`: the native regional host, for example `https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com` (optionally ending in `/api/v1`), rather than the chat `/compatible-mode/v1` address.
- `MPT_WAN3_MODEL_NAME`: optional, defaults to `wan3.0-video`.

The helper writes these values into `[app]` and adds missing Wan3 fields to old configurations without replacing existing values. Never print credentials or complete configuration. A complete storyboard requires no text-model key. Inputs that need generated narration or visual prompts use MPT's own text-model configuration (`MPT_LLM_PROVIDER`, `MPT_LLM_API_KEY`, and optionally `MPT_LLM_BASE_URL` / `MPT_LLM_MODEL_NAME`).

## Handle results

Success (exit 0):

```text
MPT_RESULT
VIDEO_FILE=<absolute final MP4 path>
TASK_DIR=<task directory>
LOG_FILE=<log path>
RESULT_FILE=<result manifest path>
```

Deliver only files from this run with the existing chat file tool. If successful output is truncated, read the selected root's `.agent-logs/moneyprinterturbo-video/latest-result.json` once and require `status=completed`. Do not inspect unrelated old logs or videos.

Missing configuration (exit 10): `MPT_NEEDS_INPUT` lists `MISSING` fields. Request only these settings together, keep secrets out of chat, and rerun after configuration is supplied. Other explicit material sources retain their own credentials and charge-confirmation requirements.

Failure (exit 1): report `MPT_ERROR` and the relevant log location. Read only the needed log tail. Repair a recoverable local problem before retrying; a remote task or download failure must not trigger blind resubmission of a paid generation task.
