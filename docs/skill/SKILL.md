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

视频供应商不默认 Wan3。helper 只根据环境变量选择 Wan3 或 Seedance 兼容网关：只有一家有 Key 时自动使用，没有 Key 或有多家时询问，不读取 `config.toml` 里的旧视频凭据。

Other defaults are one Chinese portrait (`9:16`) video, Chinese Edge TTS, subtitles, and background music. Preserve explicit user choices.

## Execute

Resolve the actual directory of this `SKILL.md`; the helper is adjacent. Set the terminal working directory to that directory (`workdir=SKILL_DIR`) and use the relative filename:

```text
uv run --no-project --python 3.11 python mpt_agent.py --subject "<topic>" -- --storyboard "<absolute storyboard.md path>"
```

On a machine without MoneyPrinterTurbo, the helper downloads the internal fork archive (`https://github.com/Phenomenax/MoneyPrinterTurbo-Workie/archive/refs/heads/main.zip`) into `~/MoneyPrinterTurbo`. Pass `--root` only to select a different installation directory. An existing valid project is reused and is not overwritten.

Use the storyboard skill's fixed Markdown table and clean narration section. Every storyboard uses the same per-shot workflow: read its times and lines, voice each spoken shot once, then lengthen or shorten that shot's picture to the measured narration. Silent shots keep their planned duration. The finished video can be shorter or longer than the planned total. Do not rewrite lines, change the speaking rate, or divide one narration evenly across shots to restore the plan. The selected material adapter supplies generation limits and downloads. Wan3 accepts 2–30 integer seconds; a Seedance-compatible gateway uses the limits of the configured model family. An unadapted source stops before TTS instead of using the whole-narration path. An explicit `--video-clip-duration` still applies to source requests; any shot it cannot cover, or that exceeds the adapter's limit, stops before video generation. Separate shot readings can sound less continuous than one full read. Generic topic-only and explicit CLI inputs remain available when requested; forward other options after `--`.

Run one foreground job with a timeout of at least 20 minutes. Resume the same terminal session if it yields. Do not start another job while waiting. The helper handles uv dependencies and final video validation; do not add a separate runner or start Docker, WebUI, or API services.

## Configuration

Do not add `--video-source` until the user has chosen a provider. Video settings come only from the helper process environment and are not read from or written to `config.toml`:

- Wan3 requires `MPT_WAN3_API_KEY` (or `DASHSCOPE_API_KEY`), `MPT_WAN3_BASE_URL`, and `MPT_WAN3_MODEL_NAME`. The address is the native video API, not `/compatible-mode/v1`. There is no default model name.
- The Seedance-compatible gateway requires `MPT_SEEDANCE_API_KEY` (or `MPT_VOLCENGINE_ARK_API_KEY`, then `VOLCENGINE_ARK_API_KEY`), `MPT_SEEDANCE_BASE_URL`, and `MPT_SEEDANCE_MODEL_NAME`. Describe it as a Seedance-compatible gateway, not as Volcano Engine's official service. Send the platform model name unchanged. If the name does not identify a known family, also set `MPT_SEEDANCE_MODEL_FAMILY`.

Zero keys, multiple keys, or a missing address, model, or family returns exit 10 with `INPUT_REASON`, `VIDEO_SOURCE_OPTION`, , `MISSING`, and `INVALID`. Ask only for those names and rerun the same storyboard after the variables are visible to the helper process. Never print credentials. A complete storyboard requires no text-model key. Inputs that need generated narration or visual prompts use MPT's own text-model configuration (`MPT_LLM_PROVIDER`, `MPT_LLM_API_KEY`, and optionally `MPT_LLM_BASE_URL` / `MPT_LLM_MODEL_NAME`).

An older installation that lacks this environment mode must be upgraded or selected with `--root`. Do not overwrite it and do not silently fall back to Wan3. The public project archive does not contain this working tree until the revision is published.

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

For explicit Infistar video generation, use `MPT_INFISTAR_VIDEO_API_KEY` (separate from the Infistar LLM key). When `INFISTAR_CHARGE_CONFIRMATION_REQUIRED` is reported, explain the per-task charge and add `--confirm-infistar-charge` only after the user confirms.

For explicit Jalapeno Cloud video generation, use `MPT_JALAPENO_VIDEO_API_KEY`; an empty video key reuses the configured Jalapeno Cloud LLM key. When `JALAPENO_CHARGE_CONFIRMATION_REQUIRED` is reported, explain the per-task charge and add `--confirm-jalapeno-charge` only after the user confirms.
