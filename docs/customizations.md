# 内部定制清单

这里只记录同步 upstream（MoneyPrinterTurbo）时可能发生冲突、被覆盖或需要复核的定制点：修改了哪些上游文件，以及合并时必须保留的差异。

内部新增模块、测试清单和功能实现细节不在这里逐项登记；内部扩展只登记其对上游文件的接入改动。只有上游原文件的实际差异或同步注意事项发生变化时才更新本清单。Wan3 并发实现与运行边界见 [wan3-concurrency.md](wan3-concurrency.md)。

清单核对基线：2026-11-03，官方提交 [`ab6004e5`](https://github.com/harry0703/MoneyPrinterTurbo/commit/ab6004e5858998b2ce99ac46c0f16675eea9ab6f)。后续同步须重新核对实际差异。

| 文件或目录 | 类型 | 原因 | 维护说明 |
|---|---|---|---|
| `app/models/schema.py`、`config.example.toml` | 默认素材源与配置 | 默认使用 Wan3，并提供其接入、时长及并发配置 | 保留 `video_source = "wan3"` 和 Wan3 配置字段；接纳官方新增字段及枚举默认值修复；不要覆盖用户的 `config.toml` |
| `cli.py` | Workie 执行入口 | 接入分镜、进度、环境选源、检查点及显式原生音频分发 | 保留 `--storyboard-audio-mode tts|native`，默认仍为 `tts`。native 只支持环境选源的完整分镜单任务，使用原生解析并在收费前拒绝替代声音、本地字幕、固定片段时长和变速参数；保存的旧配音、配乐和字体设置不进入资源校验。保留原分镜、进度与检查点参数；原生分镜 Resolution 优先于供应商环境变量，仅作用于模型请求。宽高比仍为显式 CLI > 分镜 Aspect > `9:16`。环境配置不与 CLI 原生 batch 组合；恢复沿用原 task ID，重建必须明确标记；镜头修改仅凭冻结方案及其 SHA-256 启动新的原声检查点任务 |
| `app/services/llm.py` | Wan3 画面提示词 | 为视频生成提供完整场景描述 | 保留 `generate_video_prompts` 接口及其顺序、数量要求；与官方库存搜索关键词生成分开维护 |
| `app/services/material.py` | Wan3 与分镜素材接入 | 接入素材适配器、共享有界调度、远端任务记录、Seedance 分镜及镜头检查点 | 保留有序返回、共享调度与环境快照。native 下载后的时长探测只读视频，先保存素材检查点，再由独立原生路径检查音轨；不得把缺音轨转成可重新收费的生成失败，也不按配音时长拒绝较短原生素材。旧 TTS 校验不变。恢复先核对全部原提交，复用成功镜头；未知且无编号的提交不得重提 |
| `app/services/task.py` | 流水线编排 | 接入分镜计时、供应商任务 ID、阶段进度、检查点和原生模式分发 | `tts` 保留逐镜头配音及校准时间线，旧检查点身份不变。仅显式 native 分发到独立模块，跳过 TTS、识别、字幕叠加和额外配乐；使用素材、合成、完成进度及现有 render 检查点。失败保留真实错误、镜头及远端任务 ID。官方生成上下文、分镜参数与 `video_run_config` 须一起传递，不把快照密钥写入状态 |
| `app/services/video.py` | 分镜合成 | 保留旧分镜计时并提供原生音轨拼接接入 | 旧 `storyboard_shot_plan`、时长校验与直接拼接优化不变。公共 FFmpeg 拼接新增默认关闭的 `preserve_audio`；native 显式映射音视频，不能继承 stream-copy 的 `-an`，PCM 中间段拼接后只编码一次 AAC。复制失败可回退编码，超时不重试；其他调用沿用原行为 |
| `webui/Main.py`、`webui/settings_providers.py` | Wan3 界面接入 | 提供素材源、凭据、端点、模型和时长选择 | 保留 Wan3 默认源、2～30 秒选项和生成前配置校验；同步官方设置页重构时，迁移到新的标签、表单及就绪状态入口，保留切换供应商前保存最终输入的回调 |
| `docs/skill/mpt_agent.py` | Agent 执行协议 | 接入环境选源、分镜路径处理及机器可读反馈 | 标准调用不注入 Wan3；视频 Key、地址和模型只来自环境变量，完整分镜路径不写回 `config.toml`。保留旧配置补字段而不覆盖已有值；预检不得误索取完整分镜不需要的 LLM Key；能力检查包含配置模块 API 与 `--video-config-from-env`，不能只看版本号。保留任务 ID 匹配、当前运行结果校验、`MPT_FEEDBACK` 和脱敏；日志走 stderr；保留官方新增来源、凭据及计费确认逻辑 |
| `docs/skill/SKILL.md` | Workie 调用约定 | 按环境变量选择视频供应商，按分镜执行并交付本次结果 | 保留相对 helper 命令、真实工作目录、长任务等待、逐镜头流程及失败恢复边界；不把 Wan3 写回默认来源，同步供应商说明时避免重复配置询问或盲目重提付费任务 |
| `.gitignore` | 运行记录隔离 | 忽略 Wan3 与 Seedance 跨进程协调记录 | 保留 `/.agent-logs/wan3/` 和 `/.agent-logs/seedance-*/`；协调文件属于运行状态，不得提交到 Git，也不能在同步代码时顺手清理 |
| `app/services/volcengine_seedance.py` | Seedance 分镜接入 | 接入任务配置快照、共享名额协调及原任务查询恢复 | 保留快照配置在提交、轮询及下载中的传递；恢复原 ID 只查询和取回结果，不执行新提交。保留独立协调命名空间，提交或远端状态不确定时不得释放名额或自动重提；兼容官方非分镜调用 |
| `.github/workflows/ci.yml`、`.github/workflows/docker-ghcr.yml`、`.github/workflows/security.yml` | 官方 CI/CD 删除 | 内部分支已移除这三份官方 workflow | 同步时检查上游修改或重新引入的 workflow，避免无意恢复已删除的自动执行流程 |

## 同步 upstream 时必须保证

- 在独立同步分支上合并。`origin` 指向 `Phenomenax/MoneyPrinterTurbo-Workie`，`upstream` 指向 `harry0703/MoneyPrinterTurbo`。先更新自己的 `main`，再执行 `git fetch upstream` 和 `git merge --no-ff --no-commit upstream/main`；逐项检查上述接入点。通过 PR 使用 merge commit 合回 `main`，保留官方提交的祖先关系；不要整文件覆盖、全局选 ours/theirs 或强推共享分支。
- 同时保留两边能力。Wan3 默认值不影响显式选择其他来源；官方新增供应商、OAuth、字幕、安全修复及生成活动反馈应继续可用。检查支持来源列表、参数定义、调用端和配置表单是否一致，不能只删掉冲突标记。
- 保持分镜和反馈协议。默认 `tts` 每镜头配音一次，以实测 PCM 时长驱动画面，静音镜头保留计划时长；不改写台词、变速或均分旁白。显式 native 根据计划整数秒请求并保留返回的完整音画，首尾整轨各淡化 0.2 秒，直接有序拼接；缺音轨只复查原素材，不自动收费重提。两种模式的 `stage`、`percent`、`clips_completed`、`clips_total` 及失败阶段仍可读取；旧 TTS 指纹构成和默认行为必须保持兼容。
- 保持付费任务边界。Wan3 共享名额仍按同一 runtime root 限制为最多 5 个远端任务，并保留提交节流；失败后停止补交并保存已提交任务的信息。网络失败、本地超时、进程退出均不能视为远端已结束，不得自动重提或删除不确定记录。升级前确认旧进程及其远端任务已结束，不能让运行中的任务切换协调实现。
- 保护用户配置和运行数据。不要覆盖 `config.toml`、存储目录或登录凭据；不要输出密钥、完整配置或带签名的下载地址。自动测试使用临时 `MPT_HOME`、示例配置和模拟凭据，不带入真实供应商凭据，不以真实付费生成代替回归测试。

## 合并验证

在 MPT 仓库根目录执行。先按锁文件同步依赖，再进行语法、lint 和测试检查：

```powershell
uv sync --frozen
uv run --no-sync python -X utf8 -m compileall -q app cli.py main.py webui test docs/skill
uv run --no-sync ruff check app cli.py main.py webui test docs/skill
uv run --no-sync python -X utf8 -m pytest -q test
git diff --check
git diff --cached --check
git diff --name-only --diff-filter=U
```

最后一条应无输出。重点复核分镜宽高比与时长、Wan3 并发和远端任务保留、CLI/helper 成败反馈、进度导出、设置页切换保存及官方新增供应商。测试若仍假设默认来源是 Pexels，应明确指定来源；日志断言需区分 stdout 协议和 stderr 日志。

报告实际执行命令及结果，区分模拟测试、环境相关失败和真实生成验证；不得省略失败，或把跳过的测试当作通过。新增定制时只补充上游接入点与同步注意事项，同一文件优先合并到已有条目。
