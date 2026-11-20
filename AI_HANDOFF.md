# MoneyPrinterTurbo AI 交接

更新日期：2026-11-19。本地 `F:\TFZY\MoneyPrinterTurbo` 检出 `main`，已提交基线为 `fafeecbe`。工作树已有未提交的分辨率、成片修订和 Wan3 参数改动。继续前先看 `git status --short --branch` 和 `git diff`，保留其他人的工作。若基线已变化，以当前源码和运行环境为准。

## 项目定位和架构

本仓库是 MoneyPrinterTurbo 的 Workie 定制分支。Python 3.11+ 工具链将主题、脚本或完整分镜转换为视频，处理素材、配音、字幕、音乐和 FFmpeg 合成。Workie 负责对话和批次编排，MPT 负责单个制作任务、供应商调用及检查点。[README](README.md)

- **入口**。Workie 的 helper 通过 CLI 提交完整分镜。项目也提供 FastAPI 和 Streamlit 独立入口。CLI 解析参数和环境快照，然后交给共用的任务服务。[CLI](cli.py) · [API](app/asgi.py) · [WebUI](webui/Main.py) · [任务服务](app/services/task.py)
- **分镜**。解析在 `storyboard.py`，TTS 按实测配音时长驱动镜头，静音镜头保留计划时长。显式 native 路径按分镜计划整数秒请求原生音画，保留音轨并按镜头顺序合成。[解析](app/services/storyboard.py) · [TTS 时间线](app/services/storyboard_timing.py) · [原生链路](app/services/storyboard_native.py) · [FFmpeg 合成](app/services/video.py)
- **素材与恢复**。素材模块分发 Wan3、Seedance 等供应商调用，分镜检查点记录阶段、远端 ID、素材文件哈希和成片。运行目录通过 `MPT_HOME` 决定，源码检出默认使用项目根目录。Workie 的 `MPT_ROOT` 选择 CLI 安装目录。[运行目录](app/paths.py) · [素材分发](app/services/material.py) · [检查点](app/services/storyboard_checkpoint.py) · [镜头恢复](app/services/checkpoint_materials.py)

## 已完成和已验证

- Git 中已有 Wan3 跨进程并发（`5c62f1bf`）、Seedance 分镜源（`41c9603e`）、分镜检查点（`1ff03139`）、native 音画生成和拼接（`d93e6670`）、从逐镜台词派生原生旁白（`fafeecbe`）。
- 本地已实现读取发布分镜的 `Resolution`，用于预检及 Wan3、Seedance 请求。2026-11-18 的分辨率功能定向验证中，MPT 114 项通过，覆盖模拟供应商、旧分镜回退和恢复配置。本地横屏素材验证导出为 1920×1080。该次验证对应分辨率功能，当前修订和 Wan3 改动仍需独立复核。

## 技术决策和排障入口

- **环境快照**。Workie 分镜调用走 `--video-config-from-env`，由 `video_run_config.py` 读取进程里的 Wan3 或 Seedance 配置并冻结本次参数，独立入口仍保留自己的 TOML 配置流程。该模块由 helper 提前按文件载入，保持仅依赖标准库。设置 Seedance 时，`MPT_SEEDANCE_MODEL_NAME` 是模型名或端点 ID，`MPT_SEEDANCE_MODEL_FAMILY` 对应实际能力范围。模型支持的时长、分辨率和原生音轨能力要在提交前校验。[供应商配置](app/services/video_run_config.py) · [示例配置](config.example.toml)
- **生成与导出**。新分镜显式档位用于模型素材请求，旧分镜沿用原有配置回退。Seedance 2.0 Fast 和 Mini 当前能力表仅支持 480p、720p。成片仍按画幅使用原有导出尺寸。native 使用供应商生成的音画，不叠加独立 TTS、字幕或配乐。旧的 TTS 默认行为及其检查点兼容性需要保留。[原生预检和渲染](app/services/storyboard_native.py) · [定制合并约束](docs/customizations.md)
- **任务身份与付费恢复**。`storage/tasks/<task_id>/storyboard-checkpoint.json` 对应 MPT 本地任务，Seedance 的 `seedance-tasks.json` 记录供应商镜头任务。检查点恢复使用原输入和原安装，先核对原任务，再复用素材或取回已提交的镜头。本地超时或退出后，远端可能仍在运行。缺音轨先复查原素材，未知提交且无远端 ID 时停止补交。保护 `config.toml`、运行数据、凭据和带签名 URL。[检查点身份](app/services/storyboard_checkpoint.py) · [供应商记录](app/services/material.py) · [Wan3 协调](docs/wan3-concurrency.md)
- **上游同步**。`origin` 是 Workie fork，`upstream` 是官方仓库。在独立分支同步，检查官方原文件的差异，保留内部接入与其他供应商入口。定制清单登记影响上游同步的原文件差异。[定制清单与同步步骤](docs/customizations.md)

## 未完成工作

1. **配合 Workie 验收成片修订。** 本地 `storyboard_revision.py` 在 native 成片任务下保存 `native-shots-v1.json`。它通过请求和素材哈希决定可复用镜头，CLI 校验冻结方案及摘要后启动新任务。局部修改要求可信清单及相同镜头顺序和数量，旧成片没有可信清单时仅支持整片重做。已有模拟用例覆盖五镜头仅重做一镜、续跑和旧成片回退。接下来验证 Workie 发来的目标和确认流程、素材缺失时的行为、配置变化和中断恢复。[修订实现](app/services/storyboard_revision.py) · [模拟用例](test/services/test_storyboard_native.py)
2. **复核 Wan3 本地参数改动。** 当前改动将默认并发调整为 3，跨进程上限仍为 5，并将默认轮询间隔和等待上限改为 5 秒和 1800 秒。发布前核对 TOML、环境快照、协调器与并发测试。升级运行环境前先确认已有本地进程和远端任务状态。[Wan3 实现](app/services/wan3.py) · [并发说明](docs/wan3-concurrency.md) · [并发测试](test/services/test_wan3_concurrency.py)
3. **部署及真实供应商验收。** 当前工作树的完整回归、真实 Wan3/Seedance 修订恢复及部署版本尚未在本次交接中核验。先用临时 `MPT_HOME` 和模拟凭据跑定向测试，联调时核对 Workie 实际选中的安装及其 CLI 能力，再按约定确认真实生成范围和费用。

## 验证入口

源码依赖用 `uv sync --frozen` 安装，查看 CLI 用 `uv run python cli.py --help`。定向验证可运行 `uv run python -X utf8 -m pytest -q test/services/test_storyboard_native.py test/services/test_storyboard_checkpoint.py test/services/test_video_run_config.py test/services/test_wan3_concurrency.py`。全量测试、模拟与真实集成测试入口见测试说明。[测试说明](test/README.md)

配套仓库位于同级目录。[Workie AI 交接](../wokie-test/AI_HANDOFF.md)
