# Wan3 并发与跨进程名额

分镜与非分镜 Wan3 素材生成默认同时运行 5 个工作单元，使用同一滑动窗口调度器。每个单元提交、等待、下载并校验一个镜头，完成后才补位；返回路径和素材记录始终按原镜头顺序排列。提示词、请求时长、分辨率和后续旁白、字幕及合成流程沿用原逻辑。

`config.toml` 的 `[app]` 可配置：

```toml
wan3_concurrency = 5
```

允许 1～5，超过 5 按 5；无效值（包括小于 1、布尔值和非整数）回退为 5。旧配置没有此字段时立即使用 5，不为补充此字段写回用户配置。设置为 1 可恢复串行素材调度；它仍受安装级共享名额和提交节流限制。

首次启用前，须确认旧版 MPT 进程及其已提交的百炼任务均已结束，再启动新版。旧任务未登记到协调器，运行中的任务不能切换实现。

同一 runtime root（支持 `MPT_HOME`）下的 MPT 进程共享固定 5 个远端任务名额。协调器使用 Windows/Linux 标准库文件锁及原子 JSON，在 `.agent-logs/wan3/` 保存记录，不依赖 Redis。`wan3.generate_videos()` 本身执行准入，因此 CLI、API、WebUI 和直接调用均经过同一限额。创建请求共用提交锁，间隔至少 250 毫秒；轮询不持有提交锁，下载不占远端名额。

镜头失败后，调度器停止补交，等待名额或提交锁的工作单元在 POST 前再次检查停止标记。已提交的其它任务继续等待、下载并保存可用结果，最后报告首个实际错误，不进入最终合成。完成数量仅统计校验通过的素材，原进度文件格式不变；不会自动重新生成失败镜头。

进程退出、本地轮询超时及网络中断均不能证明百炼任务已结束，也不代表已取消任务。协调记录中的 `reserved` 表示尚未提交，可以安全重用；`submitting` 表示提交可能已发生；`submitted` 包含远端任务 ID。记录先于现有任务 ID 回调持久化。

后续请求取得已退出进程的名额锁后，只读查询已知 ID；只有 `SUCCEEDED`、`FAILED`、`CANCELED` 或 `CANCELLED` 才能回收。正在运行、无法查询、不同服务地址或没有 ID 的不确定记录均保留，不根据记录年龄、PID 或超时清除。没有可用名额且有不确定记录时会报告需要核实的错误。无 ID 的记录需先在百炼控制台核实该提交并确认终态，再由维护者处理对应记录；核实前不要删除记录或自动重试付费提交。

共享范围仅包括使用同一 runtime root、本地磁盘协调文件的 MPT 进程。不协调其它安装、其它机器和外部应用的账户调用。不得将协调目录放在不支持这些锁语义的网络文件系统上。

验证使用模拟服务、线程事件与独立子进程，不提交真实付费任务：

```powershell
uv run pytest test/services/test_wan3_concurrency.py test/services/test_storyboard_timing.py test/services/test_cli_progress.py test/services/test_cli.py test/services/test_task.py test/services/test_material.py -q --tb=short -p no:cacheprovider
```

模拟测试证明任务实际重叠执行及共享上限、节流生效，不承诺供应商运行时间缩短。此变更不包含本地编码优化、模型切换或前端改动。

2026-11-03 本地验收：上述命令得到 355 passed、2 skipped、1 failed，另有 74 subtests passed。唯一失败是 `test_invalid_download_is_not_reused_as_cached_video` 的 Windows 路径分隔符断言，已在未修改的 HEAD 代码副本中复现；没有为此改动无关的下载逻辑。测试使用隔离的临时 `MPT_HOME`、示例配置和模拟 Wan3 凭据，未使用用户真实生成配置。并发专项包含提交后节流记录写入失败、未知恢复状态、跨进程五路上限及退出恢复测试。

实际运行以下静态检查均通过：

```powershell
uv run ruff check app/services/material.py app/services/task.py app/services/wan3.py app/services/wan3_coordinator.py test/services/test_wan3_concurrency.py test/services/test_cli_progress.py test/services/test_storyboard_timing.py test/services/test_task.py
git diff --check
```
