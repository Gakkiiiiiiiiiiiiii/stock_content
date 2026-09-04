# 待完成工作与当前证据（2026-09-04）

本文记录当前仓库可核验的完成证据，以及仍需外部协作或生产环境验证的事项。未列为完成的项目不得在发布说明中表述为已交付。

## 当前完成证据

- PostgreSQL 注入回归已真实执行，使用 `postgres:16-alpine`；publication/claim-event 测试共 `13 passed`。对应测试文件为 `tests/postgres/test_publication_postgres.py` 与 `tests/postgres/test_claim_event_postgres.py`。
- PostgreSQL 回归过程中发现并修复了 boolean backfill bug。没有本地 DSN 时，这 13 个测试显示为 skip，属于预期的环境门禁，不代表测试失败。
- API、core-worker、media 镜像已完成实际构建；media 容器已验证 psycopg、qdrant、faster-whisper、paddleocr 可导入。multimodal 镜像完整构建仍未完成。
- readiness 的 SQL publication/outbox/contract-inventory 回归与 foundation/API 定向测试通过；readiness 不再读取内存 snapshot 私有字典。
- platform manifest 校验、SBOM 生成、compose 配置校验和相关 Ruff 检查已有通过证据。

## 待完成项

### P0：发布与平台门禁

1. **stock_factor v5.1 schema hash 漂移**
   - 外部依赖：`stock_factor` 对端 schema/发布流程。
   - 当前事实：发现 `DFF5...` 与本仓 `BC4A...` 两个 hash，不得猜测哪一个为权威。
   - 完成标准：双方确认唯一权威 schema，重新计算 platform manifest 与 strict consumer fixture，并使 contract CI 全绿。

2. **GitHub main branch protection / required checks**
   - 外部依赖：GitHub 仓库管理员权限。
   - 完成标准：main 分支启用 branch protection，要求 lint、contract、P0、quality 和相关 Docker checks，且规则在 PR 上实际生效。

3. **真实 Nightly 连续绿与 Postgres→Qdrant rebuild**
   - 外部依赖：GitHub Actions runner、Docker、Postgres、Qdrant 服务和稳定凭据。
   - 完成标准：Nightly 连续多次成功，真实从 Postgres 读取并写入 Qdrant，报告可追溯且验证 collection/alias 指向。

4. **Qdrant native alias swap**
   - 外部依赖：Qdrant 运维权限与发布窗口。
   - 完成标准：实现并演练原生 alias create/swap/rollback，查询只经稳定 alias，不直接依赖重建 collection 名称。

### P1：运行时与数据治理

5. **multimodal image 完整构建与 GPU/import smoke**
   - 外部依赖：可缓存的 Docker 构建环境、GPU runner 和 CUDA 运行时。
   - 完成标准：`Dockerfile.multimodal` 成功构建，导入 psycopg/qdrant/多模态依赖，并完成 GPU capability/import smoke。

6. **有序 Postgres migration runner 与旧 schema 实升**
   - 外部依赖：真实旧版本数据库备份、迁移执行权限和回滚窗口。
   - 完成标准：按编号有序执行 migration，验证旧 schema 升级、约束/backfill、幂等重跑和回滚/恢复策略。

7. **source retention/governance 生产作业**
   - 外部依赖：正式来源清单、保留期限审批、对象存储/KMS 与删除作业调度。
   - 完成标准：生产入口 fail-closed，metadata、retention、access、tombstone 和 immutable audit 全链路持久化，并完成真实定时作业演练。

### P2：架构与运营质量

8. **`stages.py` / `service.py` 完整模块拆分**
   - 外部依赖：与并行 P0/history/publication 改动协调，避免语义漂移。
   - 完成标准：完成 source/media、claim/evidence、lifecycle/verification、snapshot/signal/persist 的真实拆分；兼容 re-export 仅保留过渡层，golden/replay/signal hash 不变并通过 characterization/architecture tests。

9. **生产 SLO / 性能门禁**
   - 外部依赖：生产指标、基线流量、模型/锁 hash 和性能采集平台。
   - 完成标准：建立可审计的 latency、throughput、outbox/index lag、资源预算与错误预算门禁；性能变化必须绑定相同 hash 或显式新 policy，semantic/replay mismatch 必须阻断发布。

10. **生产 snapshot/membership writer 的真实 PostgreSQL 回滚直接覆盖**
   - 当前事实：13-case PostgreSQL publication 测试对 snapshot phase 使用 no-op `snapshot_writer`，已直接证明 publication run、manifest、sealed signals 和 outbox 的事务回滚；生产同 session 路径已完成代码审查，但仍缺少真实 writer failure injection。
   - 完成标准：调用生产 `SqlSnapshotStore` writer 注入 snapshot phase 异常，并验证异常后 snapshot、snapshot membership、artifact/membership 等相关写入全部无残留。
