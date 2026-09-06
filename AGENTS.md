# AGENTS.md

## Repository Ownership

This repository owns financial content ingestion, transcription, chaptering,
knowledge/evidence, summaries, retrieval, and the `content-factor-signal.v3`
producer contract.

It does not own `stock_agent` or `stock_factor` implementations and must not
import either service directly. Quant integration is through the explicit HTTP
client/contract boundary already present in this repository.

## Architecture Boundaries

- Keep domain code independent of framework, database, and transport adapters.
- Cross-service integration must use an explicit contract, HTTP, or message
  interface; do not import another service's implementation.
- Do not move another service's responsibility into this repository to finish
  a local task.
- Database schema changes require a numbered migration and matching tests.
- Public contract changes require compatibility/regression coverage.

## Critical Paths

- `src/stock_content/domain/`: pure domain models and deterministic policies.
- `src/stock_content/application/`: pipeline orchestration and lifecycle use
  cases.
- `src/stock_content/api/`: HTTP entry points.
- `src/stock_content/adapters/`: external, persistence, media, and search
  integrations.
- `tests/`: deterministic unit, contract, architecture, and pipeline tests.
- `contracts/` and `migrations/`: externally meaningful schemas and database
  evolution.

## Build and Test Matrix

Install the repository's test dependencies with:

    python -m pip install -e ".[test]"

Fast unit suite:

    python -m pytest tests/test_claim_type_system.py tests/test_verification_lifecycle.py tests/test_knowledge_conflict.py tests/test_knowledge_lifecycle.py tests/test_content_factor_signal_v3.py tests/test_content_snapshot_identity.py tests/test_checkpoint_resume.py tests/test_vector_rebuild.py tests/test_financial_event_extractor.py tests/test_support_status_contract.py -q

Architecture gate:

    python -m pytest tests/test_architecture.py -q

Contract gate:

    python -m pytest tests/test_artifact_contracts.py tests/test_content_factor_signal_v3.py tests/test_support_status_contract.py -q

Full relevant deterministic pipeline suite:

    python -m pytest tests/test_pipeline_integration.py tests/test_pipeline_replay.py tests/test_content_snapshot_identity.py tests/test_checkpoint_resume.py tests/test_quant_snapshot_lineage.py -q

Full local test suite:

    python -m pytest -q

Lint:

    python -m ruff check src tests scripts

Run only commands whose dependencies are available locally. Docker/Postgres/
Qdrant integration is an environment gate; do not bypass it with business-code
changes when the environment is unavailable.

## Generated and Large Paths

Avoid reading or changing runtime data, database files, caches, logs, generated
artifacts, or model/media data unless the task explicitly requires it.

## Change Rules

- Make the smallest coherent patch.
- Do not perform unrelated refactors or repository-wide formatting.
- Do not add production dependencies unless the task explicitly permits it.
- A bug fix must include regression coverage when feasible.
- Do not weaken, delete, skip, or rewrite a test only to make an implementation
  pass.
- Do not modify `AGENTS.md` or `.codex/` as part of an ordinary feature task.

## Code Review Rules

Review in this order:

- correctness and failure handling;
- data integrity, idempotency, and replay behavior;
- service-boundary and contract violations;
- backward compatibility and unexplained contract drift;
- missing meaningful tests;
- unrelated files or generated artifacts.

P0/P1 findings block acceptance. P2 findings should be fixed or explicitly
recorded with a reason. P3 style-only findings do not block acceptance.

## Definition of Done

A repository-level feature is complete only when:

- every acceptance criterion has concrete evidence;
- targeted tests pass;
- the full relevant deterministic suite passes;
- lint passes, or its non-applicability is explicitly recorded;
- the final diff contains no unrelated changes;
- no high-severity review finding remains;
- no public-contract or schema drift is unexplained;
- the baseline result and any pre-existing failures are recorded.

## Conversation Workflow Trigger

Treat a user message beginning with the following exact form as a workflow
control instruction:

    @开工 <task-id> [economy|safe] :: <feature request>

`economy` is the default. Confirm the parsed task id, mode, and request in one
line, then take action without asking for a plan confirmation unless the form
is incomplete or the requested action is unsafe.

- In `economy`, delegate all source edits to exactly one `terra_implementer`.
  At most one narrowly scoped, read-only Luna explorer or tester may run beside
  it. Do not let the parent or any second agent edit source files.
- In `safe`, first obtain a read-only `sol_reviewer` assessment, then delegate
  source edits to exactly one `terra_implementer`; request a final Sol review
  for the changed invariants. `sol_implementer` is allowed only when an
  explicit escalation documents why Terra is insufficient.
- Check the worktree status before edits and preserve unrelated user changes.
  If this chat is not in an isolated worktree, state that fact and offer the
  repository launcher rather than silently mixing a feature into the base
  checkout.
- Do not edit another repository. Report the required contract change as a
  bounded task for that repository or for `@联调`.
