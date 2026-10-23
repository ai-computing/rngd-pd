# RNGD resident optimization implementation plan

> Use superpowers:executing-plans for inline implementation and a fresh final
> whole-branch review. Specialized investigators own read-only analysis/probes.

**Goal:** Reduce RNGD resident update preparation and transfer cost while
preserving the existing fast decode graph and exact fresh outputs.

**Architecture:** Retain existing shared allocations. Select a native transfer
route from measured probes, stage only required data, and make the update
transaction safe under interruption. Reuse the existing planner and API.

**Tech stack:** Python 3.10+, PyTorch, furiosa-torch 2026.2.0, existing Onggi CLI.

**Spec:** ../specs/2026-10-23-rngd-resident-optimization-design.md

## Global constraints

- Prior provider/checkout read-only; ATOM branch unchanged.
- Remote access only hub.orca.run_remote; owned allocated devices only.
- No SDK patches, global configuration changes or production dependencies.
- Native route must pass original-parent update, neighbor and execution checks.
- Exact q conversion order, LM chunk/padding and old-KV rejection are mandatory.

## Review focus

- Partial copy or cancellation after native mutation cannot leave serving enabled
  with mixed weights; rollback interruption must poison the engine.
- Sparse source rows become noncontiguous destination columns for projections.
- Supplied row metadata must be bounded and cannot bypass real layout checks.
- Missing row metadata, full-tensor changes, and group-mode callers stay valid.
- Timing must include synchronization/deferred work through the first token.

## Tasks

- [x] Inspect connected code/evidence and run clean baseline swap suite.
- [x] Task 1: Harden the touched update transaction.
  - Tests in tests/swap/test_rngd_delta.py: KeyboardInterrupt during copy/sync,
    interrupted rollback, partial group/reference commit, and stale KV.
  - Observe RED, then modify rngd.py rollback boundaries/state restoration and
    generation handling. Run the focused suite GREEN and commit.
- [ ] Task 2: Implement the smallest hardware-proven transfer optimization.
  - Probe CPU→resident versus temporary-device copies and sparse column/LM row
    transfers; record SDK layout/lifetime/completion evidence.
  - Tests: real fresh-transform parity, non-square q scaling, disjoint/edge rows,
    LM chunk boundaries/padding, malformed row metadata, rollback and payload.
  - Implement checkpoint row reads/native copying in rngd.py only after RED and
    native support; preserve conservative whole-tensor paths where unproven.
  - Add exact phase/payload reporting needed to assess the optimization. Improve
    planner duplication/bounded refinement only if measured and independently
    covered by tests in test_delta.py/test_load.py.
- [ ] Task 3: Validate the frozen change and publish.
  - Matching full/sparse native workload, fresh references, both directions,
    retained weights/programs and every actual sample/source hash.
  - Full relevant CPU suite, fresh independent review and one RED→GREEN fix pass.
  - README/API/evidence update, isolated feature PR, CI and UID-checked cleanup.

## Execution ledger

- Base da85c80ba5f0eb979d32bbcbbbcd4b1de13b0c13; branch
  feat/rngd-resident-delta-opt in independent clone.
- Baseline swap suite: 55 passed in 1.85 s.
- Ruling: continue autonomous implementation under the user's repeated request
  to optimize, with no additional design/permission menu. This plan records the
  implementation choices and evidence gates for review.
- Code investigation: refined rows are ignored; projection transpose makes
  destination columns noncontiguous. Existing KV source warns of host bridging.
- SDK source investigation: CPU→existing-device copy is documented; HAL copy
  exposes a rectangular sequencer. Actual device behavior is pending probes.
- Environment ruling: old allocation/deletion ran on dell_0 control plane; data
  and SDK live on rebellion. Newly inherited rebellion kubeconfig was invalid;
  use existing authorized dell_0 configuration without credential changes.
- Task 1 RED: interrupted writes left new weights/old KV usable; interrupted
  rollback escaped without poisoning; interrupted group commit retained new
  references. Four new regression cases reproduced the missing behavior.
  GREEN: catch BaseException, restore original group/per-batch/host references,
  poison any failed/interrupted rollback and invalidate KV before mutation.
  Focused RNGD suite: 13 passed in 1.44 s.
