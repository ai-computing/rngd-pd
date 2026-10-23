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
- [x] Task 2: Implement the smallest hardware-proven transfer optimization.
  - Probe CPU→resident versus temporary-device copies and sparse column/LM row
    transfers; record SDK layout/lifetime/completion evidence.
  - Tests: real fresh-transform parity, non-square q scaling, disjoint/edge rows,
    LM chunk boundaries/padding, malformed row metadata, rollback and payload.
  - Implement checkpoint row reads/native copying in rngd.py only after RED and
    native support; preserve conservative whole-tensor paths where unproven.
  - Add exact phase/payload reporting needed to assess the optimization. Improve
    planner duplication/bounded refinement only if measured and independently
    covered by tests in test_delta.py/test_load.py.
- [x] Task 3: Validate the frozen change and publish.
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
- Native group-8 [8,14336,4096] probe passed original-parent compiled consumption
  and neighbor checks. Full-layer direct CPU copy measured 19.6–21.9 ms versus
  temporary upload 18.8–26.1 ms plus copy 5.2–11.4 ms. Sparse eight-column staged
  upload/copy measured about 3.3–4.1 ms versus direct host sequencing 22.9–26.6 ms.
  Ruling: direct copies for contiguous full projections/LM rows; small temporary
  device patches for sparse projection columns. Avoid a new HAL dependency.
- Task 2 RED: 23 sparse-projection/LM/bounds cases failed on the old updater.
  GREEN: slice before conversion, stage matching old patches, retain allocations,
  coalesce/validate rows, split LM runs at chunk boundaries, preserve padding,
  and report validation/staging/rollback-staging/copy phases. Missing row metadata
  still updates whole tensors; header changes conservatively update all rows.
- Added partial-write interruption/rollback, real safetensors planner-to-updater,
  dtype/q rounding and overlapping/header-change coverage. Full local suite:
  294 passed, 6 skipped, 4 subtests passed in 22.33 s.
- Ruling: do not change planner header/refinement code without evidence of a
  material remaining bottleneck. This keeps the production diff in rngd.py.
- Fresh independent review: 89 swap tests and eight extra tied-embedding CPU
  cases passed. Important finding 1 reproduced RED: alternating row runs caused
  256 native copies. GREEN: after coalescing, more than four runs or at least
  half the tensor rows falls back to one complete effective range. New/rollback
  staging and byte accounting consume the same selected ranges. The threshold
  is conservative for this validated SDK, not a universal optimum.
- Important finding 2 is a native-validation gap, not a demonstrated CPU bug:
  nonzero-offset LM sources and one-row projection sources need native parent
  consumption/readback checks. Hardware investigator is completing those checks.
- Review scope rulings: immutable checkpoint and complete manual-plan contracts
  remain caller requirements; 2026.3 SDK behavior is unclaimed. Full 8B speed and
  token parity and ownership cleanup remain required hardware/publication gates,
  not silently deferred work. No Minor findings were reported.
- Review fix verification: 45 focused RNGD tests passed; full local suite
  296 passed, 6 skipped, 4 subtests passed in 23.29 s. Run the full-engine
  comparison only on the frozen post-review source.
- Native validation gap closed: nonzero-source-offset single-column projection
  and packed LM chunk-3 probes passed original-parent execution and complete
  readback/neighbor checks, three repetitions each without fallback.
- Frozen 719c09b hardware comparison: all 24 updates/768 tokens equal independent
  fresh references. Sparse projection payload 64x smaller, LM 16032x smaller;
  all optimized allocations/compiled objects retained. Full-tensor run showed
  no speedup under host-load drift; retain every sample and state that limit.
- Independent publication audit verified 86 original file SHA256/length pairs,
  canonical Git source snapshots, live Onggi ranges, actual payloads, all token
  comparisons and UID-checked cleanup. No owned processes/device handles remain.
- Local 296 passed/6 skipped/4 subtests; frozen Linux CI 300 passed/2 skipped/
  4 subtests. README/API and experiment evidence published on the isolated
  feature branch. Draft PR #29 stacks on pending prepared-serving PR #27;
  production deployment/merge is outside this change.
