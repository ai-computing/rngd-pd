# Resident update implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Reduce resident NPU update and execution cost without reloading the model.

**Architecture:** Inspect narrow SDK binding/constant-buffer functions with REA
while profiling the existing external-weight graph. Implement the smallest
route supported by actual device evidence; preserve transactional updates.

**Tech stack:** Python 3.10+, PyTorch, rebel 0.11.0 / torch-rbln 0.2.2,
furiosa-torch 2026.2.0, REA CLI.

**Spec:** ../specs/2026-10-23-resident-update-design.md

## Global constraints

- Prior provider checkout/session read-only; work in this isolated clone.
- Device probes only in owned allocated Pods, through hub.orca.run_remote.
- Preserve immutable checkpoint inputs; never patch installed vendor binaries.
- No new production dependencies; record SDK and native binary provenance.

## Review focus

- Changed weight layouts must be identical on initial load and partial update.
- A failed transfer/probe must retain usable previous weights or fail closed.
- Different B/S programs must share unchanged resident weights.
- Fresh-output parity must use a discriminating changed fixture.
- Timing must include deferred transfers and synchronization.

## Tasks

- [x] Inspect current implementation and run existing swap tests (55 passed).
- [x] Trace constant binding and device layout with REA; record narrow evidence.
- [x] Run bounded layout/binding probes and choose an evidenced route.
- [x] Add regression tests for that route, observe RED, then implement GREEN.
- [x] Run all CPU tests and independent code review; address material findings.
- [x] Validate partial updates and steady decode on actual ATOM/RNGD as relevant.
- [x] Publish sanitized evidence, update usage/limits, and create a reviewable PR.

## Execution ledger

- Baseline: swap suite 55 passed. Initial all-tests collection lacked src on
  PYTHONPATH and HTTP dependencies; creating a private test venv, not changing
  production dependency declarations.
- REAP downloaded at 349a59ba476675889b6d88536fb42936ec02af83. Selected
  feature tracing/function deep-dive/triage, rather than broad library import.
- Ruling: proceed without another design approval question: the user explicitly
  requested autonomous implementation after correcting the goal twice.
- Baseline test environment repaired with a private editable install: 260 passed,
  6 skipped and 4 subtests passed. No production dependencies were changed.
- Native toy: one 32 MiB transform group updated within 512 MiB of resident
  weights. B1/B4 changed outputs matched CPU; production updater's reverse
  change reproduced both original NPU outputs exactly.
- Independent whole-branch review found two Important issues: overlapping
  eligible transform counts and interruption during native mutation. Both
  reproduced RED and fixed GREEN. Interrupted rollback poisons the updater;
  interruption after updater commit closes the owning engine.
- The real two-layer smoke exposed gen-mode compilation's requirement for
  explicit CPU example inputs. Regression reproduced RED, then fixed GREEN.
- Review also set aside host-parameter pointer lifetime and standalone aliased
  replacement tensors. Both were promoted to fixes: keep host binding owners
  alive and stage independent tensor copies. Weak-reference and swap-value
  regressions reproduced RED, then fixed GREEN.
- Review scope rulings: HTTP factory deployment and new RNGD behavior remain
  outside this opt-in SDK change; dedicated worker processes are required for
  the SDK's global context; only the intended Llama graph is claimed. Cost:
  this is an experimental SDK engine, not a default production-serving rollout.
- Delta-call time alone does not independently prove native transfer completion;
  the benchmark also measures synchronous prompt-through-first-token latency.
- Final fix suite: Windows Python 3.11, 279 passed / 6 skipped / 4 subtests.
  Linux Python 3.10 GitHub run 39908041073: 283 passed / 2 skipped / 4 subtests.
- The fixed engine passed a real two-layer Llama smoke (one native program,
  23.96 ms median decode step); the full 8B matrix uses frozen ecfc6c3.
- Full ATOM 8B hardware matrix passed: four alternating updates, all 32
  generated tokens equal to discriminating fresh references, one unchanged
  native runtime/program, and one 117,440,512-byte transform group per update.
  Live onggi planning through the first updated token measured 2.66–2.74 s;
  steady steps measured 68.71–68.75 ms. Three initial compile/load observations
  were 151–168 s. No new RNGD behavior is claimed by this ATOM optimization.
- Root independently validated all result assertions, the original Linux result
  SHA-256 and all 74 canonical Git source hashes before publishing the data.
- Matched existing-path control passed three steady requests: median decode
  510.49 ms versus 68.71 ms for the constant path (7.43× ratio). This compares
  graph/KV execution paths, not an isolated weight-packing or update-call change.
- All 23 exported result/log/config files match their remote byte hashes. Root
  joined native context PIDs to same-sample owned Pod cgroups: three constant
  worker contexts and one control context were observed, with rounded maximum
  allocations of 14.2/14.9 GiB respectively.
- Owned Pod exited normally, exit 0. UID-precondition deletion removed the Pod
  and namespace; no own processes or device contexts remained. Evidence, source
  and caches were retained. Current-card firmware version was not captured.
- Publication: PR #28, stacked on prepared-serving PR #27. Keep it draft while
  its draft base is unresolved; the SDK implementation, hardware gate, review
  fixes and evidence are complete. No production rollout or new RNGD behavior.
