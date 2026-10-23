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
- [ ] Run all CPU tests and independent code review; address material findings.
- [ ] Validate partial updates and steady decode on actual ATOM/RNGD as relevant.
- [ ] Publish sanitized evidence, update usage/limits, and create a reviewable PR.

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
