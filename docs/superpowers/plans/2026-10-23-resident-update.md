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
- [ ] Trace constant binding and device layout with REA; record narrow evidence.
- [ ] Run bounded layout/binding probes and choose an evidenced route.
- [ ] Add regression tests for that route, observe RED, then implement GREEN.
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
