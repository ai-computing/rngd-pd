# Common runtime transition implementation plan

Goal: execute one caller-verified variant change through the real RNGD or ATOM
owner using their existing contract, without modifying vendor SDKs or drivers.

Architecture: add a shared explicit caller and thin transition methods to both
owners. Hold the existing owner RLock across describe/resident/plan and either
B2 apply or explicit lifecycle reload. Expose this additional caller operation
through the existing persistent JSON transport; preserve the five neutral ops.

Scope: existing RNGD bounded BF16 B2, ATOM public B1, no GPU or ATOM B2 claim.
Caller authenticates immutable artifacts/manifests and a complete truthful byte
plan. Wrong resident base is an error even when B1 could otherwise reload.
Never hide a failed admitted update behind automatic retry or reload. Native
inference must use request() and therefore participates in the same owner lock.
Existing measured sources and raw evidence are frozen and will not be edited.

Ruling: reuse the previous public AtomSession's parsed compiled object during
replacement only when a freshly streamed graph SHA matches the prior admission.
AtomSession already accepts a public compiled object. Retain that host object
across old runtime teardown; create a new runtime with no live-card overlap.
Explicit close releases it; changed graph bytes reparse. This avoids repeated
host parsing, not device weight upload. Native performance remains unmeasured.

- [x] Add tests on both real owner classes for common dispatch, wrong base,
  malformed plans, admitted-update failure/rollback, failed reload and draining.
  Add transport end-to-end caller coverage. Run and observe missing API failures.
- [x] Implement src/rngd_pd/swap/binding_transition.py and thin owner entrypoints;
  extend binding_cli transition --target reading a switchplan from stdin.
- [x] Run focused/connected tests; document actual support and ATOM SDK boundary.
  Verify no native compiler/device imports in transport. Read-only SDK/public
  documentation checks are not native execution or a proof of API absence.
- [ ] One fresh read-only whole-change review, fix important issues with tests;
  commit, draft PR, read exact-head CI and integrate only after required checks.

Tradeoff: a caller composes existing operations rather than inventing new stage
tokens/SDK hooks. It does not implement driver-level atomics, expose allocation
addresses, or connect CSI automatically. Current result is CPU-tested orchestration;
new native validation is separate from previously published binding evidence.

Validation: swap suite441 passed/1 Windows Unix-socket skip, independent reviewer
92 passed/1 skip and no Important/Critical findings. Initial CI had745 passing
tests and one failure: the historical B2 verifier compared evolving current
runtime files to a measured old revision. Explicitly refreshed the stale local
integration tracking ref and merged d5da9d1 without conflicts. A RED/GREEN
regression now verifies pinned historical Git source bytes; missing objects
still require matching historical checkout bytes. Raw data/golden summaries
are unchanged. CI checks the new verifier SHA separately and compares all
historical summary fields except the old verifier's own SHA. The complete
recomputed summary matches under that rule;12 verifier tests pass.
