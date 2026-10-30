# Native device binding implementation plan

> For agentic workers: use executing-plans and verification-before-completion.

Goal: actual RNGD B2 and ATOM B1 with persistent allocation ownership, unchanged
PR76 JSON schemas, runtime-valid conformance fixtures, native evidence and LOC.
The user's detailed execution request authorizes these implementation choices.

Architecture: a small serial JSON socket proxy calls the same in-process owner
for all kit operations. RNGD maps supported BF16 artifact rows to existing
serving allocations, journals actual device readback, verifies every written
unit and retains successful undo. Public ATOM session loading reuses compiled
graphs and always plans reload. Existing experimental constant attachment is
separately labeled. No unused mirrored device buffer is conformance evidence.

- [ ] Add streaming external fixtures to the neutral kit; retain its default
  synthetic cases and all tightened assertions. No vendor data parser in onggi.
- [ ] Verify public Torch device readback/copy on one bounded RNGD Job first.
- [ ] Implement pure resident metadata planning for BF16 lossless transpose,
  grouping and LM padding only. Refuse unmappable/header/host/lossy changes.
- [ ] Stage all units, serialize whole requests, journal actual resident bytes,
  verify writes before publishing identity; restore/verify or unload on failure.
- [ ] Test wrong base, tampered plan, corruption, partial write, failed restore,
  retained successful rollback and concurrent request boundaries on CPU.
- [ ] Freeze source/input/full fixture SHAs and run the unchanged kit assertions
  with declared native fixture-provider input; compare generated tokens before,
  after and after rollback to independent fresh references.
- [ ] Implement/test ATOM B1 then native conformance using compiled graph reuse;
  one ATOM card only while measuring, no runtime compilation.
- [ ] Native corruption/drain checks and optional n5 ATOM timing after B1 proof.
- [ ] Report adapter-only/runtime physical and nonblank LOC, exclusions and
  exact source; audit public API and enumerate inherited internal dependencies.
- [ ] Issue branches/PRs, independent review, green checks read, owned Jobs/Pods
  cleaned, board A3-RNGD-B2/E9 rows and milestones. NUMBERS.md owned by Claude.

Fixed safety/validity: one cdx- Job at a time per hardware stream, request/limit1,
TTL/deadline, explicit allocated ordinal, preserve failed attempts, no measured
compilation/retry. Readback comes from actual serving allocations; target-file
hashes are expected values only. Initial load/compile and pointer journals are
accounted separately. Native assertions cannot be inferred from fake tests.

Scope decisions: native runtime fixture cases exercise only supported lossless
units; default arbitrary model.bin fixtures are not passed to a Llama engine.
Unsupported switches return reload with reason; unsupported apply fails without
changing the serving state. Existing private SDK graph/cache imports are flagged
as inherited dependencies, distinct from the public row-copy/readback adapter.
