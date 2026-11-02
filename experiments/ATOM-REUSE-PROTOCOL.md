# Native ATOM parsed-host-model reuse

Issue53. Runtime57e610e, public AtomSession, one ATOM card per cdx K8s Job.
Existing authentic B1/S64 Llama-3.1-8B graph, 32layers, unchanged 8-token prompt
and8-token greedy output. Target differs only in publisher metadata; compiled
graph bytes are identical. This is a parsed-host-object optimization, not a
new-shape or changed-weight/device-residency experiment.

Independent variable: whether the previous parsed public RBLNCompiledModel
can be retained. Both arms use merged AtomBinding.transition and the same
public loader. Full-parse arm supplies that loader as an explicit factory,
which disables default-owner parse reuse. Reuse arm uses default public owner.
The baseline factory is a real public native loader, not a fake engine.

Each arm independently loads and generates its base outside the trigger;
only one runtime is live. The trigger starts immediately before transition.
It includes full graph SHA, old close, parse if necessary, new runtime creation,
and all work before the first native target token. Native endpoint is reconstructed
from generation-call entry plus engine prompt TTFT. Entire continuation is
generated and checked but excluded from first-token timing. No HTTP claim.

Both arms use the same already-loaded host Tables. Parse/create native call
counts and session flags must show target parse1/create1 for full, parse0/create1
for reuse; target old runtime must be closed, fresh target tokens identical,
zero compile attempts and15 real native run calls per generation. Reuse retains
the parsed host object identity, not a device buffer identity. Device ordinal
is derived from init container allocation and public rbln-smi table; broad
device visibility is refused. Source/SDK/graph/checkpoint hashes are retained.

Off-card full input hashing precedes card allocation. Admission still repeats
the graph SHA inside every measured transition. Warm/uncontrolled shared OS
cache, fixed source/inputs, no system cache dropping or SDK/driver modifications.
External exact trace collection may read small evidence files on the host; it
is not another benchmark and is recorded as uncontrolled host activity.

Excluded smoke: one pair, full then reuse. Main: five pairs, reversing order
on odd pairs. Fresh target reference is generated and closed before each Job's
matrix. Every failed/partial attempt is retained, no outlier exclusion or
automatic retry. Any invalid sample stops the dependent matrix. Main n5 must
not pool smoke. Report every observation, median and inclusive Q1/Q3/IQR.

Bounded unprivileged Jobs: ATOM request/limit1 for init and application, backoff0,
ttl600, activeDeadline900 smoke/1800 main. Card retained only during the native
measurement Job. Record UID/Pod terminal state and export raw before UID-guarded
deletion. No foreign Job/Pod is touched; live preflight required before creation.

Implementation/verification:
- CPU regressions: alternating order, mode-specific parsed identity/call gates,
  same-target output, positive nonoverlapping clocks, missing/failed trial rejection.
- One fresh read-only review before allocation; fix important findings.
- Freeze/push harness source; off-card source/input receipt; smoke then main.
- Independent local reparse of exact exports; draft PR and green CI before merge.
- Do not edit prior raw, data/NUMBERS.md, or source measured by other jobs.

Outcome: native smoke and five main pairs completed. See
[report](REPORT-ATOM-REUSE-20261102.md) and
[recomputed summary](atom-reuse-summary-20261102.json). Measured source remains
fec2d370; later commits add verification and evidence only.
