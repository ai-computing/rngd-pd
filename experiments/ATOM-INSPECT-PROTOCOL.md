# Public ATOM artifact inspection plan

Issue: https://github.com/ai-computing/rngd-pd/issues/55

## Question and scope

What metadata does the installed SDK actually expose for the frozen compiled
artifacts? Use only `rebel.RBLNCompiledModel.inspect(path)`, documented at
https://docs.rbln.ai/v0.11.0/software/api/python/python_api.html.
Keep the complete JSON metadata and errors. Do not interpret proprietary
bytes, construct a compiled-model object, create a runtime, compile, infer,
or request an accelerator. The SDK owns the format interpretation.

The result is metadata evidence, not a serving benchmark. An input shape is
not exhaustive bucket coverage; missing fields are unknown. Allocation
estimates do not establish measured memory or live weight sharing. The
installed package version does not establish an artifact's compiler version.
Retain both independently. No SDK or system installation is part of this work.

## Input and output contract

`bench/inspect_atom_artifacts.py --manifest INPUT.json --input-root ROOT
--output NEW_DIRECTORY` consumes this explicit, pre-frozen input list:

```json
{"schema":"rngd-pd.atom-inspect-inputs.v1","files":[
  {"path":"relative/model.rbln","bytes":123,"sha256":"64 lowercase hex digits"}
]}
```

Use 1–18 unique regular `.rbln` files, a manifest at most256KiB, and an
immutable, read-only input tree. Reject absolute/traversing/backslash paths,
symlinks and output paths inside that tree. Compute each complete input SHA
and compare it before calling inspect; charge hashing separately. Check file
identity/stat stability across hashing and inspection. Those checks do not
replace the immutable-mount requirement or prove absence of adversarial
concurrent writes. Never derive expectations from names or stat alone.

Create a fresh output directory and atomically replace only its own
`result.json` progress snapshot. Preserve every file attempt and continue
after an individual inspection failure. Environment/manifest failures are
explicit nonzero outcomes, not missing capabilities or successful probes.
Checkpoint the running path before hashing and its verified full SHA before
SDK dispatch. A pre-call checkpoint uses `native_api_called: null`: abrupt
termination may occur on either side of the actual call, so entry is unknown.
Interrupted `running` rows never count as completed observations. This is a
progress record, not a claim of crash/power-loss durability or native-call tracing.
Preserve native metadata only when it is a JSON-compatible dictionary;
otherwise report a serialization/type failure without stringifying arbitrary
SDK objects. Record actual command, source SHA, manifest SHA, timestamps,
package versions and separate hashing/inspection durations. Save stdout and
stderr separately at execution time. This is not an n5 latency result.

## Execution boundary

**Native status: not_run.** Existing serial transport owners must finish
before any new node call. A later CPU-only `cdx-` Job needs a deadline, TTL,
read-only model mount, no NPU resource request and an actual inspected input
manifest. No automatic launch, controller or follow-on serving run is added.
Follow-up native dispatch requires its own frozen workload, native alternatives,
one-card budget, quality criterion and smoke/main admission.

## Implementation plan

Goal: make the public metadata prerequisite reproducible without widening its claims.
Architecture: one stdlib CLI; the only vendor boundary is the lazy public inspect call.
Tech stack: Python3.10+, existing Rebel SDK only on the execution node.
Spec: the contracts above; base8124570a36074338930ec7b98b1a66124d4ac9c2.

- [x] Write behavioral tests against real temporary files and an isolated SDK
  boundary; observe failure with an unimplemented probe.
- [x] Implement input admission, exact hash binding, inspection/failure
  collection and fresh atomic output. Run the new tests and related suite.
- [x] Run the full repository command; distinguish environment limitations
  from source failures. Existing Windows Python lacks httpx, so its full
  baseline collection fails in tests/native_v4/test_serve.py. No install is
  authorized here; use existing Linux CI for complete regression coverage.
- [ ] Request one independent connected-source review, fix Important issues
  with observed regression failures, open a draft PR and read actual CI logs.
- [ ] Preserve source/check receipts and update the private experiment board.

Review focus: path escape/symlink or changed inputs; SDK/import/API absence;
non-JSON metadata; failure preservation/partial matrices; rerun/output overwrite.
Tests must not be described as real native inspection.

Ruling: independent source review found that a result appended only after
inspection loses the current attempt on interruption. Two observed failing
regressions therefore require pre-hash and pre-call progress checkpoints;
SDK calls and completed observations remain distinct.

Local validation: new29 passed/1 Windows symlink skip; independent recheck
ran the two interruption regressions plus atomic snapshot failure (3passed)
and found no remaining Important issue. The full local command cannot collect
`tests/native_v4/test_serve.py` because httpx is absent. With only that module
excluded, an intermediate-source run ended801passed/11skipped/4subtests and
two existing package-entry-point assertions failed:
`SelectionTests.test_shm_is_registered_and_default` and
`RunnerAndMetricsTests.test_runner_registered` in tests/test_airm.py. This
worktree is not installed as a distribution; setting PYTHONPATH supplies
imports but not installed entry-point metadata. This is not a full-suite pass.
Complete installed Linux CI is required before integration. Native status
remains not_run; no package install or node call was performed here.
