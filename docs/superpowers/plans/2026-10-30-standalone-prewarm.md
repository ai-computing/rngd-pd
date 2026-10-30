# Standalone prepared serving implementation plan

> Execute inline with executing-plans and TDD; no further subagents authorized.

Goal: run real prepared serving without vLLM and optionally execute each native
variant before readiness. Spec: ../specs/2026-10-30-standalone-prewarm-design.md.
Keep the reviewed runtime source and wire format intact; no new dependencies.

1. RED: fresh subprocesses block vLLM/rm_core imports, publish real shm K/V with
   a standalone worker connector, consume through cluster's actual loader and
   verify literal values; plugin import must fail clearly when vLLM is blocked.
   GREEN: extract shared worker connector, retain vLLM adapter contract and
   switch cluster._conn. Run new tests and existing connector tests.
2. RED: config rejects nonboolean warmup; real adapter factory runs every shape
   before worker readiness only when enabled; private KV clears after success,
   native failure and cancellation; failed startup closes engine. GREEN: minimal
   adapter warmup and exception-safe factory cleanup; run focused/full suite.
3. Review the full integration, document production/native replay contract,
   freeze and push feature branch. No main merge or PR retarget/close.

Pre-flight: both tasks meet at cluster startup; connector imports must remain
lazy and warmup completes in the factory before PreparedDecodeWorker publishes
readiness. Existing prepared/default atom-inputs preparation stays intact.

## Ledger

- Source: 568ef517e9094b9eff4a25ef7b206604d033b5b5, reviewed merge with
  371 passed / 8 skipped / 4 subtests passed.
- Ruling: parent explicitly authorized this detailed extension; implement the
  supplied architecture without another approval gate. Cost if wrong: the
  feature remains isolated on the integration branch, unmerged into main.
- Task 1 complete: initial RED included missing standalone connector and
  unhelpful optional-plugin import failure. GREEN: fresh producer/consumer
  subprocesses transfer literal K/V through real shm with all vLLM and rm-core
  imports blocked. The consumer uses cluster's actual loader and consumes the
  producer's files after the producer has exited. No stub modules are injected.
  Existing vLLM adapter metadata and TCP tests remain green.
- Task 2 complete: initial RED included invalid warmup values accepted, no
  warmup steps and no cancellation/error cleanup. GREEN: exact bool validation,
  every-shape real adapter execution before readiness and private KV cleanup.
  CPU RNGD tests cover both host-worker counts with warmup off/on; embedded
  ATOM tests cover omitted/false/true warmup and RuntimeError/KeyboardInterrupt.
- Review fix RED: an initialized ATOM runtime was not closed when its artifact
  lacked a configured shape. GREEN: protect preparation/validation as well as
  warmup with engine cleanup. The regression checks released runtime/program
  references after the rejected startup.
- Final complete CPU suite: 387 passed, 8 skipped, 4 subtests passed in 23.85 s
  using the existing resident-opt Windows venv with PYTHONPATH=src.
  `git diff --check` passed. No dependency additions or NPU/remote actions.
- Final review: self-review under the explicit no-subagents instruction;
  checked shared wire implementation, vLLM constructor/base/metadata retention,
  default behavior, cancellation cleanup and readiness ordering. No material
  unresolved source findings. Native runtime behavior and actual prompt replay
  remain assigned to the hardware owner; CPU tests do not establish them.

## Fresh final review and bounded fix pass

The parent supplied a fresh read-only review: no Critical/Important findings
under the reviewer's original grading, with one cleanup finding. Static
comparison against the vLLM 0.18.0 base found no MRO/metadata incompatibility;
the reviewer reported 69 focused tests passed and 3 skipped.

- Regrade: Important. A warmup KeyboardInterrupt followed by a failed RNGD close
  synchronization replaced the cancellation with RuntimeError and left the
  engine usable. This violates the explicit startup cancellation/fail-closed
  contract even though readiness was never published.
- RED: three real tiny-CPU RNGD adapter cases (KeyboardInterrupt, SystemExit,
  RuntimeError) reproduced replacement of the original startup exception.
- GREEN: preserve the original BaseException object with cleanup failure as its
  explicit cause. Mark RNGD failed before close synchronization; if the fence
  fails, retain native weights, graphs and checkpoint handles rather than
  forcibly freeing potentially live buffers. Private adapter KV remains
  discarded and readiness remains false. A retry close releases ownership
  after synchronization recovers. All three regressions pass.
- The serialized kernels and native computational bodies are unchanged.
  This is the single authorized review fix pass; no second review requested.
- Final fix-pass CPU suite: 390 passed, 8 skipped, 4 subtests passed in 24.61 s
  with the existing resident-opt Windows venv and PYTHONPATH=src.
  `git diff --check` passed. Native hardware gates below remain open.

The reviewer's declined-to-judge list, verbatim:
"installed-vLLM execution; nativeSDK synchronization/allocation behavior;
realprompt/KV HTTP replay; hardwareperformance; UCXrecovery; deployment.
Unchangednativekernels notexhaustivelyreviewed."

Parent rulings on every declined item:

- Ruling: installed-vLLM execution remains unvalidated at runtime; existing CPU
  plugin contracts and static review are the evidence. Cost if wrong: a live
  installed-vLLM deployment can still expose integration differences.
- Ruling: native SDK synchronization/allocation effects and real prompt/KV HTTP
  replay remain mandatory hardware gates. Cost if wrong: CPU green could hide
  device ownership or native correctness failures; it must not authorize rollout.
- Ruling: make no performance/SLO claim from this integration run. Cost if wrong:
  startup/serving latency could be represented without native measurements.
- Ruling: UCX recovery is not covered; the upcoming replay selects shm. Cost if
  wrong: UCX DMA/error recovery can violate ownership assumptions independently.
- Ruling: generated Kubernetes ownership proof is not production deployment.
  Cost if wrong: rollout/operational behavior would be asserted without evidence.
- Ruling: unchanged kernels retain their previous frozen proof and are not
  recertified for arbitrary configurations. Cost if wrong: untested shapes,
  model layouts or SDK versions could be mistaken for validated ones.
