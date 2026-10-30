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
