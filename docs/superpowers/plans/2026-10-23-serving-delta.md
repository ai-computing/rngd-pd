# Safe serving and prepared delta integration Implementation Plan

> **For agentic workers:** Use Superpowers execution and TDD skills task-by-task.

**Goal:** Repair the reviewed failures and connect prepared variants to deployed serving.

**Architecture:** Keep the cluster entry point and reuse resident swap engines.
Independent connector and swap fixes may run in parallel with disjoint ownership.

**Tech Stack:** Python >=3.10, NumPy, existing vendor SDKs and pytest.

**Spec:** ../specs/2026-10-23-serving-delta-design.md

## Global Constraints

- No new dependencies; previous provider session is read-only.
- Prepared graph selection only at drained request boundaries, no request compilation.
- Synchronize the chart relay copy with connector/tcpkv.py.
- No hardware, deployment, merge or push claims without corresponding evidence.

## Review Focus

- Malformed lengths, object arrays and silent TCP peers: fail within memory/deadline bounds.
- Mixed prompt/output lengths at S: completed lanes never index outside S.
- Cancellation while loading or stepping: retain resources until the operation is safe.
- Engine replacement and concurrent release: reject foreign KV and preserve active state.
- Missing/incompatible prepared variants: reject before serving, retaining the current engine.

## Tasks

- [x] 1. Host transport (`connector/transport.py`, `connector/tcpkv.py`, chart copy;
  connector tests): prove unsafe deserialization, fallback drift, large headers and
  stalled reads; implement numeric serialization, stable root, frame/store bounds
  and client/server deadlines; run focused suites.
- [x] 2. UCX and adapter (`connector/ucxkv.py`, `connector/rngd_kv.py`; tests):
  reproduce outstanding-key leaks, polling exceptions and ignored request metadata;
  protect allocation ownership, quarantine uncertain transfers, use request metadata
  or explicitly reject unsupported vLLM scheduler batching; run focused suites.
- [x] 3. Swap ownership (`swap/atom_bundles.py`, `swap/atom_inputs.py`, `swap/rngd.py`,
  `experiments/verify_npu_delta.py`; tests/swap): demonstrate release race and foreign
  KV acceptance; add activation revalidation and per-engine KV identity, assert
  unchanged allocation retention; run all swap tests.
- [x] 4. Serving correctness (`serving/cluster.py`; tests/test_serving_limits.py,
  continuous backlog and solo tests): reproduce mixed-length overrun, max_tokens=1,
  replica ID collisions, silent prompt truncation and unbounded KV preparation;
  enforce per-lane budgets, UUIDs, context validation and finite admission/loading;
  ensure HTTP overload/cancel cleanup; run serving suites.
- [x] 5. Prepared integration (`serving` adapter, existing swap/engine interfaces,
  Helm values and docs; CPU contract tests): adapt resident engines to worker KV
  intake, choose compatible prepared B/S only between requests, retain existing
  continuous same-S tiering, document matching checkpoint identity and opt-in
  device-input limitations; test no-compile selection and unchanged resident weights.
- [x] 6. Verification and review (`.github/workflows/ci.yml`, docs): add portable
  transport suites and relay sync check to CI, run the full local suite with explicit
  environment limitations, independently review the combined diff, fix important
  findings and leave concrete hardware acceptance commands.

Each task uses failing regression tests, minimal implementation, then focused
verification. Final verification runs the combined suite and checks diff/status.

## Execution record

- Baseline f43925c1, clean independent feature checkout. Previous fresh CPU CI
  subset: 90 passed, 1 skipped. Full suite: 143 passed, 3 skipped, 9 environment
  failures (vLLM absent, entry points not installed, Windows os.getuid absent).
- Prior detailed review and user acceptance supply design authorization; continue
  without repeating approval questions. Safety fixes precede integration.
- Implemented disjoint transport, UCX/connector and swap fixes in parallel with
  regression-first tests. Root implemented serving and prepared engine adapters.
- Ruling: preserve the existing feature checkout in a new local branch
  `feat/prepared-serving-safety`, based on f43925c1; do not rewrite or publish the
  earlier PR branch. This keeps the original delta work and provider session intact.
- Ruling: use AtomSession for embedded ATOM B/S switching; the slower device-input
  partial update mode requires explicit config. Arbitrary embedded weight updates
  remain unsupported rather than attempting binary patches.
- Ruling: uncertain UCX RX and advertised TX allocations remain owned until proven
  completion; wall TTL/failed publication ACK is not a DMA fence. This can exhaust
  a finite arena and requires process teardown without SDK drain/cancel support.
- Ruling: vLLM paged scheduler/block layouts fail explicitly. Portable unit tests
  isolate missing vLLM base imports; they do not claim real scheduler compatibility.
- Installed this checkout editable without dependencies to register its entry
  points. POSIX-only shm conformance skips on Windows; CI installs declared
  dependencies and runs every test instead of the former selected subset.
- Final independent review found continued static work after cancellation and
  unbounded HTTP body reads. Both were reproduced RED, then fixed GREEN: stop
  canceled lanes between operations and skip later canceled KV loads; cap fixed
  and chunked body sizes and enforce an absolute read deadline.
- Final full CPU verification: `python -m pytest -q tests --tb=short` completed
  with 260 passed, 6 skipped, 4 subtests passed in 21.77 s. Skips cover absent real
  vendor SDKs and Windows POSIX conformance. Relay source/chart copies match.
- Hardware token parity/latency, actual NIXL DMA recovery, real vLLM integration,
  images and Kubernetes rollout remain separate acceptance gates documented in
  docs/prepared-serving.md. No hardware jobs or prior-session resources touched.
