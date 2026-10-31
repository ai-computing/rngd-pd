# Reviewed runtime integration, 2026-10-30

Integrate the reviewed RNGD and ATOM branches without changing their behavior.
Use a new worktree and feature branch; leave source checkouts and main unchanged.
Hardware serving verification belongs to the hardware owner.

## Sources

All hashes were checked against GitHub PR metadata on 2026-10-30. PR 28 was
fetched from the public GitHub HTTPS repository, with its fetched hash checked
against the PR head before merging.

| Source | Branch | Exact head |
| --- | --- | --- |
| [PR 27](https://github.com/ai-computing/rngd-pd/pull/27) | validation/prepared-serving-20261023 | da85c80ba5f0eb979d32bbcbbbcd4b1de13b0c13 |
| [PR 29](https://github.com/ai-computing/rngd-pd/pull/29) | feat/rngd-resident-delta-opt | 75252049416ee65397113afbcaddc11d71a7f6d2 |
| [PR 31](https://github.com/ai-computing/rngd-pd/pull/31) | feat/rngd-sdk264-resident | 0bc244db0239bddfdda005454809aee0812877d8 |
| [PR 32](https://github.com/ai-computing/rngd-pd/pull/32) | perf/rngd-host-staging | 52507e49e57fb9ddf2b8936d76672dc609cdc48c |
| [PR 28](https://github.com/ai-computing/rngd-pd/pull/28) | feat/atom-resident-delta-opt | 5375f542a84a7085997c4949732b3c9945a39ebd |

The integration branch `integration/onggi-prepared-serving` starts at PR 32,
which already contains PRs 31, 29 and 27. PR 28 shares PR 27 as merge base.
The merge retains both source heads as ancestors.

## Plan and ledger

- [x] Verify source identities and isolate work at
  `C:/Users/USER/orca/scratch/rngd-pd-integrated-20261030`.
- [x] Run the RNGD baseline CPU suite: 352 passed, 8 skipped,
  4 subtests passed in 31.74 s.
- [x] Merge PR 28 using `--no-ff --no-commit`. Git automatically merged
  README.md, README.ko.md and docs/delta-swap.md; there were no conflicts.
- [x] Review the combined source and run the complete CPU suite:
  371 passed, 8 skipped, 4 subtests passed in 23.19 s.
- [x] Confirm RNGD engines, swap implementation and serving files are unchanged
  from PR 32, and the ATOM constant implementation/tests and delta verifier/tests
  are unchanged from PR 28. `git diff --cached --check` passed.
- [x] Document the HTTP/KV contract and native smoke boundaries below.
- Final review: author self-review; no further subagents were authorized.
  Existing branch behavior and dependencies were preserved. No production-code
  fixes were needed, so no new RED/GREEN cycle was introduced.
- Ruling: use a Git worktree at the explicitly requested target path because the
  available managed worktree tool targets the current hub repository and has no
  repository/path argument. Cost if wrong: the host does not track this attachment;
  the worktree remains registered with Git and is retained for the hardware owner.
- Ruling: preserve the experimental ATOM API boundary. PR 28's
  `ConstantWeightDecode` is available as a Python API and through the delta
  verification harness, while prepared HTTP config still accepts only `rngd`,
  `atom` and `atom-inputs`. Adding an `atom-constants` factory is separate work;
  claiming this merge exposes constant updates over HTTP would be incorrect.

Full-suite command (PowerShell, existing complete CPU environment):

```powershell
$env:PYTHONPATH='src'
& 'C:/Users/USER/orca/scratch/rngd-pd-resident-opt-20261023/.venv/Scripts/python.exe' -m pytest -q tests
```

## Prepared decode contract

The deployed entry point is `rngd_pd.serving.cluster --role decode
--backend rngd --S 256 --sched static --prepared-config CONFIG`. A cached RNGD
configuration uses the existing immutable checkpoint ID, actual checkpoint/cache
paths and a validated variant such as `B4-S256-g2`; `host_workers` may be 1 or 2.
`build_prepared` loads cached graphs with `allow_compile=False`, preparing every
listed variant before readiness. A missing/incompatible cache must fail startup.
Prepared serving rejects continuous scheduling.

`POST /decode` accepts, for example:

```json
{"req_id":"native-smoke-unique-id","first":1,"pos":1,"max_new":4,"checkpoint_id":"sha256:<actual-immutable-checkpoint-id>"}
```

- `req_id` names the separately published KV payload; HTTP does not carry KV.
- `first` is a nonnegative integer, normally the prefill's next token. The
  caller must keep it within the model vocabulary.
- `pos` is an integer at least 1, normally the prompt length; `max_new` is a
  positive integer, defaults to 8, and `pos + max_new` must fit a prepared S.
- `checkpoint_id` must exactly match the prepared config. Missing/mismatching
  IDs fail with 409 before KV is read.
- The response's `tokens[0]` echoes `first`; only `tokens[1:]` establish native
  decode output. `max_new=1` performs no native decode step.
- The worker chooses the smallest fitting B*S and pads unused lanes. Responses
  include `variant`, `batch_width`, `checkpoint_id`, `swap_ms`, `load_ms`,
  `decode_ms` and `tpot_ms`; `swap_ms` measures graph selection only.

For each layer i, publish `req_id/layer.i/k` and `req_id/layer.i/v` using
`rngd_pd.connector.transport.ser(array, wire="bf16")` and the same selected
transport as decode. Use full wire-width arrays to avoid ambiguity:
K per layer `(nkv, hd, wire_S)`, V per layer `(nkv, wire_S, hd)`.
The loader constructs float32 K `(nl,nkv,hd,wire_S)` and
V `(nl,nkv,wire_S,hd)`; the prepared adapter selects the active S prefix and
casts RNGD inputs to its native BF16 device dtype. Producer and consumer must
agree on layer count, rotary/model configuration and checkpoint identity.
This is whole-request KV, not a vLLM paged-cache mapping.

Default transport is shared memory under `PD_SHM_ROOT`; both processes must
share that directory. Explicit TCP requires both to select `AIRM_KV_TRANSPORT=tcp`
and the same `AIRM_KV_TCP_ADDR`. Do not enable UCX merely for this smoke.

## Native verification sequence after standalone/prewarm extension

The subsequent [standalone/prewarm plan](2026-10-30-standalone-prewarm.md)
removes the cluster's vLLM dependency and adds optional `"warmup": true`.
The merge's source hashes and test record above remain its original scope.

1. On hardware already allocated by its owner, load the exact frozen source,
   existing validated group2 cache and matching checkpoint/config. Record source
   hash, artifact/checkpoint identity, SDK, native device and cache hit/failure.
   Do not allocate a second NPU or compile an unrelated graph for this check.
2. Replay the real native prompt through the matching producer and record its
   K/V, next token, position, immutable checkpoint ID and fresh reference tokens.
   Publish full-width layer arrays with `WorkerKVConnector(WorkerRole.WORKER,
   req_id=..., transport=select_transport(), n_layers=...)`; set transport WIRE
   to bf16. Call `wait_for_save`, then close the producer. Its real shm payload
   remains available for the later HTTP consumer. Record payload checksums.
3. Start the actual prepared HTTP consumer with `"warmup": true`, wait for
   readiness and POST those real request fields with `max_new >= 2`. Record
   all output tokens and compare against fresh reference tokens. Do not inject
   vLLM stubs or bypass its production loader; standalone cluster serving needs
   neither. The optional vLLM plugin itself still requires vLLM.
4. Label sequential replay as real prompt/KV decode through HTTP, and separately
   establish a live prefill/gateway deployment. Record cold/prepared TTFT,
   KV import, allocations/HBM, repeated B/S switches, mixed lengths, overload,
   cancellation and stream disconnects before broad deployment claims.

No native hardware, end-to-end prefill, real vLLM paged integration, container
rollout or UCX recovery was run by this integration task. Existing per-branch
native measurements retain their original documented scope.

## Subsequent native gate completed

The separate hardware owner completed the frozenfafd820/onggi466ef918 run.
See [the final experiment record](../../../experiments/2026-10-30-onggi-prepared-serving.md).
Real init, standalone warmup before readiness, actual native prompt/KV,12HTTP
requests/96reference-equal tokens and resident-buffer retention passed. The
first prefix-only KV harness failure is preserved separately; corrected full
S256 wire arrays passed under the same source/profile/resource conditions.
All owned Pods/namespace were deleted with UID guards and independently absent.
The parent independently rehashed205 small evidence files, canonical409runtime
and125Onggi source files, token arrays, profiles/manifests and buffer identities.
Native payloads were not downloaded. This closes this configuration's mandatory
native replay gate; the deployment/transport/arbitrary-shape limits above remain.
