# Prepared B/S switching in the cluster worker

`python -m rngd_pd.serving.cluster --role decode --prepared-config FILE` connects
the resident swap engines to the existing `/decode` route. It is optional because
the deployment must supply matching checkpoint files and compiled artifacts.
Without a config, existing static/continuous workers remain available.

## RNGD

Example `prepared-rngd.json` (paths are relative to this file):

```json
{
  "engine": "rngd",
  "checkpoint": "/models/llama-base",
  "checkpoint_id": "sha256:<immutable-checkpoint-manifest-digest>",
  "cache_dir": "/artifacts/rngd",
  "variants": ["B1-S256-g8", "B4-S256-g8", "B4-S512-g8"]
}
```

```bash
python -m rngd_pd.serving.cluster --role decode --backend rngd --chip -1 \
  --S 512 --sched static --batch-wait 10 --prepared-config prepared-rngd.json
```

`SwappableDecode` uploads the shared projections and LM head once. Every listed
variant is prepared at startup with `allow_compile=False`; missing compiled artifacts fail
startup. No compilation or model-weight upload occurs when selecting a prepared
variant. Use the graph-baking commands in [delta-swap.md](delta-swap.md) first.

RNGD can opt into two CPU weight-layout workers at startup with
`"host_workers": 2`; the default is 1. This preserves caller-thread checkpoint
reads and q/BF16 arithmetic. At Torch intra-op=4 it can use eight CPU threads;
budget CPU capacity rather than enabling it on a CPU-limited worker. It applies
to initial projection staging; resident updates remain serial and keep their
existing rollback behavior. The option is rejected for ATOM engines.

For the validated SDK2026.4 B4/S256/group2 cache, an example is:

```json
{
  "engine": "rngd",
  "checkpoint": "/models/llama-base",
  "checkpoint_id": "sha256:<immutable-checkpoint-manifest-digest>",
  "cache_dir": "/artifacts/rngd",
  "host_workers": 2,
  "variants": ["B4-S256-g2"]
}
```

## ATOM embedded weights

Compile all B/S buckets together with the existing native session builder:

```bash
python -m rngd_pd.swap.bench_atom build --ckpt /models/llama-base \
  --variants B1,S256 B4,S256 B4,S512 \
  --session-out /artifacts/atom/buckets.rbln --out build.json
```

The `.rbln` and its `.rbln.json` sidecar must both be available:

```json
{
  "engine": "atom",
  "checkpoint": "/models/llama-base",
  "checkpoint_id": "sha256:<immutable-checkpoint-manifest-digest>",
  "artifact": "/artifacts/atom/buckets.rbln",
  "variants": ["B1-S256", "B4-S256", "B4-S512"]
}
```

```bash
python -m rngd_pd.serving.cluster --role decode --backend atom --S 512 \
  --sched static --batch-wait 10 --prepared-config prepared-atom.json
```

One `AtomSession` retains the compiled model/runtime. Input shape selects its
co-compiled bucket. The worker verifies that every configured shape and the
checkpoint's layer count match the session metadata. This path does not patch
embedded weights or load several independent full models on a small card.

## Request and checkpoint boundaries

The batcher collects at most the largest configured B. Each batch chooses the
fitting prepared variant with smallest B*S, breaking ties by S then B. The
largest B/S combination must be present so mixed long requests can always share
a batch. Requests have individual output budgets; completed and padding lanes
use safe dummy positions. The variant stays selected through the whole batch.
`--sched continuous` with a prepared config is rejected: this implementation
does not migrate in-flight KV across S. Existing same-S continuous tiering is a
separate path.

The prefill process must set `PD_CHECKPOINT_ID` to the exact immutable identifier
in the decode config, and both processes must use the same `--S` wire layout and
model/rotary configuration. A model name such as `main` is not an immutable ID.
The prefill response carries its ID; the gateway forwards it; prepared decode
rejects a missing/mismatching ID with 409 **before reading KV**. Operators must
bind this ID to the actual checkpoint and compiled artifact, and keep files
immutable. The worker cannot cryptographically infer embedded model weights
from an opaque `.rbln` file.

The prepared worker disables speculative `/prefetch`. KV loading starts after
bounded batch admission and stays at the configured wire S until import, where
only the selected S prefix is needed. Responses report `variant`, `swap_ms`,
`checkpoint_id` and actual `batch_width`; gateway responses expose these in `pd`.
`swap_ms` measures graph selection only, not KV import or first-token latency.

## Partial checkpoint updates

The worker exposes a Python control-plane method; there is no HTTP weight-update
endpoint or automatic publication watcher:

```python
result = worker.load_delta(
    new_checkpoint_dir, new_immutable_checkpoint_id,
    store=onggi_store, base_ref="base", new_ref="new",
)
```

Drain traffic from both prefill and decode first. This method rejects any
admitted/queued decode request, blocks new admission during the update, reuses
the existing onggi planner and engine rollback/staging, and commits the new ID
only after success. Failed rollback marks the worker unready. Publish matching
prefill weights and its new ID before resuming traffic. Changing only decode
cannot safely serve KV made with old prefill weights.

ATOM embedded sessions require a newly prepared artifact/restart for changed
weights. To use actual partial ATOM weight updates, explicitly select
`"engine": "atom-inputs"`, replace `artifact` with `cache_dir`, and keep the
same checkpoint/variant fields. `DeviceWeightDecode` prepares every shape in
the serving process before readiness (at most eight); initial compilation can
be expensive on rebel 0.11 and cross-process device-input caches are unsafe on
the tested stack. Subsequent requests use those prepared shapes. This opt-in
path retains the slower device-input decode rate documented in
[delta-swap.md](delta-swap.md); it is not the default ATOM choice.

## Native warmup and standalone KV

Set `"warmup": true` in any prepared config to execute one actual native step
for every configured variant before the worker becomes ready. The default is
false; only JSON booleans are accepted. Warmup uses token/position zero and
private blank KV, discarded after every attempt. A preparation/warmup error or
cancellation fails startup and attempts to close the initialized engine. The
original startup exception is preserved if cleanup also fails, with the cleanup
failure as its cause. A failed RNGD close synchronization disables further
serving while retaining potentially live native buffers; retry close only after
synchronization recovers, or recover through process teardown. RNGD still loads
only existing cached graphs; ATOM device-input startup retains its existing
materialization even when optional warmup is disabled. Warmup is kernel execution,
not prompt prefill or evidence of token quality.

The cluster's whole-request KV path uses
`rngd_pd.connector.worker.WorkerKVConnector` and `WorkerRole.WORKER` without
requiring vLLM. The optional `rngd_pd.connector.rngd_kv.RngdKVConnector` plugin
retains the genuine vLLM base class and shares the same transfer implementation;
importing that plugin without vLLM raises an actionable error. No vLLM shim is
installed by production code.

To replay real prefill after its producer exits, publish each layer's full-width
arrays through the selected shared-memory transport with the worker connector's
`save_kv_layer` and `wait_for_save`. Preserve the real prefill's `next` token,
prompt length and immutable checkpoint ID. POST `/decode` with `req_id`, `first`
(that `next` token), `pos` (prompt length), `max_new` and `checkpoint_id` after
the consumer is ready. K per layer is `(nkv, hd, wire_S)` and V is
`(nkv, wire_S, hd)`; use the same `PD_SHM_ROOT` and bf16 serialization on both
sides. `tokens[0]` echoes the supplied `first`; compare the native-produced
suffix using `max_new >= 2` against a fresh matching reference. Closing the
shared-memory producer preserves its published files until the consumer loads
them. This replay does not establish a live gateway/prefill measurement.

## Admission, transport and upgrade limits

- `PD_MAX_REQUESTS=32` bounds concurrent POST handlers, including streaming and
  synchronous B=1 workers, and each batcher's admitted/queued requests. Excess
  requests return 429; the gateway preserves upstream refusal status.
- `PD_REQUEST_TIMEOUT=600` bounds batcher request lifetime. Cancellation keeps
  ownership until an in-flight producer/step returns. Stream disconnects cancel
  their generation. An unresponsive native SDK call still requires process-level
  recovery; Python cannot safely interrupt a DMA or native kernel.
- `PD_HTTP_MAX_BODY_BYTES=1048576` and `PD_HTTP_BODY_TIMEOUT=15` bound POST
  bodies, including chunked frames, by total size and absolute read deadline.
  Oversized bodies return 413; partial bodies time out with 408 and release
  admission. Invalid/ambiguous framing closes the connection.
- `PD_KV_LOAD_AHEAD=2` bounds pending background caches, rather than only loader
  threads. Legacy speculative prefetch has a separate budget capped by this,
  `PD_PREFETCH_CAP` and `PD_MAX_REQUESTS`. Expired running loads retain their slots;
  only completed expired entries are reclaimed. Prepared serving has no extra
  speculative cache budget.
- KV uses validated numeric `.npy` (float16 or bf16 bits), not pickle. **Upgrade
  all communicating peers together**; old pickle payloads are rejected.
- TCP defaults: 256 MiB/blob, 1 GiB total payload including transfers, 4096-byte
  keys, 65,536 entries, 128 active connections, 60 s absolute frame timeout.
  `AIRM_KV_TCP_MAX_BLOB_BYTES`, `MAX_STORE_BYTES`, `MAX_KEY_BYTES`,
  `FRAME_TIMEOUT_S` and `MAX_WAIT_MS` use the same `AIRM_KV_TCP_` prefix.
  Replacement needs temporary capacity for both payloads. Pulls have an actual
  operation deadline of requested wait plus 100 ms reply allowance. Increase
  limits consistently at all peers when using larger compiled contexts.
- UCX only recycles a destination after observed `DONE` and a source after its
  READ completion notification. Timeout, `ERR`, unknown states and ambiguous
  publication acknowledgements retain ownership. Wall TTL cannot prove DMA has
  stopped. Abandoned spans can exhaust the finite arena; graceful close refuses
  outstanding transfers and process teardown is needed until SDK drain/cancel
  semantics are verified. This favors failure over cross-request memory reuse.
- The vLLM-shaped connector supports this project's whole-request layer arrays.
  Paged block/scheduler mappings are explicitly rejected pending a real layout
  adapter. CPU import-boundary stubs do not establish real vLLM compatibility.

Helm can mount these files via the existing `hostMounts` and set
`decode.env.PD_PREPARED_CONFIG`; set `prefill.env.PD_CHECKPOINT_ID` to its ID.
The chart continues to run the same `rngd_pd.serving.cluster` module.

## Verification and hardware gate

```bash
python -m pip install -e .
python -m pytest -q tests
```

CPU tests execute real worker batching, imported KV layouts and swap engines
with only SDK/device boundaries replaced. They cover one retained ATOM session,
resident RNGD weights across B/S switches, and a real-tensor ATOM partial update.
Transport tests use actual loopback sockets, malformed frames, executable pickle
rejection, full admission and stalled/disconnected peers. CI runs the complete
CPU suite and checks the chart relay matches its source.

Before deployment on either device, run this worker with matching prefill and
record: token equality with a fresh destination load; cold and prepared TTFT;
`swap_ms` separately from KV import; unchanged device allocations; stable HBM
across repeated B/S switches; mixed context/output lengths; overload and stream
disconnects. Repeat on the site's SDK versions. Real UCX error/DMA recovery,
container builds and Kubernetes rollout have not been validated by these CPU
tests. Existing [hardware results](../experiments/2026-10-23-npu-delta-load.md)
cover the underlying loaders, not a new end-to-end serving measurement.
