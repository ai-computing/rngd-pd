# rngd_pd — PD-disaggregated serving across two NPU vendors

Llama-3.1-8B prefilled on a Furiosa RNGD card and decoded on a Rebellions ATOM
card, with the KV cache crossing between them through the vLLM `KVConnector`
interface over `/dev/shm`. The whole path runs as llm-d pods on `npu-node-1`:
Envoy → InferencePool → EPP → router → two NPU pods.

Nothing here re-prefills on the decode side. That is the point.

## Layout

```
src/rngd_pd/          the package — imported, never sys.path-injected
  kernels/            compiled graphs: furiosa DSL + the torch modules on top
    dsl_b1.py           DSL for single-token decode
    dsl_batch.py        DSL parameterised by batch width
    prefill_full.py     kv-free whole-prompt prefill   (the fast one)
    prefill_chunk.py    chunked prefill against a live cache
    decode_batch.py     batched decode, B streams per step
    decode_resident.py  single-stream decode, weights resident
  engines/            those graphs with real weights and an artifact cache
    prefill_full.py  decode_batch.py  decode_resident.py  prefill_chunk.py
    model.py            checkpoint loading, config, rmsnorm
    artifact_cache.py   compile once, keyed by graph hash
  vendors/            the same two jobs on Rebellions hardware
    rbln_prefill.py  rbln_decode.py  rbln_decode_batch.py
  connector/          KV handoff
    rngd_kv.py          KVConnectorBase_V1 implementation
    transport.py        /dev/shm segments, bf16 wire
  serving/
    cluster.py          prefill / decode / router HTTP workers
    proxy.py
  paths.py            repo root, artifact cache, shm root — worked out once

bench/                measurement entry points (run directly)
deploy/k8s/           gen_sweep.py emits the deployment; apply the output
docs/                 findings and vendor correspondence
attic/                superseded variants, kept for reference only
tests/
```

The rule: `kernels/` knows about graphs, `engines/` knows about weights and
caching, `serving/` knows about HTTP and llm-d. Dependencies point downward.

## Running

Everything imports `rngd_pd`, so `src/` has to be on the path:

```bash
export PYTHONPATH=$PWD/src:$HOME/.local/lib/python3.10/site-packages
export VLLM_PLUGINS=
PY=~/venvs/furiosa/bin/python

# a kernel on its own, checked against CPU fp32
$PY -m rngd_pd.kernels.decode_batch --B 4 --nl 32 --group 2 --chip 2 --validate

# an engine with real weights
$PY -m rngd_pd.engines.prefill_full --chip 2 --C 256

# a serving worker (this is what the pods run)
$PY -m rngd_pd.serving.cluster --role prefill --backend rngd-full --chip 0
```

Bench scripts add `src/` themselves, so they run without the export:

```bash
python3 bench/sweep_llmd.py --out bench/sweep_llmd.jsonl
python3 bench/sweep_report.py bench/sweep_llmd.jsonl
```

## Deploying

```bash
python3 deploy/k8s/gen_sweep.py > deploy/k8s/sweep-matrix.yaml
kubectl apply -f deploy/k8s/sweep-matrix.yaml
```

Pods mount this directory at `/app` and run `python -m rngd_pd.serving.cluster`.
One card per pod, so deployments use `strategy: Recreate` — a rolling update
puts two pods on one card and the new one dies with `allocator != nullptr`.

Graphs must be compiled **on the host**, not in the pod: the container's C
compiler stage dies with a broken pipe. Warm the cache first, then start the
pod, which will load in seconds:

```bash
$PY -m rngd_pd.engines.decode_batch --B 4 --group 2 --chip 2
```

## Where the numbers are

`docs/` holds the write-ups; `bench/*.jsonl` holds the runs behind them, so any
figure can be traced to the sweep that produced it.
