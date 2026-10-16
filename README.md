# rngd_pd — PD-disaggregated serving across two NPU vendors

Llama-3.1-8B prefilled on a Furiosa RNGD card and decoded on a Rebellions ATOM
card, with the KV cache crossing between them through the vLLM `KVConnector`
interface over `/dev/shm`. The whole path runs as llm-d pods on `etri-001`:
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

## ai-rm

**계층/역할**: `l3-serving` 서비스. ai-rm(AI 반도체 통합 자원관리)에서 Llama-3.1-8B를 Furiosa RNGD prefill과
Rebellions ATOM decode로 나눠 서빙하는 워크로드를 제공한다. 구성 요약은 [`ai-rm.yaml`](ai-rm.yaml).

**k8s + llm-d 연동**: llm-d 경로(Envoy → InferencePool → EPP → router → prefill/decode Pod)로 동작한다.
`deploy/k8s/gen_sweep.py`가 매니페스트를 생성하고 `kubectl apply -f`로 적용한다. 현재 한계:
- Pod는 `ubuntu:22.04`에 이 저장소와 호스트 SDK·드라이버를 hostPath로 마운트한다. 독립 컨테이너 이미지는 없다.
- KV는 같은 노드의 `/dev/shm`으로 넘어가며 단일 노드(`nodeSelector`로 고정)에서만 검증했다.
- 그래프는 호스트에서 미리 컴파일해 `artifacts_cache/`를 채워야 한다. 디렉터리는 kustomize 형식이 아니다.

**설치/테스트** (NPU 불필요):

```bash
export PYTHONPATH=$PWD/src            # 별도 설치 단계 없음
python -m pytest -q tests/native_v4
python -m rngd_pd.connector.transport
python tests/test_connector_tcp.py
```

`tests/run_selftests.sh`는 예전 디렉터리 구조를 가리키므로 현재는 사용하지 않는다.

**인터페이스**: provides `Workload`(PD 서빙 경로), `KVTransport`(vLLM `KVConnectorBase_V1` 구현,
shm/TCP transport), `Metrics`(워커 `/metrics`). 다른 ai-rm 구성요소를 사용하지 않는다.

**지원 KPI**: MFU·전력효율(`bench/kpi.py`, `bench/sweep_llmd.py`의 보드 전력 샘플링), 이종자원수(RNGD+ATOM
조합), SLO(`bench/kpi_slo.py`: p99 TPOT 150 ms 달성 비율).

**`cluster.local.env` 변수** (`cluster.example.env`를 복사해 채우고 커밋하지 않는다. 셸에 export한 값이 우선):

| 변수 | 필수 | 용도 |
|---|---|---|
| `LLMD_GATEWAY_URL` | 예 (`PD_GATEWAY`가 없을 때) | llm-d 게이트웨이 기본 URL, 예 `http://<gateway-host>:<port>` |
| `PD_GATEWAY` | 아니오 | completions 전체 URL. 설정하면 `LLMD_GATEWAY_URL`보다 우선 |
| `KUBECTL_CMD` | 아니오 | kubectl 실행 방식 (기본 `kubectl`), 예 `ssh -p <ssh-port> <user>@<node-ip> kubectl` |

`bench/kpi_slo.py`, `bench/load_gen.py`, `bench/sweep_llmd.py`는 게이트웨이 변수가 없으면 즉시 오류로 종료한다.
