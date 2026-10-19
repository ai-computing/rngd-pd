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

## ai-rm

**계층/역할**: `l3-serving` 서비스. ai-rm(AI 반도체 통합 자원관리)에서 Llama-3.1-8B를 Furiosa RNGD prefill과
Rebellions ATOM decode로 나눠 서빙하는 워크로드를 제공한다. 구성 요약은 [`ai-rm.yaml`](ai-rm.yaml).

**k8s + llm-d 연동**: llm-d 경로(Envoy → InferencePool → EPP → router → prefill/decode Pod)로 동작한다.
ai-rm 배포용 chart는 `deploy/helm/rngd-pd/`(prefill·decode·gateway, `airm.kvTransport` 값,
`airm.ai-computing.io/furiosa.RNGD`·`airm.ai-computing.io/rebellions.ATOM` 레이블로 노드 선택)이고,
사이트 값(hostPath, 이미지, 모델 아티팩트 경로)은 ai-rm 쪽 override로 넣는다. 실험용 sweep 매니페스트는
계속 `deploy/k8s/gen_sweep.py`가 생성하고 `kubectl apply -f`로 적용한다. 현재 한계:
- Pod는 `ubuntu:22.04`에 이 저장소와 호스트 SDK·드라이버를 hostPath로 마운트한다. 독립 컨테이너 이미지는 없다.
- KV는 같은 노드의 `/dev/shm`으로 넘어가며 단일 노드(`nodeSelector`로 고정)에서만 검증했다.
- 그래프는 호스트에서 미리 컴파일해 `artifacts_cache/`를 채워야 한다. 디렉터리는 kustomize 형식이 아니다.

**설치/테스트** (NPU 불필요):

```bash
python3 -m pip install --user -e .    # rm-core와 entry point 등록. NPU SDK·vllm은 호스트의 것을 쓴다
export PYTHONPATH=$PWD/src
python -m rm_core.conformance airm.kv_transports shm
python -m pytest -q tests/native_v4 tests/test_airm.py
python -m rngd_pd.connector.transport
python tests/test_connector_tcp.py
helm template deploy/helm/rngd-pd
```

설치하지 않고 `src/`만 PYTHONPATH에 둔 경우(Pod)에도 `shm`은 `rngd_pd.airm`에서 직접 찾는다. 다른 이름은
그 이름의 entry point가 설치돼 있어야 한다.

### ai-rm protocol v0.1 adapter (`rngd_pd/airm.py`)

| entry point | 이름 | 객체 |
|---|---|---|
| `airm.kv_transports` | `shm` | `rngd_pd.airm:ShmTransport` |
| `airm.workloads` | `rngd-pd` | `rngd_pd.airm:run` |

**`shm` = host staging 경로다.** KV는 prefill 카드에서 host로 내려와 `/dev/shm` 파일(bf16 wire)로 넘어가고
decode 쪽에서 다시 카드로 올라간다. 장치 간 P2P DMA가 아니다(그것은 hetero-npu-p2p의 `p2p`).
`ShmTransport(root=None)`: `root`는 공유 디렉터리, 기본 `$PD_SHM_ROOT`, 없으면 `/dev/shm/rngd_kv_cluster`.
prefill·decode Pod가 같은 노드의 `/dev/shm`을 마운트해야 한다. host-dram → host-dram(같은 머신)만 지원하고,
dtype은 `bf16`/`fp16`, 양쪽 layout이 같아야 한다. 실패를 다른 경로로 대체하지 않는다.
`conformance_case()`는 같은 호스트의 shm tier 둘을 쓰며, 서빙 디렉터리 대신 옆의 전용 디렉터리에 쓴다.

**KV 경로 선택**: connector(`rngd_pd/connector/rngd_kv.py`)와 서빙 워커는
`rngd_pd.connector.transport.select_transport()`로 경로를 고른다. 이름은 `AIRM_KV_TRANSPORT`(chart 값
`airm.kvTransport`)이고 기본값은 지금 동작과 같은 `shm`이다. 이름은
`rm_core.registry.load("airm.kv_transports", name)`으로 불러온다. connector의 push/pull을 직접 지원하는
구현(`shm`)은 그대로 쓰고, tier 사이 복사만 하는 KVTransport(`p2p` 등)는 `FencedTransport`로 감싼다.
이 경우 push는 plan+submit, pull은 완료 fence(`wait`)가 `ok`일 때만 반환하며 handle이 프로세스 안에 있으므로
prefill과 decode가 한 프로세스(hetero-npu-p2p coordinator)에 있어야 한다(Pod가 나뉘면 pull이 timeout된다).
connector가 쓰는 바이트는 프로세스 안에 남고, 감싼 transport는 자기 tier 사이 복사를 fence할 뿐이다.
모르는 이름은 `KeyError`다. 구현 클래스는 프로세스당 한 번만 찾는다. rm-core가 import되지 않는 환경에서는
`shm`만 같은 `/dev/shm` 경로로 동작하고 다른 이름은 `KeyError`다.
`RngdKVConnector`에 transport를 주입하지 않았을 때 `AIRM_KV_TRANSPORT`(또는 `kv_transport=`,
vLLM `kv_connector_extra_config["kv_transport"]`)가 있으면 그 경로를, 없으면 예전처럼 in-process를 쓴다.

**workload runner** `run(workload, **config)`: `Workload.kind`가 `prefill`이면 RNGD prefill 워커,
`decode`면 decode 워커(기본 ATOM), `serve`면 gateway를 `python -m rngd_pd.serving.cluster`로 띄우고
`Popen`을 돌려준다. 설정 키:

| 키 | 기본 | 용도 |
|---|---|---|
| `kv_transport` | `shm` | `airm.kv_transports` 이름, `AIRM_KV_TRANSPORT`로 전달 |
| `port` | `8000` | 워커/게이트웨이 HTTP 포트 |
| `seq_len` | `256` | 그래프를 컴파일한 S |
| `backend` | prefill `rngd-full`, decode `atom` | `cluster.py --backend` |
| `prefill_url`, `decode_url` | 로컬 8101/8102 | gateway가 부를 워커 |
| `metrics_port` | 없음 | gateway의 `airm_*` 포트(`AIRM_METRICS_PORT`) |
| `args`, `env` | 없음 | 추가 `cluster.py` 인자, 추가 환경 변수 |
| `dry_run` | `False` | 실행하지 않고 `(argv, env)` 반환 |

`Workload.model`은 `MODEL`, decode의 `Workload.batch_hint`는 `--B`가 된다.

**지표** (rm-core 이름·단위): gateway에 `AIRM_METRICS_PORT`가 있으면 그 포트의 `/metrics`에 최근 512요청의
`airm_ttft_p90_ms`, `airm_tpot_p90_ms`, `airm_tok_s`, `airm_kv_transfer_ms{kv_transport=...}`를 낸다
(chart 기본 9400, `airm.ai-computing.io/scrape: "true"`). `bench/kpi_slo.py` 결과 행에도 같은 이름이 들어간다.
`kv_transfer_ms`는 요청별 `kv_save_ms + kv_load_ms`의 중앙값이다(prefetch가 켜져 있으면 load는 prefill과
겹친다). `power_w`는 아직 `airm_`로 내지 않는다. `bench/kpi.py`의 `power_w`는 구성이 쓴 카드들의 합이라
`scope=card|node` 어느 쪽에도 맞지 않는다.

`tests/run_selftests.sh`는 예전 디렉터리 구조를 가리키므로 현재는 사용하지 않는다.

**인터페이스**: provides `Workload`(PD 서빙 경로, runner `rngd-pd`), `KVTransport`(`shm`; vLLM
`KVConnectorBase_V1` 구현은 shm/TCP transport를 주입받음), `Metrics`(gateway `airm_*`, 워커 JSON `/metrics`).
다른 ai-rm 구성요소는 import하지 않고 KV 경로 이름으로만 고른다.

**지원 KPI**: MFU·전력효율(`bench/kpi.py`, `bench/sweep_llmd.py`의 보드 전력 샘플링), 이종자원수(RNGD+ATOM
조합), SLO(`bench/kpi_slo.py`: p99 TPOT 150 ms 달성 비율).

**`cluster.local.env` 변수** (`cluster.example.env`를 복사해 채우고 커밋하지 않는다. 셸에 export한 값이 우선):

| 변수 | 필수 | 용도 |
|---|---|---|
| `LLMD_GATEWAY_URL` | 예 (`PD_GATEWAY`가 없을 때) | llm-d 게이트웨이 기본 URL, 예 `http://<gateway-host>:<port>` |
| `PD_GATEWAY` | 아니오 | completions 전체 URL. 설정하면 `LLMD_GATEWAY_URL`보다 우선 |
| `KUBECTL_CMD` | 아니오 | kubectl 실행 방식 (기본 `kubectl`), 예 `ssh -p <ssh-port> <user>@<node-ip> kubectl` |

`bench/kpi_slo.py`, `bench/load_gen.py`, `bench/sweep_llmd.py`는 게이트웨이 변수가 없으면 즉시 오류로 종료한다.
