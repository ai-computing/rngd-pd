# RNGD × llm-d 통합 — 개요 · 사용법 · 호환성

Furiosa RNGD NPU에서 **PD disaggregation + KV 관리를 vLLM/llm-d 규격으로** 구현하고, 그 앞에 **KV-aware
게이트웨이**를 둔 스택. 통짜(1 wired EDF) Llama/Qwen 런타임 위에서 동작하며, 전 구성요소가 실측 검증됨.

> 상태: **5개 Phase 전부 완료(PoC 하드닝 레벨).** 로직·성능·직렬화는 실 NPU/실 HTTP로 검증.
> 실제 K8s 클러스터 배포와 Envoy IGW 이식은 blueprint(미실행) — [§6 스코프 경계](#6-스코프-경계-정직-고지) 참조.
> 상세 계획·acceptance 체크리스트: [`llmd-integration-plan.md`](./llmd-integration-plan.md) · 배포 상세: [`../../scripts/deploy/README.md`](../../scripts/deploy/README.md)

---

## 1. 이게 뭔가 (한 문단)

Furiosa `furiosa-llm`은 봉인된 엔진(닫힌 `KVCacheManager`)이라 vLLM worker로 못 쓴다. 그래서 **우리 통짜
런타임이 곧 엔진**이 되고, 그 위에 (a) 진짜 `KVConnectorBase_V1`을 상속한 KV 커넥터, (b) llm-d EPP식
KV-aware 라우팅 게이트웨이, (c) resident-KV·멀티버킷·continuous batching·아티팩트 캐시 같은 서빙 하드닝을
얹었다. KV 전송은 RNGD에 RDMA가 없으므로(D2D=host PCIe) **shm/TCP**를 정식 transport로 쓰고, NIXL은
동일 인터페이스의 스텁으로 열어둔다.

---

## 2. 아키텍처 (현재 구성)

```
                 ┌─────────────────────────────────────────────┐
   HTTP 요청 ───▶│  Gateway (llm-d IGW 최소구현)                │
 /v1/completions │  EPP KV-aware 라우팅 · registry · /metrics   │
                 └───────────────┬─────────────────────────────┘
                        route(mode = pd | replica)
             ┌───────────────────┴───────────────────┐
             ▼                                        ▼
    ┌──────────────────┐   KV (KVConnectorBase_V1)  ┌──────────────────┐
    │ Worker: prefill  │ ── save_kv_layer ────────▶ │ Worker: decode   │
    │  rngd:0          │   shm(/dev/shm) | TCP      │  rngd:1          │
    │  통짜 wired EDF  │ ◀─ wait_for_layer_load ─── │  통짜 wired EDF  │
    └──────────────────┘                            └──────────────────┘
       resident-KV(16.3ms/tok)                        multi-bucket · batching
       artifact_cache(콜드스타트 제거)
```

### 레이어 ↔ 파일 맵 (`scripts/` 기준)

| 레이어 | 파일 | 역할 |
|--------|------|------|
| **런타임(엔진)** | `llama_engine_resident.py` | 통짜 32L Llama, resident-KV, 16.3ms/tok, 9/9 |
| | `engine_wired_generic.py` | **config 구동** 통짜 런타임(Llama/Qwen2 계열 단일 코드) |
| | `dsl_gen.py` | dim→L1 DSL(rms/zip/swiglu/softmax) 생성 |
| **KV 커넥터** | `rngd_kv_connector.py` | 진짜 `KVConnectorBase_V1` **서브클래스** |
| | `kv_transport.py` | `Transport` ABC + InProc/Shm/Tcp/**Nixl(스텁)** |
| **게이트웨이** | `gateway/registry.py` | 워커 레지스트리(role/load/heartbeat) |
| | `gateway/kv_events.py` | block-hash→worker 인덱스(prefix-cache) |
| | `gateway/epp.py` | Endpoint Picker(KV-hit 우선, PD/replica) |
| | `gateway/server.py` | HTTP `/v1/completions` |
| | `gateway/demo_e2e.py` | **실 2칩** PD/replica e2e 데모 |
| **서빙 하드닝** | `serving/bucket.py` | 멀티버킷 런타임 선택 |
| | `serving/batching.py` | continuous batching 스케줄러 |
| | `serving/serving_app.py` | 통합 서버(게이트웨이+버킷+배칭+/metrics) |
| | `serving/artifact_cache.py` | EDF 직렬화 저장/로드(콜드스타트 제거) |
| **배포** | `deploy/Dockerfile.{gateway,worker}`, `deploy/docker-compose.yaml`, `deploy/k8s/*.yaml` | 컨테이너·K8s·NIXL 결정 |
| **검증** | `run_selftests.sh` | NPU-free 회귀 12/12 |
| | `probe_resident_kv.py`, `probe_artifact_cache.py` | NPU 실측 프로브 |

---

## 3. 사용법

### 3.1 빠른 검증 (NPU 불필요)
```bash
cd scripts
./run_selftests.sh          # 12/12 GREEN: 커넥터·transport·게이트웨이·DSL·서빙
```

### 3.2 통합 서버 스모크 (NPU 불필요, mock 런타임)
```bash
python3 serving/serving_app.py --role all --mock
# → KV-hit 라우팅, PD 라우팅, /metrics(KV-hit rate·throughput·PD bytes) 실 HTTP로 확인
```

### 3.3 실 NPU — 통짜 런타임 (정확도 + 성능)
```bash
python3 llama_engine_resident.py --model meta-llama/Llama-3.1-8B-Instruct --n 4
#   → argmax 9/9 PASS | 16.3 ms/token   (최초 1회 ~30분 컴파일)
python3 engine_wired_generic.py  --model Qwen/Qwen2.5-0.5B-Instruct     --n 4
#   → Qwen2.5-0.5B: 9/9 PASS | 6.4 ms/token  (bias+tied),  Llama-3.1-8B: 9/9 | 14.6 ms  — 단일 코드
```

### 3.4 실 NPU — PD disaggregation (2칩, 커넥터 경유)
```bash
python3 llama_pd_llmd.py --model meta-llama/Llama-3.1-8B-Instruct --n 4 --split 3 --transport shm
#   → prefill@rngd:0 --save_kv_layer--> shm --wait_for_layer_load--> decode@rngd:1, 출력==monolithic
python3 gateway/demo_e2e.py --model meta-llama/Llama-3.1-8B-Instruct --n 4 --split 3
#   → 게이트웨이 EPP 라우팅으로 PD/replica 둘 다 실행, ==mono, A/B 지연 측정
```

### 3.5 콜드스타트 제거 (아티팩트 캐시)
```bash
python3 probe_artifact_cache.py     # compile→save→(fresh proc)load, 출력 해시 동일, load≈0s
```
런타임 사용:
```python
from serving.artifact_cache import get_or_compile, cache_key
key = cache_key(model=model_id, H=cfg.hidden, bucket=S, tactic="Decode")
m, mode, secs = get_or_compile("/artifacts", key, compile_fn=lambda: build_runtime(), device="rngd:0")
# 최초: mode="compile"(~30분) → 디스크 저장. 이후: mode="load"(수초, 재컴파일 없음)
```

### 3.6 컨테이너 / K8s (blueprint)
```bash
FURIOSA_INDEX_URL=<wheel-index> docker compose -f deploy/docker-compose.yaml up   # gateway+2워커
kubectl apply -f deploy/k8s/         # gateway(Deploy/Svc/ServiceMonitor) + prefill/decode 풀(xPyD)
```

### API (OpenAI 유사)
```bash
curl -s localhost:8080/v1/completions \
  -d '{"tokens":[791,6864,315,9822,374],"mode":"pd","max_new":4}'
# {"choices":[{"tokens":[12366,13,578,6864,315]}], "routing":{...}, "timing_ms":{...}}
# 12366 = " Paris"   →   "The capital of France is Paris."
curl -s localhost:8080/metrics       # Prometheus text
```

---

## 4. llm-d / vLLM 호환성 매트릭스

| 규격 항목 | 상태 | 근거 / 한계 |
|-----------|------|-------------|
| `vllm…KVConnectorBase_V1` 상속 | ✅ **완전** | `isinstance` True, abstract 7개 구현, `RngdKVConnectorMetadata(KVConnectorMetadata)` |
| scheduler API (`get_num_new_matched_tokens`·`update_state_after_alloc`·`build_connector_meta`) | ✅ 구현 | 시그니처 vLLM 0.18.0 정렬 |
| worker API (`register_kv_caches`·`start_load_kv`·`wait_for_layer_load`·`save_kv_layer`·`wait_for_save`) | ✅ 구현 | 레이어별 save/load 오버랩 |
| PD disaggregation (xPyD) | ✅ **실동작** | 실 2칩, KV 이송, 출력==monolithic |
| KV transport | ✅ shm/TCP | 교차프로세스 bit-exact. **NIXL은 스텁(loud-fail)** — RNGD RDMA 없음 |
| KV-cache-aware 라우팅 (EPP) | ✅ 기능 동등 | chained block-hash prefix 인덱스, KV-hit 우선 |
| KV cache events (`take_events` 계열) | 🟡 부분 | 인덱스 동형이나 vLLM 이벤트 스트림 정식 연결 아님 |
| OpenAI `/v1/completions` | ✅ 최소 | tokens/prompt·mode·max_new |
| 실제 vLLM 엔진이 이 커넥터를 로드·구동 | ❌ | **furiosa-llm이 vLLM 백엔드 아님**(vllm_compat=prompt 포맷뿐) → 우리 런타임이 엔진 |
| 실제 llm-d K8s 배포 / Envoy IGW | ❌ blueprint | 매니페스트 YAML 유효까지. 클러스터 실배포 미실행 |
| NIXL 정식 통합 | ❌ 스텁 | 인터페이스·handshake 훅만. RNGD host-staged 지원 시 실장 |

**요약**: 커넥터는 **진짜 상속 = 코드 계약 완전 호환**. PD·KV 관리·KV-aware 라우팅은 **실동작**. 다만 실행
주체가 vLLM 엔진이 아니라 **우리 런타임**이고, 실 llm-d 클러스터/Envoy/NIXL은 다음 단계(설계·문서화 완료).

### llm-d 관점 성숙도 (레벨)
- **L1 프로토콜 호환** ✅ — 커넥터가 실제 vLLM 계약을 상속.
- **L2 기능 동작** ✅ — PD·KV 이송·KV-aware 라우팅이 실 NPU에서 정답.
- **L3 배포 통합** ⏳ — 컨테이너/K8s 매니페스트 준비, 클러스터 실배포·Envoy·NIXL 미실행(blueprint).

---

## 5. 성능 (실측)

| 런타임 | ms/token | 정확도 | 비고 |
|--------|----------|--------|------|
| host-attn fast | 189 | — | 초기 |
| per-layer wired | 626 | 9/9 | 레이어별 host 왕복 |
| fullwire(정확본) | 48.2 | 9/9 | per-token KV 재업로드 |
| **resident-KV** | **16.3** | **9/9** | KV device 상주 (2.96× ↑) |
| Llama-3.1-8B (generic) | 14.6 | 9/9 | **config 구동** 단일 코드 |
| Qwen2.5-0.5B (generic) | 6.4 | 9/9 | **config 구동** 단일 코드 (bias+tied) |
| perf 상한(랜덤가중치) | 11.86 | — | 순수 처리량 |

- **PD vs replica** (짧은 단일요청): replica 5698ms < pd 6343ms. PD 이득은 무거운 prefill·대배치·xPyD.
- **배치 처리량**(기존 실측): B=128 → 28134 tok/s.
- **콜드스타트**: compile ~30분 → 아티팩트 load 수초.

---

## 6. 스코프 경계 (정직 고지)

**검증됨**: 커넥터 상속·bit-exact 전송, 2칩 PD==mono, EPP 라우팅, resident-KV 16.3ms, config 범용성
(Qwen 9/9), 아티팩트 재로딩, 배칭/버킷 로직, NPU-free 회귀 12/12, 매니페스트 YAML 유효.

**미실행(blueprint — `scripts/deploy/README.md`)**: 게이트웨이→워커 HTTP 프록시 디스패치(현재 in-process
로직 검증), Furiosa device-plugin 실설치, 진짜 Envoy 기반 llm-d IGW ext-proc 이식, NIXL 실장.

**스코프 밖**: furiosa 봉인 엔진의 vLLM native worker화(불가), MoE/MLA 아키텍처(dense 계열까지).
