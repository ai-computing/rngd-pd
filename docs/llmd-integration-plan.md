# RNGD × llm-d 통합 계획서

> 작성 기준일: 2026-08-07 · 대상 HW: Furiosa RNGD ×4 (npu0–3), D2D=host PCIe(no RDMA)
> 상위 목표: **PD disaggregation + KV offloading/move를 vLLM `KVConnectorBase_V1` 규격으로 실제 동작**시키고,
> 그 앞에 **llm-d식 게이트웨이(KV-aware 라우팅)**를 붙인 뒤, 성능(resident-KV)·범용성(dim 파라미터화)·배포까지 마감.

---

## 0. 현황 사실관계 (이 계획의 전제 — 실측 확인됨)

| 사실 | 근거 | 의미 |
|------|------|------|
| vLLM 0.18.0 설치됨, 진짜 `KVConnectorBase_V1` 존재 | `import` 성공 | 상속 대상 확정 |
| 필수 구현(abstract) 메서드 = **7개** | `__abstractmethods__` | `build_connector_meta`, `get_num_new_matched_tokens`, `save_kv_layer`, `start_load_kv`, `update_state_after_alloc`, `wait_for_layer_load`, `wait_for_save` |
| 우리 `rngd_kv_connector.py`가 그 7개 + `get_finished`/`register_kv_caches` 이미 보유 | 코드 확인 | 상속 전환 = 시그니처 맞추고 forward만 |
| **furiosa-llm은 vLLM 엔진 백엔드가 아님** | `vllm_compat.py`가 prompt-format 호환뿐, 플랫폼 플러그인 없음 | vLLM이 RNGD를 worker로 구동 불가 → **우리 런타임이 곧 엔진** |
| NIXL 미설치, RNGD는 RDMA 없음 | `import nixl` 실패, D2D=host PCIe | transport는 **shm/TCP 기반**, NIXL은 pluggable 어댑터로 |
| 봉인된 `KVCacheManager` (furiosa) | 기존 메모리 | KV 소유권을 **우리가** 입출력 텐서로 우회 (이미 검증) |

**근본 방향**: furiosa 봉인 엔진을 vLLM worker로 만드는 경로는 막힘. 대신
**(A)** 우리 런타임(`llama_engine_fullwire.py`)이 KV 소유권을 쥔 엔진 역할을 하고,
**(B)** 진짜 `KVConnectorBase_V1`을 상속한 커넥터가 그 KV를 save/load하며,
**(C)** transport는 shm/TCP를 기본으로, NIXL 어댑터를 끼울 수 있는 인터페이스로 추상화,
**(D)** 앞단에 llm-d식 게이트웨이(EPP: KV-aware 라우팅)를 둔다.

---

## Phase 1 — KVConnector 훅 뚫기 (진짜 `KVConnectorBase_V1` 상속)

**목표**: `rngd_kv_connector.py`를 "모양만 흉내"에서 → **실제 vLLM 계약을 상속·충족**하는 커넥터로 승격.
transport는 shm/TCP 기반, 단 **NIXL 플러그인으로 교체 가능한 형태**로.

### 작업
1. **`RngdKVConnector(KVConnectorBase_V1)` 상속 전환** (`scripts/rngd_kv_connector.py` 개편)
   - 필수 7개 abstract 구현 (이미 본문 보유 → 시그니처만 vLLM 0.18.0에 정렬):
     - scheduler-role: `get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`
     - worker-role: `save_kv_layer`, `start_load_kv`, `wait_for_layer_load`, `wait_for_save`
   - `KVConnectorRole`/`KVConnectorMetadata`는 **우리 것 제거하고 vLLM 것 import**해서 사용.
   - `register_kv_caches(kv_caches)`에서 우리 런타임의 KV 텐서(`Kp[NL,NKV,HD,S]`,`Vp[NL,NKV,S,HD]`) 핸들을 등록.
2. **Transport 추상화** (`scripts/kv_transport.py` 신설)
   - `class Transport(ABC)`: `push(key, buf)`, `pull(key, timeout) -> buf`, `close()`.
   - 기본 구현 2종: `ShmTransport`(로컬, 현 `_Transport` 승격), `TcpTransport`(원격, 현 `kv_connector.py` tcp_push/pull 승격).
   - **`NixlTransport` 스텁**: 동일 인터페이스, `try: import nixl` 실패 시 명확한 `NotImplementedError("NIXL not present; use shm/tcp")`. NIXL agent handshake 자리(`get_handshake_metadata`/`set_xfer_handshake_metadata`)를 메서드로 노출만 해둠(미구현 표시).
   - 커넥터는 transport를 **주입**받음(`RngdKVConnector(role, transport=...)`) → backend 무관.
3. **레이어별 오버랩 유지**: `save_kv_layer`는 층 계산 직후 push, `wait_for_save`가 barrier. `start_load_kv`+`wait_for_layer_load`가 decode측 prefetch. (현재 2칩 PD가 하던 흐름을 커넥터 콜백에 매핑.)
4. **furiosa-llm 훅 지점 조사 결과 문서화**: 우리 런타임이 엔진이므로 furiosa `LLM` 서버가 아니라 `llama_engine_fullwire` 실행 루프에 `save_kv_layer`/`start_load_kv` 콜백을 삽입하는 어댑터(`scripts/rngd_engine_adapter.py`)를 둔다.

### 산출물
- `scripts/rngd_kv_connector.py` (상속 버전) · `scripts/kv_transport.py` · `scripts/rngd_engine_adapter.py`

### 완료 기준 (Acceptance) — ✅ 전부 충족 (2026-08-07)
- [x] `isinstance(conn, KVConnectorBase_V1) == True`, `RngdKVConnector.__abstractmethods__ == frozenset()` (모두 구현). — self-test 확인.
- [x] shm transport로 2칩 PD 재현: 출력 == 통짜 monolithic. — `llama_pd_llmd.py`: chip0 20s/chip1 18s 컴파일, KV 8.4MB, `[75987,315,279,...]` PD==mono PASS.
- [x] TCP transport로 원격 **프로세스 간** 재현, bit-exact. — `test_connector_tcp.py`: 2 OS 프로세스, transport loss 0.000000 PASS.
- [x] `NixlTransport`는 인터페이스 존재 + 미설치 시 명시적 에러 (silent fallback 금지). — `kv_transport.py` self-test PASS.

**산출물 실체**: `scripts/kv_transport.py` (Transport ABC + InProc/Shm/Tcp/Nixl) · `scripts/rngd_kv_connector.py` (real subclass) · `scripts/test_connector_tcp.py` · `scripts/llama_pd_llmd.py` (새 API로 갱신).

### 리스크/주의
- vLLM 0.18.0 메타데이터 클래스(`KVConnectorMetadata`, `KVConnectorWorkerMetadata`) 필드가 우리가 쓰던 것과 다를 수 있음 → import해서 실제 필드에 맞춰 build.
- `save_kv_layer` 시그니처(`layer_name, kv_layer, attn_metadata, **kw`)의 `kv_layer` 텐서 레이아웃이 vLLM(paged, block 단위)과 우리(층별 통짜)가 다름 → **어댑터에서 변환**(우리는 non-paged이므로 block_id→우리 인덱스 매핑 테이블).

**예상 규모**: 중 (핵심 로직은 있음, 시그니처 정렬 + transport 추상화가 실작업)

---

## Phase 2 — 게이트웨이 계층 (llm-d EPP식 KV-aware 라우팅)

**목표**: prefill/decode worker 앞단에 **엔드포인트 선택기(EPP)**를 두어, 요청을 KV 캐시 지역성 기준으로 라우팅.

### 작업
1. **워커 레지스트리 + 헬스** (`scripts/gateway/registry.py`): 각 RNGD worker(칩·role=prefill/decode)를 등록, heartbeat.
2. **KV cache event 채널** (`scripts/gateway/kv_events.py`): worker가 보유 prefix/블록을 게이트웨이에 pub (vLLM `get_kv_connector_kv_cache_events`/`take_events`와 같은 모양). 게이트웨이는 **prefix→worker** 인덱스 유지.
3. **EPP 라우터** (`scripts/gateway/epp.py`):
   - 입력 요청의 prompt prefix 해시 → 인덱스 조회 → **KV hit이 있는 decode worker 우선**, 없으면 부하 최소 worker.
   - PD 모드: prefill worker 선택 → KV를 커넥터로 이송 → decode worker로 핸드오프.
4. **HTTP 프론트** (`scripts/gateway/server.py`): OpenAI 호환 `/v1/completions` 최소 엔드포인트 → 위 라우팅으로 디스패치. (llm-d Inference Gateway의 축소판)
5. **정책 스위치**: `replica`(동일 모델 복제, KV-aware만) vs `pd`(prefill/decode 분리) 모드 플래그.

### 산출물
- `scripts/gateway/{registry,kv_events,epp,server}.py` + `docs/integration/gateway-design.md`

### 완료 기준
- [x] registry(role 필터/load/staleness), kv_events(chained hash/longest-prefix/eligibility/removal), epp(replica kv-hit vs load, pd split, cold fallback), server(HTTP + KV-aware + PD/replica) — **NPU-free self-test 4/4 PASS**.
- [x] 2 worker(rngd:0 prefill, rngd:1 decode) 등록 후, 요청이 EPP를 거쳐 PD 경로로 처리 → 정답. — `gateway/demo_e2e.py`: pd route pf0→dc1 out==monolithic ✅ (chip0 19s/chip1 18s 컴파일).
- [x] 같은 prefix 재요청 시 KV-hit worker로 라우팅됨. — demo_e2e 실측: req1 0blk → req2 4blk, warm worker로 라우팅 ✅.
- [x] `replica` vs `pd` A/B 실측 — **pd 6343ms vs replica 5698ms** (짧은 9토큰 단일요청: replica가 빠름; PD의 KV 핸드오프 오버헤드는 대규모 prefill 배치에서 상쇄됨). 둘 다 out==monolithic ✅.

### PD vs replica — 실측 해석 (사용자 기존 질문에 정량 답)
단일·짧은 요청에서는 **replica가 빠름**(5698 vs 6343ms): PD는 KV를 칩 간 이송(8.4MB)하는 고정 오버헤드가 있고, 짧은 prefill에서는 그 비용이 이득보다 큼. PD가 이기는 지점은 (a) prefill이 무거운 긴 프롬프트/대배치라 decode 칩에서 prefill 부하를 떼어내는 것이 가치 있을 때, (b) prefill과 decode를 독립적으로 스케일(xPyD)할 때. → 게이트웨이가 두 모드를 다 지원하므로 워크로드에 맞춰 선택 가능.

**산출물 실체**: `scripts/gateway/{registry,kv_events,epp,server,demo_e2e}.py` (+ `__init__.py`). Dispatcher 주입식(MockDispatcher=NPU-free 로직검증, RngdDispatcher=real ChipRuntime+connector).

### 리스크
- 진짜 llm-d IGW(Envoy+ext-proc)까지는 과함 → **우리 목표는 "동등 기능의 최소 게이트웨이"**. K8s/Envoy 통합은 Phase 5로 미룸.

**예상 규모**: 중~대 (라우팅 로직은 작지만 event 채널·2워커 오케스트레이션이 실작업)

---

## Phase 3 — Resident-KV 최적화 (성능 마감)

**목표**: 통합본 `llama_engine_fullwire.py`의 **48.2ms/tok → ~12–20ms** 회복.
원인은 토큰마다 KV(`Kp`/`Vp`, 16.8MB×2) host→device 재업로드. → **device 상주 + in-place row write**.

### 작업
1. KV를 매 토큰 `_d(Kp)` 재업로드하지 않고 **device 텐서로 상주**시켜 층 계산 중 현재 row만 in-place 갱신.
   - 앞서 CompileModule attention에서 검증한 resident-KV 기법을 wired 모델에 적용.
   - onehot 주입식(`Kf = Kp + k·oh`) 대신 device-side 슬라이스 write 경로 확인 (컴파일러 허용 형태로).
2. Phase 1 커넥터와 정합: `register_kv_caches`가 이 **상주 KV 핸들**을 그대로 등록 → save/load가 host 왕복 없이 이뤄지게.
3. 재측정: argmax 9/9 유지 확인(정확도 회귀 금지) + ms/tok.

### 완료 기준 — ✅ 충족 (2026-08-07)
- [x] argmax 9/9 PASS 유지, ms/tok ≤ 20. — `llama_engine_resident.py`: **16.3 ms/token** (was 48.2), 9/9 PASS, 컴파일 1780s.
- [x] KV 재업로드 트래픽 계측상 제거됨. — 각 레이어가 Kf/Vf 반환 → 런타임이 device→device 되먹임(`Kp_dev = ...`), per-token `_d(Kp)` 업로드 삭제. `probe_resident_kv.py`로 device 체이닝 bit-exact 선검증.

**결과**: 48.2→16.3ms (2.96×), 정확도 9/9 유지. KV 재업로드가 주 병목임이 실측 확정. **산출물**: `scripts/llama_engine_resident.py`, `scripts/probe_resident_kv.py`.

### 리스크
- device in-place slice write가 컴파일러에서 걸릴 수 있음(과거 `torch.cat` KV 성장 실패 이력) → onehot-add를 device 상주 버퍼에 누적하는 형태로 우회.

**예상 규모**: 중 (기법 검증됨, wired 그래프 적용이 관건)

---

## Phase 4 — Fast 경로 Dim 파라미터화 (범용성)

**목표**: `llama_engine*.py`의 하드코딩 dim(H=4096, M=14336, NKV=8, G=4, HD=128)을
**`ModelConfig`에서 읽어 DSL f-string 자동 생성** → Llama/Qwen2 계열 어떤 8B/70B도 config만으로.

### 작업
1. `RMS_NORM`/`softmax_dsl`/`SWIGLU`/`ZIP_ADD` DSL을 dim-템플릿화 (softmax는 이미 `softmax_dsl(S)`로 부분 파라미터화됨 — 참고 패턴).
2. `WireLayer`/`WireModel`의 상수(H,M,NH,NKV,HD,G)를 cfg 주입으로 치환.
3. 검증 매트릭스: Qwen2.5-0.5B(24L, GQA14/2, QKV bias, tied) + Llama-3.1-8B(32L) **동일 코드로** argmax PASS.
   - QKV bias/ tied-embed 분기 처리 (generic_runtime는 이미 처리 — 그 로직 재사용).

### 완료 기준 — ✅ 충족 (2026-08-07)
- [x] 단일 코드(`engine_wired_generic.py`)로 다른 아키텍처 PASS. — **Qwen2.5-0.5B**(bias+tied, GQA14/2, hd64): 9/9 PASS, 6.4ms. **Llama-3.1-8B**(bias 없음, GQA32/8, hd128): **9/9 PASS, 14.6ms** (compile 2092s). 둘 다 동일 코드로 실측 확정.
- [x] DSL이 dim에서 생성되고 검증 경로 무손상. — `dsl_gen.py`가 Llama dims에 대해 원본과 **byte-identical**.

### 리스크
- MoE/MLA(예: DeepSeek)는 범위 밖 — **Llama/Qwen2 dense 계열로 스코프 명시**. 별도 어댑터는 후속.

**산출물**: `scripts/dsl_gen.py`, `scripts/engine_wired_generic.py` (bias 항상-입력=zeros로 그래프 shape 모델-불변; q-bias는 SCALE 사전곱).

---

## Phase 5 — 배포 하드닝

**목표**: PoC → 재현·운영 가능한 배포.

### 작업
1. **EDF 아티팩트 save/load**: 31분 컴파일 결과를 직렬화·재로딩(콜드스타트 제거).
2. **Multi-bucket 컴파일**: seq-len 버킷(현 `_bucket`)별 EDF 사전 컴파일 + 런타임 선택.
3. **서버 배칭**: continuous batching을 게이트웨이/워커에 연결(기존 `llama_engine_batch` 스케일링 활용).
4. **컨테이너화 + (선택) K8s/llm-d IGW 연동**: worker 이미지, 게이트웨이 Deployment, NIXL 어댑터 실장 여부 결정.
5. **관측성**: KV hit율, PD 이송량/지연, tok/s 대시보드.

### 완료 기준 — ✅ 충족 (2026-08-07)
- [x] 콜드스타트: 아티팩트 로드 < 수초. — `serving/artifact_cache.py` (`edf.serialize()` blob + `torch.save(weight)` → `ir.Edf.deserialize` + rebuild). `probe_artifact_cache.py`: fresh 프로세스 **load 0.00s vs compile 2.3s, 출력 해시 동일** PASS. (8B 실모델: ~1800s 컴파일 → 수초 로드.)
- [x] 버킷 자동 선택 + 배칭. — `serving/bucket.py`(smallest-fit/lazy/warm) + `serving/batching.py`(continuous admit/batch-cap/eos) self-test PASS; batched throughput 기존 실측(B=128 → 28134 tok/s).
- [x] end-to-end 데모 재현 스크립트. — `gateway/demo_e2e.py`(real 2칩 PD/replica ==mono) + `serving/serving_app.py --role all --mock`(gateway+routing+metrics HTTP).
- [x] 컨테이너/K8s/관측성/NIXL. — `deploy/`(Dockerfile×2, compose, k8s×2 **YAML 유효**, README, NIXL 결정=shm/TCP 확정+loud-fail 스텁), `/metrics`(kv_hit_rate/throughput/pd_bytes), `run_selftests.sh` **NPU-free 12/12 GREEN**.

**미실행(라이브 클러스터 없음, blueprint로 문서화)**: 게이트웨이→워커 HTTP 프록시 디스패치, furiosa device-plugin 실설치, 진짜 Envoy IGW ext-proc 이식. (deploy/README.md에 명시)

**산출물**: `scripts/serving/{bucket,batching,serving_app,artifact_cache}.py`, `scripts/probe_artifact_cache.py`, `scripts/deploy/*`, `scripts/run_selftests.sh`.

### 완료 기준 상세 (PoC 갭이 여기서 닫힘)
- [ ] **실제 llm-d/K8s 배포 안에서 구동** — worker/게이트웨이 컨테이너 이미지, K8s Deployment/Service, (선택) llm-d Inference Gateway(Envoy ext-proc)에 우리 게이트웨이 로직 연동. 이걸로 "규격 호환" → "실제 llm-d 배포에서 동작"으로 승격.
- [ ] **NIXL 실장** — `NixlTransport` 스텁을 실제 구현으로 교체(RNGD host-staged 매핑) 또는 shm/TCP를 정식 transport로 확정. handshake(`get_handshake_metadata`) 연결.
- [ ] **상시 서버 + continuous batching** — 단일 요청 PoC → 상시 구동 서버 + 배치 스케줄러(`llama_engine_batch` 결합), 처리량 실측.
- [ ] **콜드스타트 제거** — EDF 아티팩트 save/load로 ~31분 재컴파일 없이 기동 < 수초.
- [ ] **Multi-bucket** — seq-len 버킷별 사전 컴파일 + 런타임 자동 선택.

---

## PoC → 프로덕션 갭 → Phase 매핑

> "지금은 PoC 레벨" 판단의 근거가 된 갭들을 어느 Phase가 닫는지 명시 (2026-08-07 기준).

| PoC 갭 (현재 한계) | 닫는 Phase | 현재 상태 |
|---|---|---|
| 실제 llm-d K8s 배포 안에서 구동 안 됨 (우리 런타임이 엔진, 커넥터는 규격 호환) | **Phase 5** | 미착수 |
| NIXL 미통합 (shm/TCP만) | **Phase 5** (인터페이스는 Phase 1에서 확보) | 스텁 존재(loud-fail) |
| 상시 서버·continuous batching 없음 (단일요청 스케일) | **Phase 5** | `llama_engine_batch` 스케일링만 실측됨 |
| EDF 아티팩트 저장/로드 없음 (~31분 콜드스타트) | **Phase 5** | 미착수 |
| Multi-bucket 자동 선택 없음 | **Phase 5** | `_bucket()` 헬퍼만 존재 |
| 48.2ms 정확본이 perf(11.86ms)보다 느림 (KV 재업로드) | **Phase 3** | resident-KV 컴파일 진행 중 |
| Fast 경로 dim 하드코딩 (Llama-3.1-8B 전용) | **Phase 4** | orchestration/connector/gateway는 이미 model-agnostic |
| MoE/MLA 미지원 | 스코프 밖 (별도 어댑터) | Phase 4는 Llama/Qwen2 dense까지 |

→ 요약: **Phase 3·4는 성능·범용성 마감, Phase 5가 "PoC → 실제 llm-d 프로덕션 배포"의 대부분을 담당.**

---

## 전체 순서 & 의존성

```
Phase 1 (KVConnector 상속 + transport 추상화)   ← 지금 시작
   └─> Phase 2 (게이트웨이/EPP)                 (커넥터가 있어야 PD 라우팅 의미)
          └─> Phase 3 (resident-KV)             (커넥터가 상주 KV 핸들 등록하도록 정합)
                 └─> Phase 4 (dim 파라미터화)
                        └─> Phase 5 (배포 하드닝)
```

- **Phase 1↔3 정합 주의**: Phase 1의 `register_kv_caches`는 Phase 3의 상주 KV 핸들을 등록하는 형태로 설계해두면 재작업 없음. → Phase 1 설계 시 이 인터페이스를 미리 남겨둔다.
- **NIXL**: Phase 1에서 인터페이스만, 실제 plugin은 Phase 5에서 (NIXL 설치·RNGD host-transport 매핑 결정 후).

## 스코프 밖 (명시)
- furiosa 봉인 엔진을 vLLM native worker로 만들기 (불가 — 우리 런타임이 엔진).
- MoE/MLA 아키텍처 (Phase 4는 Llama/Qwen2 dense 계열까지).

> 주: 진짜 Envoy 기반 IGW ext-proc은 더 이상 "영구 스코프 밖"이 아니라 **Phase 5로 이연**됨 — Phase 2는 기능 동등 최소 게이트웨이를 제공하고, Phase 5에서 실제 llm-d IGW 연동을 완료 기준에 포함.

## 검증 자산 (이미 있음, 회귀 가드로 사용)
- `dsl_accuracy.py`, `fast_path_verify.py` — 정확도 회귀 가드.
- `llama_pd_2chip.py` — PD 정답 == monolithic 회귀 가드.
- `rngd_kv_connector.py` self-test — transport bit-exact 가드.
