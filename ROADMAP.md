# RNGD × llm-d — 남은 작업 (ROADMAP)

> 현재 상태: **5개 Phase 완료(PoC 하드닝), 로직·성능·직렬화 실측 검증.** L1 프로토콜 호환 ✅ · L2 기능 동작 ✅ · L3 배포 통합 ⏳.
> 아래는 **아직 안 된 것만** 정리. 완료된 내용은 [`docs/llmd-integration-plan.md`](docs/llmd-integration-plan.md) 참조.

우선순위: **A(실배포) → B(전송/규격) → C(성능측정) → D(범용성)**. 각 항목은 독립적으로 착수 가능.

---

## A. 실 배포 — L3 (PoC → 프로덕션)  ★ 가장 임팩트

| # | 작업 | 현재 | 완료 기준 |
|---|------|------|-----------|
| **A1** | 게이트웨이→워커 **HTTP 프록시 디스패치** | in-process dispatch로 로직만 검증 | 워커가 `/register`로 등록 → 게이트웨이가 선택된 워커 엔드포인트로 요청 포워딩 → 실 분산 e2e |
| **A2** | **K8s 클러스터 실배포** | 매니페스트 YAML 유효까지 (`deploy/k8s/`) | Furiosa device-plugin(`furiosa.ai/npu`) 설치 + 실 노드 경로 확정 + `kubectl apply` 후 요청 처리 |
| **A3** | **아티팩트 캐시 서빙 통합** | `serving/artifact_cache.py` 단위검증(load 0s) | 워커 기동 시 EDF 로드 → ~30분 재컴파일 없이 수초 콜드스타트 (실 서버 경로) |
| **A4** | **상시 서버 + continuous batching** | 스케줄러 로직만(`serving/batching.py`) | 실 런타임에 배칭 연결 → 상시 구동 서버로 처리량 실측 |

## B. 전송 / 규격 심화

| # | 작업 | 현재 | 완료 기준 |
|---|------|------|-----------|
| **B1** | **NIXL 실장** | `NixlTransport` loud-fail 스텁 + handshake 훅 | RNGD host-staged 전송 매핑 실장 → `backend=nixl`로 스위치 가능 |
| **B2** | **진짜 llm-d IGW(Envoy ext-proc)** | 기능 동등 최소 게이트웨이(`gateway/`) | 우리 EPP 로직을 Envoy ext-proc로 이식 → 표준 llm-d 게이트웨이에서 동작 |
| **B3** | **vLLM KV cache events 스트림** | 인덱스 동형(`gateway/kv_events.py`) | `take_events`/`get_kv_connector_kv_cache_events` 정식 연결 |

## C. 성능 / 측정 심화

| # | 작업 | 현재 | 완료 기준 |
|---|------|------|-----------|
| **C1** | **TTFT 정밀 측정** | `ms/step@B=P` 근사(`bench/serving_bench.py`) | 배치-프리필 커널(query-length=P, causal)로 실 프리필 지연 측정 |
| **C2** | **compute-bound 전환점** | 배치≤128까지 측정(memory-bound, 33% peak) | 배치를 ridge(~171 FLOP/byte) 너머로 밀어 compute-bound 전환 실측 |
| **C3** | **긴 컨텍스트 TPOT** | 128–2048 측정 | 4k–8k 컨텍스트 TPOT + KV 메모리 스케일 |
| **C4** | **실 2칩 dup vs PD 처리량** | 성분 큐잉 모델(`bench/pd_vs_dup.py`) | 스레드 동시성으로 2칩 동시 실측 (DRAM_SHARED OOM 회피: KV 버퍼 재사용) |

## D. 범용성 확장

| # | 작업 | 현재 | 완료 기준 |
|---|------|------|-----------|
| **D1** | resident-KV × generic 통합 | `llama_engine_resident`(Llama dim) + `engine_wired_generic`(config, resident 포함) | 둘을 하나로 합치고 generic 경로가 resident 성능(16.3ms) 냄을 확인 |
| **D2** | **70B config 스모크** | 8B(Llama)+0.5B(Qwen) 검증 | 70B config(레이어수/dim만) 컴파일 스모크 |
| **D3** | **MoE / MLA 어댑터** | dense Llama/Qwen2 계열까지 | MoE(전문가 라우팅)·MLA(latent KV) 어댑터 추가 |

---

## 스코프 밖 (영구)
- furiosa 봉인 엔진의 vLLM native worker화 — **불가** (vllm_compat=prompt 포맷뿐, 우리 런타임이 엔진).

## 착수 추천 순서
1. **A1**(HTTP 프록시) — 이거 하나면 "규격 호환" → "실 분산 동작"으로 승격, L3 진입.
2. **A3+A4**(아티팩트+배칭) — 실 서버 성능/콜드스타트 완성.
3. **C4**(실측 dup vs PD) — 의사결정 근거를 모델→실측으로.
4. 나머지는 필요에 따라.
