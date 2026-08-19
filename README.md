# rngd_llmd — RNGD × llm-d 서빙 스택 (포터블 패키지)

Furiosa RNGD NPU에서 **PD disaggregation + KV 관리를 vLLM/llm-d 규격으로** 구현하고, 그 앞에
**KV-aware 게이트웨이**를 둔 자립형 패키지. 어디에 복사해도 실행되도록 절대경로 없이 구성됨.

- **개요·사용법·호환성**: [`docs/llmd-usage-and-compat.md`](docs/llmd-usage-and-compat.md)
- **완료된 5-Phase 계획**: [`docs/llmd-integration-plan.md`](docs/llmd-integration-plan.md)
- **남은 작업**: [`ROADMAP.md`](ROADMAP.md)

## 구조 (계층)

```
rngd_llmd/
├── runtime/     엔진 — 통짜 wired EDF + config 구동
│   ├── engine_wired_generic.py   config→DSL, resident-KV (Llama/Qwen 단일 코드)
│   ├── llama_engine_resident.py  resident-KV whole-model (Llama, 16.3 ms/tok)
│   ├── dsl_gen.py                dim→L1 DSL 생성
│   ├── llama_engine.py / _batch.py / llama_pd_2chip.py  (엔진/배치/2칩)
│   ├── generic_runtime*.py       config 구동 오라클/런타임
│   └── _tv_patch.py
├── connector/   KV 커넥터 + 전송
│   ├── rngd_kv_connector.py      진짜 vLLM KVConnectorBase_V1 서브클래스
│   ├── kv_transport.py           Transport ABC + InProc/Shm/Tcp/Nixl(스텁)
│   ├── kv_connector.py           TCP KV 서버 헬퍼
│   └── llama_pd_llmd.py          2칩 PD (커넥터 경유) 데모
├── gateway/     EPP KV-aware 라우팅 + HTTP
│   ├── registry.py kv_events.py epp.py server.py
│   └── demo_e2e.py               실 2칩 PD/replica e2e
├── serving/     서빙 하드닝
│   ├── bucket.py batching.py serving_app.py artifact_cache.py
├── bench/        벤치마크 (+ plots/)
│   ├── serving_bench.py  pd_crossover.py  pd_vs_dup.py  pd_crossover_plot.py
├── tests/        NPU-free 회귀 + NPU 프로브
│   ├── run_selftests.sh  test_connector_tcp.py  probe_*.py
├── deploy/       컨테이너 / K8s
├── docs/         문서
└── _paths.py     포터블 경로 부트스트랩 (각 subdir에도 복제)
```

## 포터블 동작 원리
각 스크립트는 하드코딩 절대경로 대신 **`import _paths`** 로 시작한다. `_paths.py`는 `__file__`에서
패키지 루트를 찾아 모든 하위 폴더(runtime/connector/gateway/serving/bench/tests)를 `sys.path`에
추가 → `from generic_runtime import …`, `from rngd_kv_connector import …` 같은 평면 import가
**패키지를 어디로 옮겨도** 해석된다. 실행은 항상 패키지 루트 기준:

```bash
cd rngd_llmd
```

## 빠른 시작

```bash
# 1) NPU 없이 전체 로직 회귀 (12/12 GREEN)
bash tests/run_selftests.sh

# 2) 통합 서버 스모크 (NPU 없이, mock 런타임) — 라우팅·KV-hit·PD·/metrics
python3 serving/serving_app.py --role all --mock

# 3) 실 NPU — 통짜 런타임 (정확도 + 성능)
python3 runtime/engine_wired_generic.py --model Qwen/Qwen2.5-0.5B-Instruct   # 9/9 · 6.4 ms/tok
python3 runtime/llama_engine_resident.py --model meta-llama/Llama-3.1-8B-Instruct  # 9/9 · 16.3 ms/tok

# 4) 실 NPU — 2칩 PD disaggregation (커넥터 경유)
python3 connector/llama_pd_llmd.py --split 3 --transport shm     # 출력 == monolithic

# 5) 벤치 (그래프 → bench/plots/)
python3 bench/serving_bench.py     # throughput/TTFT/TPOT/roofline
python3 bench/pd_vs_dup.py         # 2칩 duplication vs PD 처리량
```

## 이식 (다른 머신/경로)
```bash
cp -r rngd_llmd /any/where/ && cd /any/where/rngd_llmd && bash tests/run_selftests.sh
```
필요 런타임: `torch` + furiosa 스택(`furiosa-torch`/`furiosa-llm`/rngd 드라이버) + `transformers`,
`numpy`, `matplotlib`. 엔진 deps는 패키지에 포함(자립). NPU-free 테스트는 furiosa 없이도 통과.
