# FuriosaAI 기술 요청서 — PD Disaggregation을 위한 KV 훅 및 컴파일러 이슈

작성: 2026-08-13 · 환경: RNGD ×4 (fw 2026.3.0), furiosa-llm 2026.3.0, furiosa-torch 2026.2.0
· 모델: Llama-3.1-8B BF16 · 전체 실측 데이터/재현 코드 보유

## 요약

Prefill/Decode 분리 서빙(PD disaggregation, llm-d/DistServe 계열)을 RNGD에서 실측한
결과, **엔진에 KV export/import 훅 하나가 열리면 2-NPU 처리량이 최대 +31%p 개선**됨을
확인했다 (재프리필 −35% → KV 전송 시 +31% vs 복제 구성). 하드웨어·컴파일러 내부는
이미 준비되어 있고, 소프트웨어 표면만 닫혀 있다. 아울러 검증 과정에서 발견한
컴파일러 무음 오답 버그 1건과 디바이스 이슈 1건을 함께 보고한다.

---

## 요청 1 — 엔진 KV export/import 훅 (최우선)

**현황**: furiosa-llm의 `KVCacheManager`가 봉인되어 있어 prefill 인스턴스가 만든 KV를
decode 인스턴스로 옮길 방법이 없다. CLI/Python API/네이티브 바이너리 전수 조사에서
`kv_transfer`/`kv_connector`/disaggregation 심볼 0건.

**실측 근거** (2×RNGD, Llama-8B, 유니크 토큰-ID 프롬프트):

| 구성 | 처리량 (in=2048, N=16) | ITL p99 | 비고 |
|---|---|---|---|
| 복제(DP=2) | 407 tok/s | 87–187 ms | 기준 |
| PD + 재프리필 (현재 가능한 유일한 방식) | 263 tok/s (−35%) | 170–226 ms | decode 카드가 프롬프트 재계산 |
| **PD + KV 전송 (prefix-cache 에뮬레이션 실측)** | **534 tok/s (+31%)** | **48–66 ms** | 전송 무비용 가정의 상한 |

- 전송 비용 자체는 이미 해결됨을 자체 런타임으로 실증: vLLM `KVConnectorBase_V1` 규격
  커넥터로 **실제 KV 이송 PASS** (출력 == monolithic), 청크 스트리밍 시
  **handoff 25 ms** (KV 34 MB, 프리필 뒤로 은닉), 요청 비용의 ~2%.
- 하드웨어 경로 확인: `p2p_accessible(npu↔npu)=True`, `/dev/rngd/npu*p2pmem` 존재.
  P2P DMA 시 ~5 ms 예상 (256 MB @ ~50 GB/s).
- 컴파일러 내부에도 존재: `OperatorTacticDramToDramBridge`,
  `interchip_tdma_throughput_per_chip_dtod`, `allow_inter_chip_dram`
  (멀티카드 -pp 아티팩트가 사용 중) — eager 표면만 미배선
  (`CopyBetween.NpuToNpu` enum과 전용 `SimpleCopy` 레이아웃이 이미 예약되어 있음).

**요청 형태 (둘 중 하나면 충분)**:
- (a) 서빙 엔진 레벨: 요청 단위 KV 블록 export(prefill 완료 시)/import(decode 시작 전)
  API — vLLM KVConnector 계약과 호환되면 llm-d 편입이 즉시 가능
- (b) furiosa-torch eager 레벨: `NpuToNpu` copy 배선 (`memory.copy`의 Dram→Dram 케이스
  — 현재 "scatter is not supported"로 미구현)

## 요청 2 — 배치 그래프 컴파일 범위 (또는 요청 1로 대체 가능)

자체 배치 resident 디코드 커널(lockstep B스트림)을 구현·검증했으나
(hidden 오차 bf16 플로어, 스텝 시간 B-불변 실증: 1.78 ms @ B=1/2/4 → per-token 3.98×@B=4),
**실전 규모의 배치 그래프가 tactic 탐색 memory explosion으로 컴파일 불가**:

| 구성 (B × nl × S) | 결과 |
|---|---|
| B≤4 · nl=2 · S=64 | ✅ |
| B∈{2,4,8} · nl∈{4,8} · S=256 | ✗ memory explosion |
| B=4 · nl=8 · S=128 / B∈{2,4} · nl=8 · S=64 | ✗ |
| (참고) B=1 · nl=32 · S=512 | ✅ / S=576·1024 ✗ |

요청 1이 해결되면 봉인 엔진의 decode bucket(batch≤256)을 그대로 쓰면 되므로 본 요청은
불필요해진다. 요청 1이 어렵다면 eager 배치 그래프의 탐색 한계 완화를 검토 바란다.

## 요청 3 — 디바이스 플러그인이 `npuNp2pmem`을 주입하지 않음 (k8s 배포 차단)

**현황**: `furiosa.ai/rngd: 1`을 요청한 파드는 카드의 디바이스 노드를 거의 전부 받지만
(`bar*`, `ch*`, `dmar`, `mgmt`, `pe0`~`pe7`, `pe0-1`, `pe0-3`, `pe4-5`, `pe4-7` — 35개)
**`npuNp2pmem` 하나만 빠진다** (호스트 36개 vs 파드 35개, 차집합이 정확히 이것뿐).

이게 8-PE 워크로드를 막는다. RNGD의 융합 디바이스 노드는 4-PE가 최대이고
(`pe0-3`, `pe4-7`; `pe0-7`은 존재하지 않음), 카드 전체를 쓰려면
`set_fusion(8)` + `MeshKind.Mesh(2)`로 두 4-PE 그룹을 메시로 묶어야 하는데
그 그룹 간 경로에 p2p 메모리가 필요하다. 없으면 8-PE 디바이스가 **열거되지 않아**
`device index out of range` → `Failed to get allocator`로 죽는다.

**대조 실험** (동일 파드, Llama-3.1-8B resident wired 엔진):

| 조건 | `get_available_devices()` | 할당 |
|---|---|---|
| 파드 + 기본 fusion(1) | `rngd:0`~`rngd:7` (카드 1장의 PE 8개) | ✅ 성공 |
| 파드 + fusion=8/Mesh(2), 플러그인 디바이스만 | (열거 전 패닉) | ❌ index out of range |
| 파드 + fusion=8/Mesh(2), `/dev/rngd` 전체 마운트 | `rngd:0`~`rngd:3` | ✅ 성공 |

파드 안에서 `furiosa-smi info`는 처음부터 정상 동작하며 카드와 PCI-BDF를 정확히 보고한다.
즉 유저스페이스·드라이버·플러그인 모두 정상이고, **디바이스 집합에 p2pmem이 빠진 것만이 원인**이다.

**현재 우회**: hostPath로 `/dev/rngd` 전체를 마운트 + `privileged: true`.
후자가 필요한 이유는 kubelet의 device cgroup이 플러그인이 할당한 노드만 허용하기 때문이라,
노드를 보이게 해도 특권 없이는 접근이 차단된다. 프로덕션에 쓰기엔 과한 권한이다.

**요청**: 디바이스 플러그인(또는 CDI 스펙 생성기)이 카드 할당 시 `npuNp2pmem`을 함께
포함해 주면, `privileged` 없이 표준 리소스 요청만으로 8-PE 융합 워크로드가 파드에서 동작한다.
참고로 `furiosa-llm serve`(봉인 엔진)는 현재도 파드에서 정상이므로, 영향 범위는
furiosa-torch eager 경로로 카드 전체를 쓰는 워크로드다.

(부가) 공식 k8s 예제의 `securityContext.capabilities.drop: [ALL]`은 사전 빌드 이미지를
전제한다. 기동 시 `apt-get`을 쓰는 파드에서는 apt가 seteuid/setgroups로 권한을 낮추지 못해
실패하므로, 문서에 한 줄 주석이 있으면 도움이 될 것 같다.

## 버그 리포트 1 — 3D 활성 텐서 슬라이스의 무음 오답 (심각)

`CompileModule` 그래프에서 3D 활성 텐서의 마지막 차원 슬라이스가 **에러 없이 잘못된
값**을 산출한다. 재현: RoPE 구현 `xr[..., :half]`, `xr[..., half:]` (xr = [B, heads, 128])
→ 컴파일·실행은 성공하나 **K 텐서가 기대값 대비 상대오차 0.72** (RoPE가 항등인 pos=0에서도).
동일 수학을 회전행렬 matmul로 재구성하면 정확 (오차 0.007 = bf16 플로어).
2D 슬라이스(순차 엔진의 동일 패턴)는 정상이므로 배치 차원 추가 시의 lowering 문제로 보임.
관련: 활성 슬라이스가 f32와 결합되면 DSL 커널 검증 실패("verification of operator failed",
zipadd에 f32 유입)로는 잡히나, 위 케이스는 **무음**이라 위험하다.

## 버그 리포트 2 — 특정 칩의 EDF 로더 실패

동일 EDF가 npu0/1/2에서는 정상 로드·실행되나 **npu3에서만**
`Edf.load_weight_and_binaries` 시 `Npu(npu:3:4-7, Buf(Os { code: -22 }))` (EINVAL).
furiosa-llm 서빙(봉인 로더)은 npu3에서 정상이므로 furiosa-torch EDF 로더 경로 한정.
초소형 모듈로도 재현되어 크기 무관. (재부팅 후 재확인 예정.)

## 재현 자료

- 코드: `rngd_llmd/` 패키지 (KV 커넥터·배치 커널·벤치 하니스) + `furiosa-bench/pd-disagg/`
- 데이터: 스윕 JSONL 전부 (TTFT/TPOT/tok-s/전력/PE 카운터 포함)
- 리포트(그림 포함): 별도 공유 가능
- 컴파일 캐시: 32L S=256/512 EDF — 재현 시 재컴파일 불필요
