# RNGD × llm-d 배포 (Phase 5)

> 아래 내용은 소스가 `src/rngd_pd`로 옮겨지기 전의 역사적 blueprint입니다.
> 현재 배포 경로는 [Helm chart](helm/rngd-pd)와
> `python -m rngd_pd.serving.cluster`입니다. 아래 Dockerfile·compose·이미지 빌드
> 예제의 옛 진입점은 현재 소스와 호환되지 않습니다.
> 새 기능의 설정과 검증 범위는 [준비된 변형 서빙](../docs/prepared-serving.md)을 참고하세요.

PoC → 프로덕션 배포 하드닝의 산출물. **정직 고지**: 이 저장소에는 실제 K8s 클러스터가 없어 매니페스트는
**구조 유효성(yaml lint)까지 검증**했고 클러스터 실배포는 미실행. 로직/서빙 경로는 `serving_app.py
--role all --mock`로 실제 HTTP 검증됨. 컨테이너 빌드는 Furiosa wheel 인덱스 접근이 있는 환경에서 수행.

## 구성요소
| 파일 | 역할 |
|------|------|
| `Dockerfile.gateway` | 게이트웨이(무-NPU, stdlib) 이미지 |
| `Dockerfile.worker` | 워커(furiosa 스택 + 우리 런타임) 이미지. RNGD 드라이버/디바이스는 **호스트에서 마운트** |
| `docker-compose.yaml` | 단일 호스트: gateway + prefill@chip0 + decode@chip1, KV는 `/dev/shm` 공유 |
| `k8s/gateway.yaml` | 게이트웨이 Deployment/Service + Prometheus ServiceMonitor(`/metrics`) |
| `k8s/workers.yaml` | prefill/decode 풀(xPyD), 각 pod가 `furiosa.ai/npu:1` 요청, 공유 아티팩트 PVC |

## 로컬 브링업
```bash
FURIOSA_INDEX_URL=<wheel-index> docker compose -f scripts/deploy/docker-compose.yaml up
curl -s localhost:8080/v1/completions -d '{"tokens":[791,6864,315,9822,374],"mode":"pd"}'
curl -s localhost:8080/metrics
```

## KV transport 선택 (D2D=host PCIe, RDMA 없음)
| 배치 | transport | 근거 |
|------|-----------|------|
| 같은 노드 두 칩 | **ShmTransport** (`/dev/shm`) | tmpfs, 네트워크 홉 없음 — PoC 실측 경로 |
| 노드 간 | **TcpTransport** | `kv_connector.py` TCP, 교차프로세스 bit-exact 검증됨 |
| (미래) RDMA/UCX | **NixlTransport** | 아래 결정 참조 |

### NIXL 결정 (Phase 5 확정)
- **현재**: RNGD은 RDMA/NVLink 없음(D2D=host PCIe). NIXL 미설치. → **shm/TCP를 정식 transport로 확정.**
- **인터페이스**: `kv_transport.NixlTransport`는 스텁으로 유지 — 동일 `Transport` 계약, 미설치 시 **loud-fail**
  (silent fallback 금지). `handshake_metadata`/`set_remote_handshake` 훅은 vLLM NIXL agent 교환과
  동형으로 노출만.
- **승격 조건**: NIXL이 RNGD host-staged 전송을 지원하게 되면 `NixlTransport` 실장만 하면 되고
  커넥터/게이트웨이 코드는 불변(주입식 transport). → 배포에서 `backend=nixl`로 스위치.

## 관측성
게이트웨이 `/metrics` (Prometheus text): `rngd_kv_hit_rate`, `rngd_throughput_tokens_per_s`,
`rngd_pd_requests_total`, `rngd_pd_transfer_bytes_total`, `rngd_kv_hit_blocks_total`,
`rngd_compute_ms_total`. ServiceMonitor가 15s 간격 스크레이프.

## 콜드스타트 (EDF 아티팩트 캐시)
워커는 `RNGD_ARTIFACT_CACHE`(PVC `rngd-artifacts`)에서 컴파일된 EDF를 로드 → ~30분 재컴파일 회피.
구현/검증: `scripts/serving/artifact_cache.py` (아래 Phase 5 task #16 결과 참조).

## PD vs replica 모드
게이트웨이 `--mode {pd,replica}`. 실측(2칩, 짧은 단일요청): replica 5698ms < pd 6343ms.
PD 이득은 무거운 prefill·대배치·독립 스케일(xPyD)에서. 워크로드에 맞춰 선택.

## 남은 실클러스터 배선 (blueprint, 미실행)
- 게이트웨이→워커 HTTP 프록시 디스패치(현재는 in-process dispatch로 로직 검증). 워커가 `/register`로
  등록하고 게이트웨이가 선택된 워커 엔드포인트로 요청 포워딩하는 부분.
- Furiosa device plugin(`furiosa.ai/npu`) 설치 + 실제 디바이스 노드 경로 확정.
- (선택) 진짜 llm-d Inference Gateway(Envoy ext-proc)에 우리 EPP 로직 이식.
