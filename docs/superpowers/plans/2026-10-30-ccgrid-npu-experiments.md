# CCGrid NPU experiments E6–E9 implementation plan

> Execute inline; one exclusive hardware owner, an independent read-only method
> review, and a disjoint E7 harness task. Use real run records, not estimates.

**Goal:** measure matched per-level reconfiguration, resident updates and three
directed B/S switches with n>=5 and traceable raw evidence.

**Spec:** user-supplied paper experiment board E6–E9 and GUIDELINE §4/§8.
Base: runtime f5fa13377a3d967ba3d4022fb73ae59b1c87441c,
Onggi8f0b4bc103cc1608a7288bd14c59f67a292d3d74. Issues34–37; branch
`exp/34-ccgrid-npu`. Previous integration and raw records stay unchanged.

## Fixed design and constraints

- SDK2026.4.0/Torch2.10.0+cpu, original Llama-3.1-8B checkpoint, group2,
  fusion8/mesh2, intra-op4, host_workers2. Opaque vendor artifacts only.
- One measurement card total; exclusive owner uses `cdx-` Kubernetes Jobs with
  `furiosa.ai/rngd:1` requests+limits, TTL and active deadline. No plain Pods.
  Check free resources first; own UID-guarded cleanup and verify absence.
- Off-card CPU graph preparation has separate provenance/budget. Never mix old
  SDK caches; all measured runtime paths forbid compilation.
- E6 scenario B4/S256→B4/S512, pending native graph/reference gates. E8 adds
  B8/S256, with at least three actual directed shape changes. No fake row-update
  fixture may replace a B/S experiment. E7 is a separate checkpoint mutation.
- E6 bundle contains the identical six checkpoint files plus graph/config
  artifacts for each shape. Base runtime uses the immutable verified checkpoint;
  every arm also gets an independent consumable base artifact to protect live
  mmap ownership from Onggi's move/claim operation. Setup counts in Pod time.
- All E6 arms first load/run the same base and drain at the same barrier.
  L1: full pull+engine subprocess restart. L1+L2: real `pull --base --base-dir`
  +engine subprocess restart. L1+L2+L3: same base-aware delivery, resident weights,
  target `swap()`. No weight rows change in E6; label avoided reload correctly.
- Record actual Pod creationTimestamp→first native-generated target token,
  and same-clock monotonic trigger→first token separately. Include scheduling,
  common setup, barrier waiting, drain, file delivery, process exit/start,
  constructor host/upload, graph/table preparation and native prompt TTFT.
  No hidden allocator retry/subtraction. Wall clock synchronization is recorded.
- Native `generate()` feeds the same real prompts on each arm. This measures a
  native decode benchmark, not HTTP/live gateway. Exact target tokens must match
  independent fresh references. Provided-prefill echo tokens are not TTFT.
- n=5 independent resets per E6 arm and application node-chunk-cache state
  (empty vs prepopulated):30 trials, blocked/interleaved arm order. Target graph
  first-use at each trial is fixed. OS page cache is uncontrolled/shared and
  never labeled globally cold. Failures remain in raw data, not silently retried.
- E7 compares inplace rows with existing `group` whole stacked tensor replacement
  and actual process restart on the same sparse fixtures. Logical uploaded bytes
  are not physical PCIe counters. E8 repeated shape directions n>=5.
- Median/IQR use inclusive linear quartiles. Aggregate valid completed trials
  only, preserve all invalid/failed attempts and declare missing n/failed validity.
- Only root edits Codex board rows/Log. Never edit NUMBERS.md or Claude rows.
  Issue→exp branch→PR→read every check green before merging experiment work.

## Tasks

- [x] Read board rules, log plan/ETA before work, create issues34–37 and branch.
- [x] Independent read-only method review: distinguish prepare/full pull from
  base-aware delta; protect base ownership; do not omit initial resident load.
- [x] Freeze graph hashes, complete artifact manifests and fixed input/run config.
- [x] Implement/test pure analysis and E6 trial/worker harness; preserve failures.
- [x] Implement/test E7 single-sample coordinator and update byte/timing evidence.
- [x] E6 real allocated Job smoke, fresh-reference/shape/native-identity gates.
- [x] E6 n5 per arm/cache:30/30 valid, all owned Jobs/Pods absent after cleanup.
- [x] E8 n5:30/30 valid, three directions x first-use/prepared x five repeats.
- [x] E7 n5:4/4 smoke +30/30 main valid, six groups n5, all Jobs/Pods removed.
- [x] Optional E9 explicitly deferred; no new ATOM results claimed.
- [x] Independently verify raw files, derive summaries and publish PR evidence.
- Exact-head CI and integration merge outcome are recorded in the shared board
  Log after publication; main integration chain remains outside this merge.

## Review focus and ledger

- A missing target graph fails before matrix launch; no measured-path compilation.
- Filesystem claim never consumes the immutable checkpoint used by a live engine.
- Pod timestamp absence is a validity failure, never a process-launch substitute.
- Counter/token failures persist and invalidate the affected sample.
- Small smoke must pass before the30-trial matrix; resource/clock/input condition
  changes require a new recorded configuration and explicit explanation.
- Ruling: the detailed user execution request authorizes implementation and tests
  without another design menu. Cost if wrong: a method mismatch would be visible
  in this plan and the shared board before card allocation, and can be corrected.
- Read-only inventory: two RNGD cards appear unrequested, no old-owned workload;
  only B4/S256 group2 SDK264 cache exists. New graphs prepared off-card separately.
- Initial full CPU suite:439passed8skipped4subtests61.89s, correct existing venv.
  Windows launcher PID differs from interpreter PID: compare actual worker
  (pid,entry_perf_ns) to establish resident/restart identity, with PID reuse allowed
  for a genuinely new interpreter. The reproduced failure and fix are retained.
- One independent review: noCritical; Important E7 incremental failure evidence
  and summary graph-state separation. Root RED reproduced first-use+prepared
  merging to n10/median50.5 and E7 modes merging; GREEN groups both separately.
  E7's bounded retention/native-observer/safe.directory fix is pending before E7
  measurements. Review declines native/clocks/reference provenance; remain gates.
- E6 cleared for bounded native reference/cache smoke. New CPU-prepared graphs:
  B4/S512 and B8/S256(+LM), exact SDK264/group2; runtime compilation still forbidden.
  Publish attempt1 incorrectly labeled reused graph provenance; preserve it and
  use corrected-v2 immutable manifests only. No native trial used attempt1.
- Review fix completed: E7 writes phase outputs into its persistent result before
  later sync/cleanup, preserves primary errors plus separate close errors, and
  observes original native calls/device type+allocated index with a compile guard.
  Command-scoped Git safe.directory handles the root Job/owned-source boundary.
  Root full CPU verification after all fixes:450passed8skip4subtests63.24s.
- E8 CPU harness covers first-use/prepared with actual target warmup and return
  to base, retained shared buffers/modules, native output checks and failure data.
  Native B8 cross-shape golden gate failed (one lane at generated token4). Do not
  change that gate silently; E8 is blocked pending diagnosis. B4/S256 and B4/S512
  native references pass, so frozen530195d E6 smoke proceeds independently.
- Review-declined rulings: native ownership/clocks/fresh-reference provenance stay
  exclusive hardware-owner gates plus root raw verification; E7 matrix inputs
  must provide both base/target fresh tokens even though public CLI permits missing
  base tokens; normalized collection fields must be independently compared with
  original raw records. CPU tests alone authorize no performance claim.
- Superseding E8 ruling, explicitly requested by the paper session: correctness
  is exact token-ID equality with a fresh load of the same target B/S variant.
  B8-vs-B4 divergence is a cross-shape observation with its first differing token,
  not a failed same-target switch. Preserve the original failed gate without
  rewriting it. No dense-logit/activation equality is inferred from token IDs.
- Root directly owns remote operations after repeated worker capacity errors.
  Full and delta E6 smoke records passed independent local gates. The persistent
  controller exports/hash-verifies raw small files before scratch removal and
  runs one Job/card at a time; only three successful smoke arms unlock30 fresh
  trials (three arms x two node-cache conditions x five repetitions).
  Root-owned scratch permissions were repaired by a CPU-only ownership Job (no
  NPU requested) and future operational cleanup runs after the first-token clock.
  ADR-0010/PR76 binding and B2 work remain deferred until E6 completion.
- E6 completed30/30 valid main trials. All six groups have n5 and no excluded
  main samples; n1 smoke is separate. Empty-cache trigger medians: full66.807s,
  delta19.199s, resident0.532s. Pod medians119.637/72.435/53.993s retain initial
  setup costs. Prepopulated groups are separate. All target tokens equal fresh
  load, compile0, actual physical0, process/buffer/module and cleanup gates pass.
  Root rehashed57 preparation records,416 frozen Git blobs, every exported
  raw file and every normalized metric. Final summary/figures preserve limits.
- After E6, E8 native matrix uses frozenbbceb17, same-target fresh token IDs,
  three directions x first-use/prepared x n5. No original cross-shape record is
  deleted. Root retains exclusive hardware ownership. ADR-0010 contract review
  begins separately; no B2 or public-only serving claim is implied by E6/E8.

- E8 completed30/30 native samples, all six groups n5, published atdb57b1f.
  Root checked original63 export files, every core source Git blob, raw numeric
  values/tokens/identities, all canonical public Git hashes and CLI reproduction.
- ADR-0010 v1 consumer integrated atf0f08bf, local590pass8skip4subtests, exact-head
  CI41051893033 read596pass2skip4subtests. Actual PR76 CLI binary47bfaae6 and
  sparse/LM fixture file SHA256 are frozen before E7, which passed4smokes.
  B2-only remains blocked on artifact-space actual readback/retained journal and
  runtime-valid fixture coverage; existing internal SDK dependencies are flagged.
  onggi#79/PR80 addresses the neutral kit false-positive verification boundary,
  with no native B2 claim. E7 measurements keep this limitation explicit.

- E7 completed4/4 smoke and30/30 main, every group n5, all exact base/target
  native tokens and device checks pass. Root independently checked77 small
  exports,470 executed Git blobs, all raw numeric/plan/token fields and full CLI
  reproduction. Direct final API: no owned Jobs/Pods or active RNGD requester.
  Sparse/LM row update-to-first-token medians0.635/0.632s; replacement0.963/1.127s;
  restart19.948/20.359s. Logical bytes and native timing limits remain explicit.
