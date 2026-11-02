# Loaded native queue: design and execution plan (issue49)

Goal: measure what a checked B2 transition versus same-target reload does to
an actual in-flight native request and two queued requests. Prepared target
files and compiled graphs are shared with the earlier B2 experiment; delivery
is excluded. This is a finite explicit caller, not an HTTP serving system.

Approaches considered: a fixed synthetic lock hold can prove exclusion but
cannot measure real drain; an HTTP gateway adds unrelated software; a FIFO
caller over the unchanged native owner supplies the required actual request
boundary with the least new code. The latter is the implementation choice
within the authorized follow-up. No scheduler/runtime behavior is changed.

One worker executes base request0, target switch, then requests1/2. The main
thread queues the switch and target requests only after an original compiled
program in request0 has returned and its device output has been checked.
There are no sleeps, fake inference or artificial request holds in native
execution. All three requests use the existing four-lane P8/O8 fixture. A
trigger after request0 returns is invalid, not a short drain observation.

Record monotonic arrivals, worker start/return, native-first marker, switch
start/ready and reconstructed first-token time (generate entry plus returned
native prompt TTFT). Report drain wait, switch work, trigger-to-target token,
each queued request's wait/TTFT/completion and host RSS/PSS snapshots. Do not
infer device peak, HTTP latency, throughput or p99 from this finite case.

Correctness: request0 matches independent fresh base; requests1/2 match fresh
target, which differs from base. No request enters during switch. Current B2
includes plan, staging, actual undo/readback and synchronization; reload closes
base before target allocation. B2 preserves serving buffer/program identity.
All original compiled outputs are checked on the allocated chip, compilation
is forbidden, and each P8/O8 request has255 native calls/checks. Initial base
load/parity and final close are outside transition clocks. Retain source,
config, input/hash/reference, exact Job/Pod UID and cleanup witnesses.

Use one excluded smoke pair then five alternating pairs. A fresh owner begins
each arm. Only bounded cdx- Jobs with one RNGD request/limit, no retries,
active deadline and TTL may execute. Avoid CPU timing-matrix I/O overlap. Any
worker failure, invalid overlap, incorrect output/native count or unsuccessful
close stops the dependent matrix; preserve partial evidence. Do not release
or replace an owner while a worker is still executing; process exit is the
fallback if a bounded join fails. Original fixtures and historical evidence
remain read-only. Remote commands use hub.orca.run_remote exclusively.

Implementation tasks (executed inline):

- [x] Add `bench/ccgrid_loaded_queue.py`: `run_queue(generate, switch,
  first_native, emit, timeout=300)` records the actual FIFO experiment and
  rejects absent/late native triggers, worker errors and incomplete joins.
- [x] Add CPU tests in `tests/swap/test_ccgrid_loaded_queue.py` for real
  queue ordering, actual lease-free synthetic file computations, early worker
  failure, late trigger and no transition/request after a failed switch.
  Observe meaningful RED before implementation and GREEN afterward.
- [x] Connect `bench/ccgrid_loaded_native.py` to existing `ccgrid_b2` validation/references and E7
  native observer; add a first-native-return observer without altering original
  program outputs. Native evidence stays false until all checks pass.
- [ ] Review/CI, freeze source/config, off-card input validation, smoke then n5,
  export/verify all raw observations and UID-clean every allocated Job.
- [ ] Publish validated statistics and limits through GitHub flow; update own
  board rows and private manuscript only after evidence gates pass.

Review focus: main/worker data races; actual trigger overlap; observer identity
and compilation prohibition; thread-local Torch settings; bounded failure and
owner-close order. CPU synthetic tests cannot establish native performance.

Execution ledger:10 CPU tests observed RED then GREEN for FIFO order, late
trigger, failed switch, early worker failure, original-call marker, and native
acceptance (wrong output/counts/identity/join rejected). Combined B2/resident
boundary tests40passed1skip. Full local Windows collection remains unavailable
without httpx; exact-head Linux CI is required. No native sample has run.
Ruling: keep this a prepared-file finite FIFO case, avoiding an HTTP stack
that would expand scope; its measured queue behavior cannot imply production
throughput or p99 latency. Runtime/core and immutable fixtures are unchanged.
