# Parallel CPU staging for initial RNGD loading

2026-10-30. This targets initial model preparation in the custom resident engine.
The improvement comes from CPU weight layout preparation, not faster device
DMA or a new Furiosa-LLM FXB loader.

## Diagnosis and implementation

On the same SDK2026.4/Torch2.10 8B configuration, an instrumented baseline
constructor took 18.746s. Its `upload_weights_s` includes CPU layout work:
15.381s in group builders/LM host padding versus 1.946s in 113 device `.to` calls
and 0.00019s final synchronization. Of CPU group work, q/k/v/o/gate/up/down
builders total 2.885/.201/.213/1.043/3.680/3.639/3.573s. These are wall intervals,
not CPU-cycle or PCIe-only measurements; nested transform/stack times overlap
their containing builder interval and must not be added again.

The selected implementation uses a bounded two-worker pool for pure CPU
transforms, with all checkpoint reads and device operations on the caller.
It preserves q scaling in float32 followed by the original BF16 conversion,
contiguous layouts, layer order and subsequent stacking. Workers disable grad
locally; the executor joins on success, exceptions and cancellation. It is
used only for initial loading; row updates/group rebuilds remain serial.

`host_workers=1` is the unchanged default. Select 2 only when CPU capacity is
available. With Torch intra-op=4, two transform workers can execute eight CPU threads
plus runtime/main-thread overhead; this allocation requested 4 CPUs with limit 12.
No global Torch thread setting is changed by the engine.

The same-host CPU-only probe checked 36 timing samples plus 12 prechecks, all
bitwise BF16-equal and contiguous. Two workers improved q 205->114ms,
gate 220->143ms and down 220->146ms, including pool setup/join. Faster q casting
was deliberately excluded to keep arithmetic/default behavior identical.
Direct strided stacking, NumPy bit transposes and other local prototypes were
slower and were not integrated. No native transfer redesign was needed.

## Actual unmodified constructor comparison

Frozen production source:`a11f9c5d7906f2c3c8dc5fd34c9afc1b32524333`.
Llama-3.1-8B-Instruct revision`0e9e39f249a16976918f6564b8830bc894c89659`,
SDK2026.4.0, Torch2.10.0+cpu, B4/S256/group2, RNGD fusion8/mesh2.
Existing compatible graph caches and immutable checkpoint files were mounted
read-only. This is fresh device model preparation with a warm filesystem and
prepared graphs, not cold-disk I/O or compilation. The known group8 compiler
limit from the [SDK migration](2026-10-30-rngd-sdk264.md) remains.

Four independent processes run workers 1,2,2,1. The measured constructor is the
unmodified production method: no profiling wrappers, replacement builders or
transfer hooks. Native output hooks check device placement outside that timer.
All 32 generated token IDs per process equal the independent prior base reference.

| Run | Workers | Constructor | Constructor + swap + first token | Peak process RSS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 18.784667s | 18.973325s | 20,138,889,216B |
| 2 | 2 | 12.936738s | 13.119473s | 20,606,210,048B |
| 3 | 2 | 13.249383s | 13.441458s | 20,384,280,576B |
| 4 | 1 | 18.572734s | 18.759265s | 20,119,244,800B |

Constructor medians: **18.678701 -> 13.093061s**, a 5.585640s/**29.9%** reduction
in this four-process comparison. Peak observed RSS increased from 20.14GB
to 20.61GB (decimal); the configured Pod memory limit was 128GiB. Keep the CPU/RAM tradeoff
when deploying this optional mode. A shared host and two samples per mode do
not establish a general speedup or statistical significance.

Full process startup/imports are separate from this constructor timer. This
does not change token decoding speed, Onggi comparison speed, or the original
weight bytes transferred to the NPU (15,009,316,864 logical BF16 bytes).

## Use and validation

```python
from rngd_pd.swap.rngd import SwappableDecode
from rngd_pd.swap.variants import Variant

engine = SwappableDecode(checkpoint, chip=allocated_chip, group=2,
                         cache_dir=cache, host_workers=2)
engine.swap(Variant(4, 256, 2))
tokens, ttft = engine.generate([prompt_ids] * 4, n_new=8)
engine.close()
```

Use the identical code with `host_workers=1` for the serial control, each in a
fresh process on the same assigned card. Match Torch intra-op threads, CPU
limits, graph cache, checkpoint, prompts and native reference. Prepared worker
configs accept the same [RNGD-only option](../docs/prepared-serving.md).

CPU tests cover BF16/F16/F32 inputs, groups 1/2/8, genuinely strided tensors,
bitwise layout equivalence, caller-only checkpoint reads, no-grad workers,
bounded concurrency and joining on RuntimeError/KeyboardInterrupt/SystemExit.
Prepared configs validate and pass the option before opening the SDK.
Full local suite after the review fix: 352 passed, 8 skipped, 4 subtests passed.
An independent reviewer found no production blockers; the minor contiguous
fixture issue was reproduced and fixed before final validation.

The two-worker engine also passed sparse-projection and sparse-LM resident
runs: eight alternating updates / 256 tokens equal fresh references, plus two
real-write interruption recoveries / 64 base tokens. All 113 projection/LM device
pointers and compiled module identities stayed unchanged; graph preparation
after warmup was forbidden and stale KV was rejected. The previous SDK cache
and row-update limits remain unchanged.

The owned Pod and namespace were deleted with UID preconditions; independent
API checks confirmed them absent. The original provider and shared SDK/driver
were unchanged. Exact raw timing/token/source and cleanup records are retained
in an immutable export; public JSON removes machine paths and node names.

## Evidence

- [Instrumented baseline and all CPU probe samples](data/rngd-host-staging-20261030/diagnostic.json).
- [All four unmodified constructors, tokens and resident recoveries](data/rngd-host-staging-20261030/full-model.json).
- [Source, artifact, fixture and cleanup provenance](data/rngd-host-staging-20261030/provenance.json).

The immutable private export contains 255 verified files; its manifest SHA256
is `8a9d61eb750926b42c543ab0b1037f9cfcb7bbf5a46500f2b3ccf51671030e3b`.
All 771 tracked files across baseline and candidate were independently compared
to canonical Git bytes, and copied production sources matched. Production `src/`
at publication is identical to the frozen native-tested commit. Token arrays
and complete allocation/compiled-identity dictionaries were independently
compared to prior fresh references, rather than trusting recorded pass flags.
Fixture hashes are a read-only copy of the prior accepted manifest; no new
whole-model hashing pass is claimed. Process RSS is measured; container cgroup
memory counters were unavailable at the inspected paths, and no peak HBM
measurement is claimed. Final publication-head CI is linked in the PR.
