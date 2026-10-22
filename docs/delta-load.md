# Delta load: swap compiled variants on a running engine

Issue: [#17](https://github.com/ai-computing/rngd-pd/issues/17). Code: [`src/rngd_pd/swap/`](../src/rngd_pd/swap).

A compiled variant is one shape of the same Llama-3.1-8B graph: a batch width `B` and a sequence bucket `S`.
Switching variants used to mean a full model load. Delta load keeps on the card what is already there and
loads only what differs.

For the current ATOM API, transactional RNGD updates and full-prompt measurements,
see [delta-swap.md](delta-swap.md) and the [independent hardware record](../experiments/2026-10-22-delta-swap.md).

## RNGD

rngd-pd's batched decode graph ([`engines/decode_batch.py`](../src/rngd_pd/engines/decode_batch.py)) takes every
weight as a graph input. The compiled EDF holds only the program, so its artifact is 1–6 MB, not 16 GB.

[`SwappableDecode`](../src/rngd_pd/swap/rngd.py) keeps **one** device copy of the weights in that layout:
- the projections, stacked per layer group;
- the chunked LM head.

Every variant of the same layer group runs against this copy.

`swap(variant)` does only the per-variant work:
1. Loads that variant's decode and LM-head EDFs from the artifact cache.
2. Builds the `[B,H]` norm rows and the position tables of `S`.
3. Makes the variant active.

The KV cache of the new shape is created for the next request. A variant that has been prepared once stays loaded,
so swapping back to it does nothing.

```python
from rngd_pd.swap.rngd import SwappableDecode
from rngd_pd.swap.variants import Variant, BatchSelector

eng = SwappableDecode("<Llama-3.1-8B-Instruct dir>", group=8, chip=0)   # weights uploaded once
eng.swap(Variant(4, 512))
outs, _ = eng.generate(prompts_4, n_new=64)

sel = BatchSelector([4, 8, 16], start=4)      # queue depth -> batch, with hysteresis
eng.swap(Variant(sel.observe(queue_depth), 512))
```

The graphs must already be in the artifact cache. `swap()` refuses to compile on the serving path. Bake them on
the host with `RNGD_MAX_SHAPES=8 bench/prebake.py --B <B> --S <S> --group 8`.

### Partial weight update

When the weights themselves change in part, `apply_delta(new_checkpoint, plan)` uploads only the tensors whose bytes
changed. A new checkpoint might share most bytes with the resident one, for example a fine-tune of a few layers.

The plan comes from [`swap/delta.py`](../src/rngd_pd/swap/delta.py):
1. `onggi diff --ranges --json` lists the changed byte ranges of each file.
2. The safetensors header, an open format, maps each range to the tensors and rows it overlaps.
3. onggi's ranges are chunk-granular, so a chunk that straddles a tensor boundary also flags its neighbour.
   `refine_rows` reads just the flagged rows from both files and keeps the rows that really differ.

Without onggi, the files are compared block by block.

```python
from rngd_pd.swap import delta
plan = delta.plan_update(base_dir, new_dir, delta.load_onggi_ranges("ranges.json"))
eng.apply_delta(new_dir, plan)          # mode="inplace" (per layer) or "group" (re-stack)
```

### Historical RNGD measurements from PR #21

These are the original measurements from PR #21. The historical harness timed
a one-token probe with fresh KV, then separately checked greedy generation on
16-token prompts. The current harness times the actual prompt through its first
generated token, so the latency boundaries differ.

Measured on one RNGD card (furiosa-torch 2026.2.0) in a Kubernetes Job. The bench is
`python -m rngd_pd.swap.bench_rngd all`, from [`deploy/k8s/swap-rngd-job.yaml`](../deploy/k8s/swap-rngd-job.yaml).

Setup:
- **Model:** Llama-3.1-8B-Instruct.
- **Graphs:** decode graphs at group 8 with `RNGD_MAX_SHAPES=8`, from the artifact cache.
- **Prompts:** 16 prompt tokens fed through the decode graph, then 16 greedy tokens per stream.
- **Repeats:** two full runs.

Definitions:
- **Full reload:** a fresh process, from launch to the first token: read the checkpoint, upload the weights,
  load the graphs, first step on the one-token probe (not the complete prompt).
- **Swap:** in the running engine, from `swap()` through a one-token probe on the new variant.

| Variant | Full reload (s) | Swap, first time (ms) | Swap back, already prepared (ms) | Greedy tokens = full reload |
|---|--:|--:|--:|:-:|
| B4, S512 | 21.8 / 22.3 | (initial load) | 113 / 145 | yes |
| B8, S512 | 22.0 / 22.2 | 728 / 941 | 148 / 275 | yes |
| B16, S512 | 22.7 / 22.5 | 867 / 1197 | 353 / 539 | yes |
| B4, S256 | 21.4 / 22.3 | 547 / 732 | 56 / 63 | yes |

Notes:
- Each cell gives run 1 / run 2. Tokens were compared for every stream, in both runs.
- **Where the swap time goes:** reading the two EDFs is 0.5–0.7 s; the rest is the first step on the new shape.
  Swapping back is the first step only.
- **Full reload:** 19–20 s of it is the 15 GB weight upload. The rest is the graph load (0.6–0.7 s) and the first
  step.
- **Device memory:**
  - free capacity is 33.5 GiB after the weights, on a 47.25 GiB card;
  - it stays 33.5 GiB with the first three variants loaded, and 33.25 GiB with all four;
  - furiosa-smi does not show furiosa-torch allocations, so this is measured by allocating 256 MiB chunks. Each
    variant therefore costs less than 0.25 GiB, against about 14 GiB for a second weight copy.

Partial weight update:
- **New checkpoint:** three tensors of the base were changed in place (`layers.30.q_proj`, `layers.31.down_proj`,
  `layers.31.post_attention_layernorm`), with the same file layout.
- **onggi:** `onggi diff --ranges` over the two changed shards gave 97.5 % same offset and two changed runs
  (151 MB).

| | Run 1 (onggi ranges as is) | Run 2 (ranges refined) |
|---|--:|--:|
| Tensors uploaded | 8 (5 flagged only by chunk boundaries, `lm_head` among them) | 3 |
| Bytes uploaded | 1.24 GB | 151 MB |
| Planning / upload (s) | 0.002 / 0.61 | 0.17 / 0.24 |
| Apply → first one-token probe, excluding planning (s) | 1.21 | 0.84 |
| Full reload of the new checkpoint (s) | 19.8 | 22.1 |
| Greedy tokens = full reload of the new checkpoint | yes | yes (and different from the base's) |

### Limits

- Variants share weights only within one layer group: the weights are stacked by group. A group-2 graph needs its
  own copy.
- `swap()` happens between requests. A request's KV cache is not carried into a variant of another shape.
- The weights live in the process that owns the device tensors. Another process cannot attach to them; furiosa-torch
  has no public API to share device tensors across processes.

ATOM co-compiled decoder selection and independent-bundle preparation are implemented
in [`swap/atom.py`](../src/rngd_pd/swap/atom.py). See [usage and limits](delta-swap.md)
and [measured token parity and memory costs](../experiments/2026-10-22-delta-swap.md).

## ATOM

On ATOM the weights are compiled into the graph (`rebel.compile_from_torch` embeds them). rebel 0.11 has no public call
that binds a compiled graph to weights another graph already put on the card. Three public-API routes were measured:
one compile session, a fast reload, and weights as device-resident graph inputs. All use one ATOM (RBLN-CA22,
15.7 GiB), rebel-compiler 0.11.0 and Llama-3.1-8B-Instruct, with fp16 weights.

Common setup:
- **Engine:** the batched ATOM decode of `vendors/rbln_decode_batch.py`, with B and S taken from the inputs.
- **Variants:** B1 and B4 at S256, plus B2 at S512 for the variant outside the first session.
- **Tokens:** 16 prompt tokens and 16 greedy tokens per stream. Tokens are compared per stream with a fresh process
  that loaded the same variant.
- **How it ran:** Kubernetes Jobs, one ATOM each; compiles in Jobs without a card
  ([`deploy/k8s/swap-atom-job.yaml`](../deploy/k8s/swap-atom-job.yaml)).
- **Python:** the host's Python with rebel-compiler 0.11.0. The device-input path also needs
  `pip install torch-rbln==0.2.2` (in a venv with `--system-site-packages`), the torch-rbln release that loads
  against rebel 0.11 (0.3.0 and later need a newer runtime).

### One compile session (bucketing): swap inside a session

`rebel.compile_from_torch(model, input_info=[...])` takes a list of input configurations and builds one model with
one graph per shape. The documented *bucketing* feature serves them from one runtime that "selects the bucket from
the input shape", with the weights once. [`swap/atom.py`](../src/rngd_pd/swap/atom.py) builds such a *session* from
a list of variants. `AtomSession.swap(variant)` only changes the shape of the next call.

```sh
python -m rngd_pd.swap.bench_atom build --ckpt <Llama-3.1-8B> --session-out s1.rbln --variants B1,S256 B4,S256      # no card
python -m rngd_pd.swap.bench_atom all --ckpt <Llama-3.1-8B> --session s1.rbln --next s2.rbln --outside B2,S512 --out atom.json
```

| | B1, S256 | B4, S256 |
|---|--:|--:|
| Full reload, launch → first token (s) | 13.3 | 13.7 |
| of which: host parse of the 15.4 GB `.rbln` / `create_runtime` (s) | 10.6 / 0.39 | 10.8 / 0.39 |
| Swap in the session, request → first token (ms), round 1 / round 2 | 113 / 124 | 194 / 199 |
| `swap()` itself (µs) | 6–12 | 11 |
| Decode step (ms) | 74 | 96 |
| Greedy tokens = full reload | yes | yes |

- **Swap:** the swap is the first step on the new shape; the switch itself costs nothing.
- **Device memory:** 14.7 GiB for the session, one weight copy for both variants.
- **Compile:** 153 s without a card. The compile reports a device allocation of 15.75 GB, against 16.88 GB on the
  card.

### A variant outside the session: fast reload and the accumulating session

A variant that is not in the running session gets a new session: the running variants plus the new one, compiled
without a card (231 s for s2 = {B1-S256, B4-S256, B2-S512}). The switch to it is the only reload. Later swaps inside it
are free again.

The reload is cut by taking the host part off the critical path. A full reload is 13.3–13.7 s, and 10.6 s of it is
reading and parsing the `.rbln` on the host. [`HostPool`](../src/rngd_pd/swap/atom.py) does that part in a background
thread while the running session keeps serving. The switch then only closes the old runtime and creates the new one.
The card never holds both sessions: two 15 GB sessions do not fit on one 15.7 GiB card.

| | s1 → s2 for B2, S512 |
|---|--:|
| Full reload of s2, launch → first token (s) | 13.5 |
| Background parse, while s1 keeps serving (s) | 12.1 |
| Switch → first token on B2-S512 (s) | **0.80** (close 0.23 + create 0.40 + first step 0.15) |
| Greedy tokens = full reload of s2 | yes |
| Device memory, s1 → s2 | 14.7 → 15.1 GiB |
| Host memory while preparing | the parsed s2, about its file size (15.6 GB) |

Reading the file into the page cache first does not help when it is already cached: 12.8 s instead of 13.3 s.

### Weights as device-resident graph inputs (torch-rbln)

The decode step takes every weight as an input, in [`swap/atom_inputs.py`](../src/rngd_pd/swap/atom_inputs.py).
- The weights are uploaded once as torch-rbln device tensors (`torch-rbln` 0.2.2, the release that matches
  rebel 0.11).
- Each variant is a `torch.compile(backend="rbln")` graph. The rebel docs say that for inputs already in device
  memory "the runtime patches their addresses"; nothing is copied.
- The KV cache is device-resident too.
- Graphs can be compiled without a card (`mode="compile_only"` with `cache_dir` and `npu`).

Toy check first: 16 fp16 matrices of 4096 × 4096 (512 MiB), batch 1 and 4, from
[`bench/swap_atom_probe.py`](../bench/swap_atom_probe.py).

| Weights | Device memory, graph B1 → B4 | Step | Second graph's weights |
|---|---|--:|---|
| embedded (`compile_from_torch`) | 524 → 1050 MiB | 2.4 ms | its own copy |
| host inputs, passed every run | 604 → 1200 MiB | 61 ms | copied in every run (8.8 GB/s); at 8B, about 1.8 s a step |
| device inputs (torch-rbln) | 1028 → 1032 MiB | 8.6 ms | the same device tensors |

Llama-3.1-8B, 32 layers:

| | B1, S256 | B4, S256 |
|---|--:|--:|
| Full reload, launch → first token (s) | 74.1 | 75.4 |
| of which: weight read / upload / graph compile (s) | 4.7 / 10.5 / 56 | 4.7 / 9.1 / 59 |
| Swap to a new variant, weights resident (s) | – | 61.4 (in-process compile) |
| Swap back to a prepared variant (ms) | 584 | 794 |
| Decode step (ms), weights as inputs / embedded (session) | 511 / 74 | 517 / 96 |
| Greedy tokens = full reload | yes | yes |
| Device memory (GiB) | 15.03 | 15.56 (no second weight copy) |

So on ATOM weights can stay resident and be shared by a separately compiled graph, but on rebel 0.11 it is not a
serving path:
- **Step time:** a step is 5.4–6.9× slower than with embedded weights.
- **Compile on the serving path:** a variant's graph has to be compiled in the serving process (56–61 s for 8B).
  A second graph loaded from another process's compile cache aborts on the card (`SYS_TASK_ABORTED`; checked with
  2 layers, both with one shared and with separate `torch.compile` callables), as in npu-graph-swap Q3 A4–A6.
  A graph compiled in the same process runs against the same tensors without error.
- **Model code:** the per-KV-head `o_proj` split of the embedded-weight graph fails device-graph conversion with
  weights as inputs beyond two layers ("operand #0 does not dominate this use"). Here it is one matmul.

### Public-API limits on ATOM (rebel-compiler 0.11.0, torch-rbln 0.2.2)

- No call binds a compiled graph to constant buffers that another graph already has resident.
- No call exports or names a runtime's weight buffers, and no compile option emits a graph with references to
  external weights.
- `rebel.get_device_memory_info` is not in 0.11. Device memory is read from `rbln-stat -j`.
- Weights as torch-rbln inputs: see the section above.

## Choosing a path

| | RNGD | ATOM |
|---|---|---|
| Variants known in advance | one resident weight copy, `swap()` loads the variant's graph: 0.55–1.2 s | one session (bucketing): swap ≈ first step, 0.11–0.20 s |
| A new variant | bake its graph, then `swap()` | recompile the session with it (no card, minutes), `HostPool` switch: 0.80 s |
| New weights, partly changed | `apply_delta()` uploads only the changed tensors | full reload (weights are compiled in) |
