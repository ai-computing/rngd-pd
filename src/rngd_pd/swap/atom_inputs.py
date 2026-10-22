"""ATOM delta load with the weights as graph inputs on the device (public API).

The decode graph takes every weight as an input instead of embedding it. The
weights are uploaded once as torch-rbln device tensors (`.to("rbln:N")`), and
each variant (B, S) is its own `torch.compile(backend="rbln")` graph run
against those same tensors. The rebel docs: for inputs already in device
memory "the runtime patches their addresses" instead of copying them. So a
variant compiled separately, in another session or on another day, still
uses the resident weights. A swap loads only the new graph (from the compile
cache).

The KV cache is device-resident too: each step returns the updated caches,
which are the next step's inputs.

Graphs are compiled ahead without a card: options mode="compile_only" with
cache_dir and npu (rebel.torch_compile options) write the artifact, and the
serving process finds it in cache_dir.

  python -m rngd_pd.swap.atom_inputs compile --ckpt <Llama-3.1-8B> --cache DIR --variants B1,S256 B4,S256
  python -m rngd_pd.swap.atom_inputs all --ckpt <Llama-3.1-8B> --cache DIR --variants B1,S256 B4,S256 --out x.json
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import wraps
from threading import RLock

T_START = time.perf_counter()

import numpy as np  # noqa: E402

from rngd_pd.swap.variants import Variant  # noqa: E402

NPU = os.environ.get("RBLN_PART", "RBLN-CA22")
PER_LAYER = ("n1", "wq", "wk", "wv", "wo", "n2", "wg", "wu", "wd")
SUFFIXES = ("input_layernorm.weight", "self_attn.q_proj.weight",
            "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "post_attention_layernorm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def weight_names(layers):
    return [f"model.layers.{i}.{suffix}" for i in range(layers) for suffix in SUFFIXES] + [
        "model.norm.weight", "lm_head.weight"]


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("engine is closed")
            return method(self, *args, **kwargs)
    return call


@dataclass
class _KV:
    tensor: object
    generation: int
    variant: Variant


def _now():
    return time.perf_counter()


def host_weights(model_dir, layers=None, wdt=None):
    """Weights in the graph's input order, as host torch tensors.

    Same dtypes as vendors/rbln_decode_batch: norms float32, projections fp16
    (out x in). o_proj stays whole: the per-KV-head split and the eight adds
    that sum it trip rebel 0.11's device-graph conversion once the weights are
    device inputs ("operand does not dominate this use" past two layers).
    """
    import torch
    from safetensors import safe_open
    wdt = wdt or torch.float16
    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    nl = layers or c["num_hidden_layers"]
    H, NH, NKV = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
    HD = H // NH
    G = NH // NKV
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    handles = {}

    def get(name):
        fn = where[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(model_dir, fn), framework="pt")
        return handles[fn].get_tensor(name)

    out = []
    for i in range(nl):
        p = f"model.layers.{i}."
        out += [get(p + "input_layernorm.weight").float(),
                get(p + "self_attn.q_proj.weight").float().to(wdt),
                get(p + "self_attn.k_proj.weight").float().to(wdt),
                get(p + "self_attn.v_proj.weight").float().to(wdt),
                get(p + "self_attn.o_proj.weight").float().to(wdt),
                get(p + "post_attention_layernorm.weight").float(),
                get(p + "mlp.gate_proj.weight").float().to(wdt),
                get(p + "mlp.up_proj.weight").float().to(wdt),
                get(p + "mlp.down_proj.weight").float().to(wdt)]
    out += [get("model.norm.weight").float(), get("lm_head.weight").float().to(wdt)]
    dims = dict(nl=nl, H=H, NH=NH, NKV=NKV, HD=HD, G=G, eps=c["rms_norm_eps"])
    return out, dims


def make_step(d):
    """The decode step as a function of (inputs, *weights); B and S from the shapes."""
    import torch
    nl, NH, NKV, HD, G, eps = d["nl"], d["NH"], d["NKV"], d["HD"], d["G"], d["eps"]
    scale = 1.0 / math.sqrt(HD)

    def mm(a, w):
        return (a.to(w.dtype) @ w.t()).float()

    def rms(x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w

    def rope(x, nh, cos, sin):
        B, half = x.shape[0], HD // 2
        xr = x.view(B, nh, HD)
        a, b = xr[..., :half], xr[..., half:]
        cos = cos.view(B, 1, half)
        sin = sin.view(B, 1, half)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

    def step(x, Kp, Vp, cos, sin, mask, oh, *W):
        B, S = x.shape[0], mask.shape[1]
        ks, vs = [], []
        for i in range(nl):
            n1, wq, wk, wv, wo, n2, wg, wu, wd = W[9 * i:9 * i + 9]
            h = rms(x, n1)
            q = rope(mm(h, wq), NH, cos, sin)
            k = rope(mm(h, wk), NKV, cos, sin)
            v = mm(h, wv).view(B, NKV, HD)
            Kf = Kp[i] + k.unsqueeze(-1) * oh.view(B, 1, 1, S)        # [B,NKV,HD,S]
            Vf = Vp[i] + v.unsqueeze(-2) * oh.view(B, 1, S, 1)        # [B,NKV,S,HD]
            sc = torch.matmul(q.view(B, NKV, G, HD), Kf) * scale + mask.view(B, 1, 1, S)
            ao = torch.matmul(torch.softmax(sc, dim=-1), Vf)          # [B,NKV,G,HD]
            x2 = x + mm(ao.reshape(B, NH * HD), wo)
            h2 = rms(x2, n2)
            x = mm(torch.nn.functional.silu(mm(h2, wg)) * mm(h2, wu), wd) + x2
            ks.append(Kf)
            vs.append(Vf)
        normw, lmw = W[9 * nl], W[9 * nl + 1]
        return mm(rms(x, normw), lmw), torch.stack(ks), torch.stack(vs)

    return step


def example_inputs(d, B, S, device=None):
    import torch
    z = lambda *s: torch.zeros(*s, device=device)
    return [z(B, d["H"]), z(d["nl"], B, d["NKV"], d["HD"], S), z(d["nl"], B, d["NKV"], S, d["HD"]),
            z(B, d["HD"] // 2), z(B, d["HD"] // 2), z(B, S), z(B, S)]


def compile_variants(model_dir, cache_dir, variants, layers=None):
    """Write each variant's graph into cache_dir. No card: mode compile_only."""
    import torch
    import rebel  # noqa: F401  registers backend="rbln"
    torch.set_grad_enabled(False)
    W, d = host_weights(model_dir, layers)
    step = make_step(d)
    out = {}
    for v in variants:
        t0 = _now()
        g = torch.compile(step, backend="rbln", dynamic=False,
                          options={"cache_dir": cache_dir, "npu": NPU, "mode": "compile_only"})
        g(*example_inputs(d, v.B, v.S), *W)
        out[v.name] = _now() - t0
        print(f"[compile] {v.name} {out[v.name]:.0f}s", flush=True)
        torch._dynamo.reset()
    return out


class DeviceWeightDecode:
    """Weights uploaded once to rbln:N; one compiled graph per variant, all sharing them."""

    def __init__(self, model_dir, cache_dir, device=0, layers=None):
        import torch
        import rebel  # noqa: F401
        import torch_rbln  # noqa: F401  registers the rbln device
        from rngd_pd.swap.atom import Tables
        torch.set_grad_enabled(False)
        self.torch = torch
        from .rngd import Checkpoint
        self.ck = Checkpoint(model_dir)
        self._lock, self._closed, self._generation = RLock(), False, 0
        self.dev = f"rbln:{device}"
        self.cache_dir = cache_dir
        t0 = _now()
        W, self.d = host_weights(model_dir, layers)
        t1 = _now()
        self.W = [w.to(self.dev) for w in W]
        del W
        t2 = _now()
        self.tables = Tables(model_dir)
        self.step_fn = make_step(self.d)
        # One compiled callable for every variant: a new shape is a new graph
        # under the same torch.compile session, which the rbln backend loads
        # from cache_dir. Separate torch.compile objects per variant abort on
        # the device when the second one runs (see docs/delta-load.md).
        self.g = torch.compile(self.step_fn, backend="rbln", dynamic=False,
                               options={"cache_dir": cache_dir, "npu": NPU})
        self.graphs = {}
        self.active = None
        self.timings = dict(host_weights_s=t1 - t0, upload_weights_s=t2 - t1)

    @_serialized
    def swap(self, v):
        """Make v active: its compiled graph (from the cache) on the resident weights."""
        t0 = _now()
        prepared = v not in self.graphs
        if prepared:
            self.graphs[v] = self.g
        if self.active != v:
            self._generation += 1
        self.active = v
        return dict(variant=v.name, prepared=prepared, swap_s=_now() - t0)

    @_serialized
    def blank_kv(self):
        d, v = self.d, self.active
        if v is None:
            raise ValueError("select a variant first")
        z = lambda *s: self.torch.zeros(*s).to(self.dev)
        return tuple(_KV(t, self._generation, v) for t in (
            z(d["nl"], v.B, d["NKV"], d["HD"], v.S),
            z(d["nl"], v.B, d["NKV"], v.S, d["HD"])))

    @_serialized
    def step(self, toks, pos, K, V):
        torch, v = self.torch, self.active
        if any(not isinstance(t, _KV) or t.generation != self._generation or t.variant != v
               for t in (K, V)):
            raise ValueError("weights or variant changed; start a fresh request")
        if (len(toks) != v.B or len(pos) != v.B
                or any(p < 0 or p >= v.S for p in pos)):
            raise ValueError("tokens/positions do not fit the active variant")
        host = self.tables.step_inputs(toks, pos, v.S)
        x, cos, sin, mask, oh = (torch.from_numpy(a).to(self.dev) for a in host)
        logits, K, V = self.graphs[v](x, K.tensor, V.tensor, cos, sin, mask, oh, *self.W)
        lg = logits.cpu().float().numpy()
        return ([int(t) for t in np.argmax(lg, axis=-1)],
                _KV(K, self._generation, v), _KV(V, self._generation, v))

    @_serialized
    def first_step(self):
        v = self.active
        t0 = _now()
        K, V = self.blank_kv()
        self.step([1] * v.B, [0] * v.B, K, V)
        return _now() - t0

    @_serialized
    def generate(self, prompts, n_new):
        v = self.active
        if (v is None or len(prompts) != v.B or not prompts[0] or n_new < 1
                or len({len(p) for p in prompts}) != 1
                or len(prompts[0]) + n_new - 1 > v.S):
            raise ValueError("prompts/generation do not fit the active variant")
        P = len(prompts[0])
        t0 = _now()
        K, V = self.blank_kv()
        nxt = None
        for p in range(P):
            nxt, K, V = self.step([pr[p] for pr in prompts], [p] * v.B, K, V)
        ttft = _now() - t0
        outs = [[t] for t in nxt]
        ts = []
        for p in range(P, P + n_new - 1):
            t1 = _now()
            nxt, K, V = self.step(nxt, [p] * v.B, K, V)
            ts.append(_now() - t1)
            for b, t in enumerate(nxt):
                outs[b].append(t)
        ts.sort()
        self.last_step_s = ts[len(ts) // 2] if ts else None
        return outs, ttft

    def load_delta(self, new_checkpoint, **options):
        """Update only changed weights, optionally querying onggi for the ranges."""
        from .load import load_delta
        return load_delta(self, new_checkpoint, **options)

    def _upload_weight(self, tensor):
        return tensor.to(self.dev)

    @_serialized
    def apply_delta(self, new_checkpoint, plan):
        """Stage changed tensors, then atomically replace their graph inputs.

        Unchanged device allocations and compiled graphs stay alive. A failed
        staging allocation leaves the active checkpoint untouched. Staging
        needs device headroom for the changed tensors, not a second model.
        This applies to device-input graphs, not embedded-weight .rbln files.
        """
        from .rngd import Checkpoint
        from .delta import CHECKPOINT_GRAPH_KEYS
        new = new_checkpoint if isinstance(new_checkpoint, Checkpoint) else Checkpoint(new_checkpoint)
        old = self.ck
        if (plan.base_dir != os.path.realpath(old.path)
                or plan.new_dir != os.path.realpath(new.path)):
            raise ValueError("delta plan does not describe the resident and target checkpoints")
        if any(old.config.get(k) != new.config.get(k) for k in CHECKPOINT_GRAPH_KEYS):
            raise ValueError("checkpoint graph/rotary configuration changed; reload required")
        if plan.removed or set(old.where) != set(new.where):
            raise ValueError("checkpoint tensor set changed; reload required")
        slots = {name: i for i, name in enumerate(weight_names(self.d["nl"]))}
        embed_name = "model.embed_tokens.weight"
        staged = {}
        for name in plan.tensors:
            if name not in slots and name != embed_name:
                raise ValueError(f"unsupported changed tensor: {name}")
            a, b = old.get(name), new.get(name)
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"tensor layout changed; reload required: {name}")
            staged[name] = b
        t0 = _now()
        weights = list(self.W)
        embed = self.tables.embed
        uploaded = 0
        for name, tensor in staged.items():
            if name == embed_name:
                embed = tensor.float().numpy()
            else:
                i = slots[name]
                host = tensor.to(dtype=self.W[i].dtype)
                weights[i] = self._upload_weight(host)
                uploaded += host.numel() * host.element_size()
        # The request lock covers planning, staging and the pointer commit.
        self.W, self.tables.embed, self.ck = weights, embed, new
        self._generation += 1
        return dict(delta_s=_now() - t0, uploaded_bytes=uploaded,
                    tensors=len(plan.tensors), mode="replace",
                    host_only=[embed_name] if embed_name in staged else [])

    def close(self):
        with self._lock:
            self.graphs.clear()
            self.g = self.step_fn = None
            self.W.clear()
            self.ck.close()
            self.active = None
            self._closed = True


# --------------------------------------------------------------------- bench

def prompts(n, P, seed=0):
    rng = np.random.default_rng(seed)
    return [list(map(int, rng.integers(100, 120000, P))) for _ in range(n)]


def _mem():
    from rngd_pd.swap.bench_atom import mem
    return mem(0)


def _write(a, obj):
    obj["argv"] = sys.argv[1:]
    with open(a.out, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"[bench] wrote {a.out}", flush=True)


def cmd_compile(a):
    vs = [Variant.parse(x, 0) for x in a.variants]
    _write(a, dict(mode="compile", compile_s=compile_variants(a.ckpt, a.cache, vs, a.layers)))


def cmd_fresh(a):
    v = Variant.parse(a.variant, 0)
    t0 = _now()
    eng = DeviceWeightDecode(a.ckpt, a.cache, a.device, a.layers)
    m_w = _mem()
    info = eng.swap(v)
    first = eng.first_step()
    t_first = _now()
    wall = time.time()
    toks, ttft = eng.generate(prompts(v.B, a.P), a.n)
    _write(a, dict(mode="fresh", variant=v.name, weights=eng.timings, graph=info,
                   first_step_s=first, load_to_first_token_s=t_first - t0,
                   launch_to_first_token_s=(wall - a.t_launch) if a.t_launch else None,
                   step_s=eng.last_step_s, mem_after_weights=m_w, mem_after_graph=_mem(),
                   prompt_ttft_s=ttft, tokens=toks))


def cmd_swap(a):
    vs = [Variant.parse(x, 0) for x in a.variants]
    eng = DeviceWeightDecode(a.ckpt, a.cache, a.device, a.layers)
    rows = [dict(mem_after_weights=_mem())]
    for rnd in range(2):
        for v in vs:
            t0 = _now()
            info = eng.swap(v)
            info["first_step_s"] = eng.first_step()
            info["request_to_first_token_s"] = _now() - t0
            toks, ttft = eng.generate(prompts(v.B, a.P), a.n)
            info.update(round=rnd, tokens=toks, prompt_ttft_s=ttft, step_s=eng.last_step_s,
                        mem=_mem() if rnd == 0 else None)
            rows.append(info)
            print(f"[swap] round {rnd} {v.name}: {info['request_to_first_token_s']*1e3:.0f} ms",
                  flush=True)
    _write(a, dict(mode="swap", weights=eng.timings, rows=rows))


def _child(a, mode, out, extra, cache=None):
    cmd = [sys.executable, "-m", "rngd_pd.swap.atom_inputs", mode, "--ckpt", a.ckpt,
           "--cache", cache or a.cache, "--device", str(a.device), "--P", str(a.P), "--n", str(a.n),
           "--out", out, "--t-launch", repr(time.time())] + \
        (["--layers", str(a.layers)] if a.layers else []) + extra
    print("[all] " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)
    with open(out) as f:
        return json.load(f)


def cmd_all(a):
    d = a.out + ".d"
    os.makedirs(d, exist_ok=True)
    res = dict(fresh={})
    for x in a.variants:
        n = Variant.parse(x, 0).name
        res["fresh"][n] = _child(a, "fresh", f"{d}/fresh_{n}.json", ["--variant", x])
    # The swapping engine compiles its variants in its own session (an empty
    # cache): a second graph loaded from another process's compile aborts on
    # rebel 0.11 (docs/delta-load.md).
    swap_cache = a.swap_cache or a.cache
    res["swap"] = _child(a, "swap", f"{d}/swap.json", ["--variants", *a.variants],
                         cache=swap_cache)
    for r in res["swap"]["rows"][1:]:
        r["tokens_equal_fresh"] = r["tokens"] == res["fresh"][r["variant"]]["tokens"]
    _write(a, res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compile", "fresh", "swap", "all"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="torch.compile cache_dir for the graphs")
    ap.add_argument("--swap-cache", help="all: compile cache of the swapping engine (empty: compile in-process)")
    ap.add_argument("--variant")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--t-launch", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"compile": cmd_compile, "fresh": cmd_fresh, "swap": cmd_swap, "all": cmd_all}[a.mode](a)


if __name__ == "__main__":
    main()
