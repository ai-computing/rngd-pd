"""Delta load on RNGD: one resident copy of the weights, many compiled variants.

rngd-pd's batched decode graph (engines/decode_batch.py) already takes every
weight as a graph input: the compiled EDF holds the program, not the model.
So a running engine can switch to another batch width or sequence bucket by
loading only that variant's EDF (KB-MB from the artifact cache) and running it
against the device tensors it already holds. Only the per-variant pieces are
built at swap time: the [B,H] norm rows, the position tables of S, and the KV
cache of the new shape.

When the weights themselves change in part (a new checkpoint that shares most
bytes with the resident one), apply_delta() uploads only the tensors whose bytes
changed, as planned by swap.delta from `onggi diff --ranges`.

The numerics are the same as BatchDecodeEngine's: q scaled and norms multiplied
by sqrt(H) in float32 before the cast to bfloat16, everything else a bf16
transpose (bf16 -> f32 -> bf16 is exact).
"""
import json
import math
import os
import time

import numpy as np

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
H, M, NH, NKV, HD = 4096, 14336, 32, 8, 128
NG_LM = 32                                   # lm_head chunks (decode_batch)

# safetensors name suffix -> (slot in the group tuple, transform)
_LAYER = {
    "self_attn.q_proj.weight": ("wq", "q"),
    "self_attn.k_proj.weight": ("wk", "t"),
    "self_attn.v_proj.weight": ("wv", "t"),
    "self_attn.o_proj.weight": ("wo", "t"),
    "mlp.gate_proj.weight": ("wg", "t"),
    "mlp.up_proj.weight": ("wu", "t"),
    "mlp.down_proj.weight": ("wd", "t"),
    "input_layernorm.weight": ("n1", "norm"),
    "post_attention_layernorm.weight": ("n2", "norm"),
}
_SHARED_SLOTS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def _now():
    return time.perf_counter()


class Checkpoint:
    """Lazy per-tensor reader over a safetensors checkpoint directory."""

    def __init__(self, path):
        from safetensors import safe_open
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                self.where = json.load(f)["weight_map"]
        else:
            files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
            self.where = {}
            for fn in files:
                with safe_open(os.path.join(path, fn), framework="pt") as h:
                    for k in h.keys():
                        self.where[k] = fn
        self._open = {}
        self._safe_open = safe_open
        with open(os.path.join(path, "config.json")) as f:
            self.config = json.load(f)

    def get(self, name):
        """torch tensor in the checkpoint's dtype (bf16 for Llama-3.1)."""
        fn = self.where[name]
        h = self._open.get(fn)
        if h is None:
            h = self._open[fn] = self._safe_open(os.path.join(self.path, fn), framework="pt")
        return h.get_tensor(name)

    def inv_freq(self):
        from types import SimpleNamespace
        from rngd_pd.engines.model import ModelConfig
        c = dict(self.config)
        c.setdefault("head_dim", None)
        return ModelConfig(SimpleNamespace(**c)).inv_freq()


def free_capacity_gib(dev, chunk_mib=256):
    """Device capacity still free, by allocating chunks until it fails.

    furiosa-smi does not show furiosa-torch allocations, so this is the only
    public way to see what a step cost. Resolution: one chunk.
    """
    import torch
    held, n = [], chunk_mib * (1 << 20) // 2
    try:
        while True:
            held.append(torch.empty(n, dtype=torch.bfloat16, device=dev))
    except Exception:
        pass
    got = len(held) * chunk_mib / 1024
    del held
    return got


class SwappableDecode:
    """Batched decode on one RNGD card with swappable compiled variants.

    variants: Variant objects that may be swapped to; the graphs come from the
    artifact cache (bench/prebake.py bakes them), never compiled on the swap
    path unless allow_compile is set.
    """

    def __init__(self, checkpoint, group=8, chip=0, cache_dir=None,
                 allow_compile=False, log=print):
        import torch
        from rngd_pd.engines.fsdk import dev_str
        from rngd_pd.paths import ARTIFACT_CACHE
        torch.set_grad_enabled(False)
        self.torch = torch
        self.BF = torch.bfloat16
        self.ck = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
        self.group = group
        self.nl = self.ck.config["num_hidden_layers"]
        assert self.nl % group == 0
        self.ngroups = self.nl // group
        self.dev = dev_str(chip)
        self.chip = chip
        self.cache_dir = str(cache_dir or ARTIFACT_CACHE)
        self.allow_compile = allow_compile
        self.log = log
        self.eps = self.ck.config["rms_norm_eps"]
        self.inv = self.ck.inv_freq()
        self.variants = {}           # Variant -> dict of its loaded state
        self.active = None
        self.timings = {}
        t0 = _now()
        self._load_host_side()
        t1 = _now()
        self._upload_shared()
        t2 = _now()
        self.timings.update(host_weights_s=t1 - t0, upload_weights_s=t2 - t1)

    # ---------------------------------------------------------------- weights

    def _host(self, name, transform):
        """One checkpoint tensor in the layout the graph takes, on the host (bf16)."""
        torch = self.torch
        t = self.ck.get(name)
        if transform == "q":
            return (t.float() * (1.0 / math.sqrt(HD))).t().contiguous().to(self.BF)
        if transform == "t":
            return t.to(self.BF).t().contiguous()
        if transform == "norm":
            return (t.float() * math.sqrt(H))          # float32 [H]; tiled per variant
        raise ValueError(transform)

    def _load_host_side(self):
        """Embedding, final norm and per-layer norms stay on the host (small or host-used)."""
        self.embed = self.ck.get("model.embed_tokens.weight").float().numpy()
        lm_name = "lm_head.weight" if "lm_head.weight" in self.ck.where else \
            "model.embed_tokens.weight"
        self.lm_name = lm_name
        self.norm = self.ck.get("model.norm.weight").float().numpy()
        self.n1 = [self._host(f"model.layers.{i}.input_layernorm.weight", "norm")
                   for i in range(self.nl)]
        self.n2 = [self._host(f"model.layers.{i}.post_attention_layernorm.weight", "norm")
                   for i in range(self.nl)]

    def _group_slot_host(self, g, slot):
        suffix = next(k for k, v in _LAYER.items() if v[0] == slot)
        tr = _LAYER[suffix][1]
        return self.torch.stack([self._host(f"model.layers.{i}.{suffix}", tr)
                                 for i in range(g * self.group, (g + 1) * self.group)])

    def _upload_shared(self):
        """The variant-independent weights: one device copy, shared by every graph."""
        self.shared = []
        for g in range(self.ngroups):
            self.shared.append({s: self._group_slot_host(g, s).to(self.dev)
                                for s in _SHARED_SLOTS})
        self.wlm_d = self._lm_host().to(self.dev)

    def _lm_host(self):
        lm = self.ck.get(self.lm_name).to(self.BF)
        V = lm.shape[0]
        Vp = (V + NG_LM - 1) // NG_LM * NG_LM
        pad = self.torch.zeros(Vp, H, dtype=self.BF)
        pad[:V] = lm
        self.V, self.Vp = V, Vp
        return pad.reshape(NG_LM, Vp // NG_LM, H)

    # --------------------------------------------------------------- variants

    def _graphs(self, v):
        """(decode module, lm_head module, how) from the artifact cache."""
        from rngd_pd.engines.decode_batch import decode_graph_spec, lmhead_graph_spec
        from rngd_pd.engines import artifact_cache as ac
        out = []
        how = []
        for key, build in (decode_graph_spec(v.B, v.S, v.group),
                           lmhead_graph_spec(v.B, self.V, NG_LM)):
            path = ac.path_for(self.cache_dir, key)
            if ac.exists(path):
                out.append(ac.load(path))
                how.append("cache")
            elif self.allow_compile:
                m, _, _ = ac.get_or_compile(self.cache_dir, key, build)
                out.append(m)
                how.append("compile")
            else:
                raise FileNotFoundError(
                    f"{v.name}: graph {key} is not in {self.cache_dir}; bake it first "
                    f"(RNGD_MAX_SHAPES=8 bench/prebake.py --B {v.B} --S {v.S} --group {v.group})")
        return out[0], out[1], how

    def _tables(self, S):
        ang = np.arange(S, dtype=np.float32)[:, None] * self.inv[None, :]
        cs, sn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        half = HD // 2
        rot = np.zeros((S, HD, HD), np.float32)
        i = np.arange(half)
        rot[:, i, i] = cs
        rot[:, i + half, i + half] = cs
        rot[:, i, i + half] = sn
        rot[:, i + half, i] = -sn
        return dict(rot=rot, mask=np.triu(np.full((S, S), -1e4, np.float32), 1),
                    eye=np.eye(S, dtype=np.float32))

    def _per_batch(self, B):
        """The B-shaped tensors: norm rows tiled to [B,H], eps/nh, final norm."""
        d = lambda a: self.torch.from_numpy(np.ascontiguousarray(a)).to(self.BF).to(self.dev)
        df = lambda a: self.torch.from_numpy(np.ascontiguousarray(a)).float().to(self.dev)
        n1 = [self.torch.stack([d(np.tile(self.n1[i].numpy(), (B, 1)))
                                for i in range(g * self.group, (g + 1) * self.group)])
              for g in range(self.ngroups)]
        n2 = [self.torch.stack([d(np.tile(self.n2[i].numpy(), (B, 1)))
                                for i in range(g * self.group, (g + 1) * self.group)])
              for g in range(self.ngroups)]
        return dict(n1=n1, n2=n2,
                    eps=df(np.full((B, H), self.eps * H, np.float32)),
                    nh=df(np.full((B, H), -0.5, np.float32)),
                    norm=d(np.tile(self.norm * math.sqrt(H), (B, 1))))

    def prepare(self, v):
        """Load a variant's graphs and build its small tensors; weights untouched."""
        if v.group != self.group:
            raise ValueError(f"{v.name}: the resident weights are stacked by group "
                             f"{self.group}; a group-{v.group} graph needs its own copy")
        if v in self.variants:
            return dict(prepared=False)
        t0 = _now()
        m, lmh, how = self._graphs(v)
        t1 = _now()
        m.to(self.dev)
        lmh.to(self.dev)
        t2 = _now()
        st = dict(m=m, lmh=lmh, tab=self._tables(v.S), pb=self._per_batch(v.B))
        t3 = _now()
        self.variants[v] = st
        info = dict(prepared=True, graph_source=how, graph_read_s=t1 - t0,
                    graph_to_device_s=t2 - t1, tables_s=t3 - t2)
        st["prepare"] = info
        return info

    def swap(self, v):
        """Make v the active variant. Returns where the time went (seconds)."""
        t0 = _now()
        info = self.prepare(v)
        self.active = v
        info["swap_s"] = _now() - t0
        info["variant"] = v.name
        return info

    def release(self, v):
        """Drop a variant's graphs and tables; the shared weights stay."""
        if v == self.active:
            raise ValueError("cannot release the active variant")
        self.variants.pop(v, None)

    # ----------------------------------------------------------------- decode

    def blank_kv(self):
        v = self.active
        K = [self.torch.zeros(self.group, v.B, NKV, HD, v.S, dtype=self.BF).to(self.dev)
             for _ in range(self.ngroups)]
        V = [self.torch.zeros(self.group, v.B, NKV, v.S, HD, dtype=self.BF).to(self.dev)
             for _ in range(self.ngroups)]
        return K, V

    def step(self, toks, positions, Kd, Vd):
        """One token for each of B streams on the active variant."""
        torch = self.torch
        v = self.active
        st = self.variants[v]
        tab, pb = st["tab"], st["pb"]
        d = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.BF).to(self.dev)
        pos = np.asarray(positions)
        x = d(self.embed[list(toks)])
        rot, maskd, ohd = d(tab["rot"][pos]), d(tab["mask"][pos]), d(tab["eye"][pos])
        for g in range(self.ngroups):
            w = self.shared[g]
            x, Kf, Vf = st["m"](x, pb["eps"], pb["nh"], pb["n1"][g], w["wq"], w["wk"],
                                w["wv"], w["wo"], pb["n2"][g], w["wg"], w["wu"], w["wd"],
                                Kd[g], Vd[g], rot, maskd, ohd)
            Kd[g], Vd[g] = Kf, Vf
        out = st["lmh"](x, pb["norm"], pb["eps"], pb["nh"], self.wlm_d)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        lg = logits.detach().cpu().float().numpy().reshape(self.Vp, v.B)[:self.V]
        return [int(np.argmax(lg[:, b])) for b in range(v.B)], Kd, Vd

    def generate(self, prompts, n_new):
        """Greedy decode of B equal-length prompts, prompt fed through the decode graph.

        Returns (tokens per stream, seconds to the first generated token).
        """
        v = self.active
        assert len(prompts) == v.B and len({len(p) for p in prompts}) == 1
        P = len(prompts[0])
        assert P + n_new <= v.S
        t0 = _now()
        Kd, Vd = self.blank_kv()
        nxt = None
        for p in range(P):
            nxt, Kd, Vd = self.step([pr[p] for pr in prompts], [p] * v.B, Kd, Vd)
        ttft = _now() - t0
        outs = [[t] for t in nxt]
        for p in range(P, P + n_new - 1):
            nxt, Kd, Vd = self.step(nxt, [p] * v.B, Kd, Vd)
            for b, t in enumerate(nxt):
                outs[b].append(t)
        return outs, ttft

    def first_step(self):
        """Request -> first token on the active variant: one step on a fresh cache."""
        v = self.active
        t0 = _now()
        Kd, Vd = self.blank_kv()
        toks, _, _ = self.step([1] * v.B, [0] * v.B, Kd, Vd)
        return _now() - t0, toks

    # ------------------------------------------------------------------ delta

    def apply_delta(self, new_checkpoint, plan, mode="inplace"):
        """Switch the resident weights to new_checkpoint, uploading only plan's tensors.

        plan: swap.delta.UpdatePlan for (current checkpoint -> new_checkpoint).
        Large matrices are replaced in place, layer by layer, in the stacked
        device tensor (mode "inplace"), or the touched [group, ...] stacks are
        uploaded again (mode "group"); the norms and embedding are host-side
        and rebuilt. Returns timings and bytes uploaded.
        """
        new = new_checkpoint if isinstance(new_checkpoint, Checkpoint) \
            else Checkpoint(new_checkpoint)
        t0 = _now()
        old, self.ck = self.ck, new
        uploaded, host_only, rebuilt_norms = 0, [], False
        restack = set()
        try:
            for name in sorted(plan.tensors):
                if name.startswith("model.layers."):
                    i = int(name.split(".")[2])
                    suffix = name.split(".", 3)[3]
                    slot, tr = _LAYER[suffix]
                    if slot in _SHARED_SLOTS:
                        g, li = divmod(i, self.group)
                        if mode == "group":
                            restack.add((g, slot))
                            continue
                        t = self._host(name, tr)
                        self.shared[g][slot][li] = t.to(self.dev)
                        uploaded += t.numel() * 2
                    else:
                        (self.n1 if slot == "n1" else self.n2)[i] = self._host(name, tr)
                        rebuilt_norms = True
                elif name == self.lm_name:
                    self.wlm_d = self._lm_host().to(self.dev)
                    uploaded += self.Vp * H * 2
                    if name == "model.embed_tokens.weight":
                        self.embed = self.ck.get(name).float().numpy()
                elif name == "model.embed_tokens.weight":
                    self.embed = self.ck.get(name).float().numpy()
                    host_only.append(name)
                elif name == "model.norm.weight":
                    self.norm = self.ck.get(name).float().numpy()
                    rebuilt_norms = True
                else:
                    host_only.append(name)
            for g, slot in sorted(restack):
                t = self._group_slot_host(g, slot)
                self.shared[g][slot] = t.to(self.dev)
                uploaded += t.numel() * 2
            if rebuilt_norms:
                for v, st in self.variants.items():
                    st["pb"] = self._per_batch(v.B)
        except Exception:
            self.ck = old
            raise
        return dict(delta_s=_now() - t0, uploaded_bytes=uploaded, mode=mode,
                    tensors=len(plan.tensors), host_only=host_only)
