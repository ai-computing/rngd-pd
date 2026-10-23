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
from functools import wraps
from threading import RLock

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


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            if self._failed:
                raise RuntimeError("resident weights are unavailable; reload the engine")
            return method(self, *args, **kwargs)
    return call


class _KV(list):
    def __init__(self, tensors, generation, owner):
        super().__init__(tensors)
        self.generation = generation
        self.owner = owner


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

    def close(self):
        self._open.clear()

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


class SwappableDecode:
    """Batched decode on one RNGD card with swappable compiled variants.

    variants: Variant objects that may be swapped to; the graphs come from the
    artifact cache (bench/prebake.py bakes them), never compiled on the swap
    path unless allow_compile is set.
    """

    def __init__(self, checkpoint, group=8, chip=0, cache_dir=None,
                 allow_compile=False, log=print):
        import torch
        from rngd_pd.engines.fsdk import dev_str, open_device
        from rngd_pd.paths import ARTIFACT_CACHE
        torch.set_grad_enabled(False)
        self.torch = torch
        self._lock = RLock()
        self._failed = False
        self._generation = 0
        self._kv_owner = object()
        self.BF = torch.bfloat16
        self.ck = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
        dims = self.ck.config
        expected = dict(hidden_size=H, intermediate_size=M, num_attention_heads=NH,
                        num_key_value_heads=NKV)
        if any(dims.get(k) != value for k, value in expected.items()):
            raise ValueError("these decode graphs require the Llama-3.1-8B dimensions")
        if (dims.get("head_dim", HD) not in (None, HD)
                or dims.get("hidden_act", "silu") != "silu"
                or dims.get("attention_bias", False) or dims.get("mlp_bias", False)):
            raise ValueError("unsupported head dimensions, activation or projection biases")
        self.group = group
        self.nl = self.ck.config["num_hidden_layers"]
        if type(group) is not int or group < 1 or self.nl % group:
            raise ValueError("group must divide the layer count")
        self.ngroups = self.nl // group
        self.dev = dev_str(chip)
        self.chip = chip
        # Set fusion/mesh BEFORE the first weight upload, not only when the
        # decode kernel is imported later by _graphs(). The default one-PE
        # address space cannot hold the resident 8B weights.
        open_device(8, 2)
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
        self._sync()
        t2 = _now()
        self.timings.update(host_weights_s=t1 - t0, upload_weights_s=t2 - t1)

    def _sync(self):
        from rngd_pd.engines.fsdk import sync
        sync(self.chip)

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

    @_serialized
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
        self._sync()
        t3 = _now()
        self.variants[v] = st
        info = dict(prepared=True, graph_source=how, graph_read_s=t1 - t0,
                    graph_to_device_s=t2 - t1, tables_s=t3 - t2)
        st["prepare"] = info
        return info

    @_serialized
    def swap(self, v):
        """Make v the active variant. Returns where the time went (seconds)."""
        t0 = _now()
        info = self.prepare(v)
        if self.active != v:
            self._generation += 1
        self.active = v
        info["swap_s"] = _now() - t0
        info["variant"] = v.name
        return info

    @_serialized
    def select(self, selector, depth, context, variants):
        """Load-driven hook for NEW requests after old KV has drained."""
        batch = selector.observe(depth)
        choices = [v for v in variants if v.B == batch and v.S >= context
                   and v.group == self.group]
        if not choices:
            raise ValueError("no compiled variant fits the selector and context")
        return self.swap(min(choices, key=lambda v: v.S))

    @_serialized
    def release(self, v):
        """Drop a variant's graphs and tables; the shared weights stay."""
        if v == self.active:
            raise ValueError("cannot release the active variant")
        self.variants.pop(v, None)

    # ----------------------------------------------------------------- decode

    @_serialized
    def blank_kv(self):
        v = self.active
        if v is None:
            raise RuntimeError("select a variant with swap() first")
        K = [self.torch.zeros(self.group, v.B, NKV, HD, v.S, dtype=self.BF).to(self.dev)
             for _ in range(self.ngroups)]
        V = [self.torch.zeros(self.group, v.B, NKV, v.S, HD, dtype=self.BF).to(self.dev)
             for _ in range(self.ngroups)]
        return (_KV(K, self._generation, self._kv_owner),
                _KV(V, self._generation, self._kv_owner))

    @_serialized
    def step(self, toks, positions, Kd, Vd):
        """One token for each of B streams on the active variant."""
        torch = self.torch
        v = self.active
        if v is None:
            raise RuntimeError("select a variant with swap() first")
        if (len(toks) != v.B or len(positions) != v.B
                or any(not isinstance(t, (int, np.integer)) or not 0 <= t < self.V for t in toks)
                or not isinstance(Kd, _KV) or not isinstance(Vd, _KV)
                or Kd.owner is not self._kv_owner or Vd.owner is not self._kv_owner
                or Kd.generation != self._generation or Vd.generation != self._generation
                or any(type(p) not in (int, np.int32, np.int64) or not 0 <= p < v.S
                       for p in positions)
                or len(Kd) != self.ngroups or len(Vd) != self.ngroups
                or any(tuple(t.shape) != (self.group, v.B, NKV, HD, v.S) for t in Kd)
                or any(tuple(t.shape) != (self.group, v.B, NKV, v.S, HD) for t in Vd)):
            raise ValueError("inputs/KV do not fit the active variant; start a fresh request")
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

    @_serialized
    def generate(self, prompts, n_new):
        """Greedy decode of B equal-length prompts, prompt fed through the decode graph.

        Returns (tokens per stream, seconds to the first generated token).
        """
        v = self.active
        if (v is None or len(prompts) != v.B or not prompts[0]
                or len({len(p) for p in prompts}) != 1 or type(n_new) is not int or n_new < 1):
            raise ValueError("need B nonempty, equal-length prompts and positive n_new")
        P = len(prompts[0])
        if P + n_new > v.S:
            raise ValueError("prompt + generation exceed the selected context")
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

    @_serialized
    def first_step(self):
        """Request -> first token on the active variant: one step on a fresh cache."""
        v = self.active
        t0 = _now()
        Kd, Vd = self.blank_kv()
        toks, _, _ = self.step([1] * v.B, [0] * v.B, Kd, Vd)
        return _now() - t0, toks

    # ------------------------------------------------------------------ delta

    def load_delta(self, new_checkpoint, **options):
        """Plan and apply changed weights (ranges JSON, live onggi, or local diff)."""
        from .load import load_delta
        return load_delta(self, new_checkpoint, **options)

    @_serialized
    def apply_delta(self, new_checkpoint, plan, mode="inplace"):
        """Update changed tensors at a drained request boundary.

        Validate and stage before touching device state. Inplace mode uploads
        only changed layer tensors and restores them on an error. Group mode
        stages replacement group tensors and commits references atomically,
        at the cost of uploading all layers in each touched group. If device
        rollback itself fails the engine stops serving until reloaded.
        """
        if mode not in ("inplace", "group"):
            raise ValueError("delta mode must be inplace or group")
        new = new_checkpoint if isinstance(new_checkpoint, Checkpoint) else Checkpoint(new_checkpoint)
        old = self.ck
        if (plan.base_dir != os.path.realpath(old.path)
                or plan.new_dir != os.path.realpath(new.path)):
            raise ValueError("delta plan does not describe the resident and target checkpoints")
        # Changing graph semantics is a full reload, not a weight update.
        from .delta import CHECKPOINT_GRAPH_KEYS
        if any(old.config.get(k) != new.config.get(k) for k in CHECKPOINT_GRAPH_KEYS):
            raise ValueError("checkpoint graph/rotary configuration changed; reload required")
        if plan.removed or set(old.where) != set(new.where):
            raise ValueError("checkpoint tensor set changed; reload required")
        for name in plan.tensors:
            if name.startswith("model.layers."):
                parts = name.split(".", 3)
                if (len(parts) != 4 or not parts[2].isdigit()
                        or not 0 <= int(parts[2]) < self.nl or parts[3] not in _LAYER):
                    raise ValueError(f"unsupported changed tensor: {name}")
            elif name not in {self.lm_name, "model.embed_tokens.weight", "model.norm.weight"}:
                raise ValueError(f"unsupported changed tensor: {name}")
            a, b = old.get(name), new.get(name)
            if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
                raise ValueError(f"tensor layout changed; reload required: {name}")

        t0 = _now()
        old_host = (self.embed, self.norm, self.n1, self.n2, self.wlm_d)
        old_pb = {v: st["pb"] for v, st in self.variants.items()}
        old_shared = [dict(slots) for slots in self.shared]
        staged, restore, replacements = {}, {}, {}
        uploaded, host_only, norms_changed = 0, [], False
        attempted = []
        try:
            # Invalidate before mutation: cancellation must not leave old KV usable.
            self._generation = getattr(self, "_generation", 0) + 1
            self.n1, self.n2 = list(self.n1), list(self.n2)
            # Host transforms can fail: do them all before the first write.
            self.ck = new
            for name in sorted(plan.tensors):
                if name.startswith("model.layers."):
                    i = int(name.split(".")[2])
                    slot, tr = _LAYER[name.split(".", 3)[3]]
                    if slot in _SHARED_SLOTS:
                        g, li = divmod(i, self.group)
                        if mode == "group":
                            replacements[(g, slot)] = None
                        else:
                            staged[(g, slot, li)] = self._host(name, tr)
                    else:
                        (self.n1 if slot == "n1" else self.n2)[i] = self._host(name, tr)
                        norms_changed = True
                elif name == self.lm_name:
                    self.wlm_d = self._lm_host().to(self.dev)
                    uploaded += self.Vp * H * 2
                    if name == "model.embed_tokens.weight":
                        self.embed = new.get(name).float().numpy()
                elif name == "model.embed_tokens.weight":
                    self.embed = new.get(name).float().numpy()
                    host_only.append(name)
                elif name == "model.norm.weight":
                    self.norm = new.get(name).float().numpy()
                    norms_changed = True
            for g, slot in replacements:
                t = self._group_slot_host(g, slot)
                replacements[(g, slot)] = t.to(self.dev)
                uploaded += t.numel() * 2
            new_pb = {v: self._per_batch(v.B) for v in self.variants} if norms_changed else old_pb
            if norms_changed:
                for pb in new_pb.values():
                    uploaded += sum(t.numel() * t.element_size()
                                    for key in ("n1", "n2") for t in pb[key])
                    uploaded += sum(pb[k].numel() * pb[k].element_size()
                                    for k in ("eps", "nh", "norm"))
            self._sync()
            # Capture rollback values from the *old* checkpoint, not from a
            # slow or unsupported device readback.
            self.ck = old
            for g, slot, li in staged:
                suffix = next(k for k, value in _LAYER.items() if value[0] == slot)
                restore[(g, slot, li)] = self._host(
                    f"model.layers.{g * self.group + li}.{suffix}", _LAYER[suffix][1])
            self.ck = new
            for key, tensor in staged.items():
                g, slot, li = key
                attempted.append(key)  # include a write that partially fails
                self.shared[g][slot][li].copy_(tensor.to(self.dev))
                uploaded += tensor.numel() * tensor.element_size()
            self._sync()
            for (g, slot), tensor in replacements.items():
                self.shared[g][slot] = tensor
            for v, pb in new_pb.items():
                self.variants[v]["pb"] = pb
        except BaseException:
            try:
                self.ck = old
                self.embed, self.norm, self.n1, self.n2, self.wlm_d = old_host
                self.shared = old_shared
                for v, pb in old_pb.items():
                    self.variants[v]["pb"] = pb
                for g, slot, li in attempted:
                    self.shared[g][slot][li].copy_(restore[(g, slot, li)].to(self.dev))
                self._sync()
            except BaseException as rollback_error:
                self._failed = True
                raise RuntimeError("delta update and rollback failed; reload required") from rollback_error
            raise
        return dict(delta_s=_now() - t0, uploaded_bytes=uploaded, mode=mode,
                    tensors=len(plan.tensors), host_only=host_only)

    def close(self):
        with self._lock:
            self._sync()
            self.variants.clear()
            self.shared.clear()
            self.wlm_d = None
            self.active = None
            self._failed = True
            self.ck.close()
