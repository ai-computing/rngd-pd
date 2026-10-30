"""Prepared resident engines for the cluster worker; no SDK imports until startup."""
import json
import os

from rngd_pd.swap.variants import Variant


def read_config(path, S):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("engine") not in ("rngd", "atom", "atom-inputs"):
        raise ValueError("prepared engine must be rngd, atom or atom-inputs")
    if type(cfg.get("warmup", False)) is not bool:
        raise ValueError("warmup must be a boolean")
    if "host_workers" in cfg:
        workers = cfg["host_workers"]
        if cfg["engine"] != "rngd" or type(workers) is not int or workers not in (1, 2):
            raise ValueError("host_workers is RNGD-only and must be 1 or 2")
    for key in ("checkpoint", "checkpoint_id"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"prepared config requires {key}")
    names = cfg.get("variants")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("prepared config requires a list of variants")
    group = 8 if cfg["engine"] == "rngd" else 0
    variants = [Variant.parse(n, group) for n in names]
    if (len(set(variants)) != len(variants) or len({v.group for v in variants}) != 1
            or (cfg["engine"] == "rngd" and variants[0].group < 1)
            or (cfg["engine"] != "rngd" and variants[0].group != 0)):
        raise ValueError("prepared variants must be unique and use one compatible group")
    largest = Variant(max(v.B for v in variants), max(v.S for v in variants), variants[0].group)
    if largest not in variants:
        raise ValueError("largest B/S combination must be prepared to serve mixed batches")
    if largest.S > S:
        raise ValueError("prepared context exceeds the configured KV wire context S")
    base = os.path.dirname(os.path.abspath(path))
    for key in ("checkpoint", "cache_dir", "artifact"):
        if cfg.get(key):
            cfg[key] = os.path.abspath(os.path.join(base, cfg[key]))
    if cfg["engine"] == "atom" and not cfg.get("artifact"):
        raise ValueError("ATOM needs a co-compiled .rbln artifact")
    cfg["variants"] = variants
    return cfg


class PreparedEngine:
    """Adapt native KV layouts while keeping one batch's cache private.

    The cluster holds DEVICE from activation through the final step. Callers
    never supply an opaque cache from another engine or checkpoint.
    """

    def __init__(self, engine, variants, kind, checkpoint_id, nl, nkv, hd):
        if not variants or not checkpoint_id or kind not in ("rngd", "atom", "atom-inputs"):
            raise ValueError("invalid prepared engine")
        self.engine, self.variants, self.kind = engine, tuple(variants), kind
        self.checkpoint_id = checkpoint_id
        self.nl, self.nkv, self.hd = nl, nkv, hd
        self.B = max(v.B for v in variants)
        self.max_S = max(v.S for v in variants)
        self.active = self._K = self._V = None

    def choose(self, items):
        context = max(it["pos"] + it["max_new"] for it in items)
        choices = [v for v in self.variants if v.B >= len(items) and v.S >= context]
        if not choices:
            raise ValueError("no prepared graph fits the batch and context")
        return min(choices, key=lambda v: (v.B * v.S, v.S, v.B))

    def activate(self, variant):
        if variant not in self.variants:
            raise ValueError("variant was not prepared at startup")
        info = self.engine.swap(variant)
        self.active = variant
        self._K, self._V = self.engine.blank_kv()
        return info

    def write_slot(self, idx, K, V):
        v = self.active
        if (v is None or not 0 <= idx < v.B
                or len(K.shape) != 4 or len(V.shape) != 4
                or tuple(K.shape[:3]) != (self.nl, self.nkv, self.hd)
                or tuple(V.shape[:2]) != (self.nl, self.nkv)
                or V.shape[3] != self.hd or K.shape[3] != V.shape[2]
                or K.shape[3] < v.S):
            raise ValueError("prefill KV layout does not fit the prepared engine")
        K, V = K[..., :v.S], V[:, :, :v.S, :]
        if self.kind == "atom":
            self._K[:, idx], self._V[:, idx] = K, V
            return
        e = self.engine
        dtype = e.BF if self.kind == "rngd" else self._K.tensor.dtype
        cast = lambda a: e.torch.as_tensor(a).to(dtype).to(e.dev)
        Kh, Vh = cast(K), cast(V)
        if self.kind == "atom-inputs":
            self._K.tensor[:, idx], self._V.tensor[:, idx] = Kh, Vh
        else:
            for g in range(e.ngroups):
                sl = slice(g * e.group, (g + 1) * e.group)
                self._K[g][:, idx], self._V[g][:, idx] = Kh[sl], Vh[sl]

    def step(self, toks, positions):
        nxt, self._K, self._V = self.engine.step(toks, positions, self._K, self._V)
        return nxt

    def discard_kv(self):
        self._K = self._V = None

    def warmup(self):
        """Run each prepared graph with private KV before admitting requests."""
        for variant in self.variants:
            try:
                self.activate(variant)
                self.step([0] * variant.B, [0] * variant.B)
            finally:
                self.discard_kv()


def build_prepared(path, S, chip):
    cfg = read_config(path, S)
    engine = adapter = None
    try:
        variants, kind = cfg["variants"], cfg["engine"]
        if kind == "rngd":
            from rngd_pd.swap.rngd import SwappableDecode, NKV, HD
            engine = SwappableDecode(cfg["checkpoint"], group=variants[0].group,
                                     chip=chip, cache_dir=cfg.get("cache_dir"),
                                     allow_compile=False, host_workers=cfg.get("host_workers", 1))
            for v in variants:
                engine.prepare(v)
            dims = engine.nl, NKV, HD
        elif kind == "atom":
            from rngd_pd.swap.atom import AtomSession, Tables
            engine = AtomSession(cfg["artifact"], Tables(cfg["checkpoint"]), device=chip)
            if any(not engine.has(v) for v in variants):
                raise ValueError("ATOM artifact does not contain every configured variant")
            if engine.nl != engine.tables.cfg.n_layers:
                raise ValueError("ATOM artifact layer count does not match the checkpoint")
            dims = engine.nl, engine.tables.cfg.n_kv, engine.tables.cfg.head_dim
        else:
            from rngd_pd.swap.atom_inputs import DeviceWeightDecode
            if len(variants) > 8:
                raise ValueError("ATOM device-input startup supports at most eight prepared shapes")
            if not cfg.get("cache_dir"):
                raise ValueError("ATOM device-input mode needs cache_dir")
            engine = DeviceWeightDecode(cfg["checkpoint"], cfg["cache_dir"], device=chip)
            for v in variants:
                engine.swap(v)
                engine.first_step()       # materialize every shape before ready
            dims = engine.d["nl"], engine.d["NKV"], engine.d["HD"]
        adapter = PreparedEngine(engine, variants, kind, cfg['checkpoint_id'], *dims)
        if cfg.get('warmup', False):
            adapter.warmup()
        return adapter
    except BaseException:
        if adapter is not None:
            adapter.discard_kv()
        if engine is not None:
            engine.close()
        raise
