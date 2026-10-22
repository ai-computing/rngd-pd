"""Delta load on ATOM with the public rebel API only.

On ATOM a separately compiled graph cannot attach to weights another graph has
already put on the card (rebel 0.11 has no public call for it; see the README).
What the public API does provide is *bucketing*: `rebel.compile_from_torch`
with a list of input configurations compiles one model with one graph per
input shape, served by one runtime that "selects the bucket from the input
shape" (docs.rbln.ai, Bucketing). So the variants go into one compile session:

  build_session()   one compile, one graph per (B, S) variant, weights once;
                    runs without a card (npu="RBLN-CA22");
  AtomSession       loads it; swap(v) only changes the shape of the next call,
                    so the runtime picks v's graph;
  HostPool          the fast reload for a variant outside the running session:
                    the next session's .rbln is read and parsed on the host
                    (RBLNCompiledModel, the slow part of a reload) while the
                    current one serves; the switch then only destroys the old
                    runtime and creates the new one;
  accumulating      a variant outside the session triggers build_session() with
                    the variant added (a card-free compile), then a HostPool
                    switch at a quiet moment; later swaps inside it are free.

The decode graph is rngd-pd's batched ATOM decode (vendors/rbln_decode_batch:
fp16 weights on the card, host KV in and the step's k/v rows out), with the
batch width and sequence length read from the inputs so one module traces at
every bucket.
"""
import json
import os
import threading
import time

import numpy as np

H_DEFAULT = 4096


def _now():
    return time.perf_counter()


def load_checkpoint(model_dir, layers=None):
    """(cfg namespace, {name: float32 numpy}) straight from safetensors; no embedding."""
    from types import SimpleNamespace
    from safetensors import safe_open
    from rngd_pd.engines.model import ModelConfig
    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    c.setdefault("head_dim", None)
    cfg = ModelConfig(SimpleNamespace(**c))
    nl = layers or cfg.n_layers
    sd = {}
    for fn in sorted(os.listdir(model_dir)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.layers.") and int(k.split(".")[2]) >= nl:
                    continue
                sd[k] = f.get_tensor(k).float().numpy()
    return cfg, sd, nl


def decode_module(cfg, sd, nl):
    """The batched ATOM decode with B and S taken from the inputs."""
    import torch
    from rngd_pd.vendors.rbln_decode_batch import LlamaDecodeBatchRBLN

    class ShapeFree(LlamaDecodeBatchRBLN):
        def _rope(self, x, nh, cos, sin):
            B, half = x.shape[0], self.HD // 2
            xr = x.view(B, nh, self.HD)
            a, b = xr[..., :half], xr[..., half:]
            cos = cos.view(B, 1, half)
            sin = sin.view(B, 1, half)
            return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

        def forward(self, x, Kp, Vp, cos, sin, mask, oh):
            self.B, self.S = x.shape[0], mask.shape[1]
            return super().forward(x, Kp, Vp, cos, sin, mask, oh)

    return ShapeFree(cfg, sd, 512, nl, 1).eval()


def input_info(cfg, nl, B, S):
    H, NKV, HD = cfg.hidden, cfg.n_kv, cfg.head_dim
    f = "float32"
    return [("x", [B, H], f), ("Kp", [nl, B, NKV, HD, S], f), ("Vp", [nl, B, NKV, S, HD], f),
            ("cos", [B, HD // 2], f), ("sin", [B, HD // 2], f), ("mask", [B, S], f),
            ("oh", [B, S], f)]


def build_session(model_dir, out_path, variants, layers=None,
                  npu=os.environ.get("RBLN_PART", "RBLN-CA22")):
    """Compile every variant into one .rbln (bucketing). No card needed."""
    import rebel
    import torch
    torch.set_grad_enabled(False)
    t0 = _now()
    cfg, sd, nl = load_checkpoint(model_dir, layers)
    mod = decode_module(cfg, sd, nl)
    del sd
    t1 = _now()
    infos = [input_info(cfg, nl, v.B, v.S) for v in variants]
    cm = rebel.compile_from_torch(mod, input_info=infos, npu=npu)
    t2 = _now()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cm.save(out_path)
    meta = dict(model=os.path.basename(os.path.normpath(model_dir)), layers=nl,
                variants=[v.name for v in variants], compiler=rebel.__version__,
                load_s=t1 - t0, compile_s=t2 - t1, save_s=_now() - t2,
                device_alloc=cm.get_total_device_alloc(),
                file_bytes=os.path.getsize(out_path))
    with open(out_path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


class Tables:
    """Embedding and per-position inputs (host side)."""

    def __init__(self, model_dir, cfg=None):
        from safetensors import safe_open
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            where = json.load(f)["weight_map"]
        name = "model.embed_tokens.weight"
        with safe_open(os.path.join(model_dir, where[name]), framework="pt") as f:
            self.embed = f.get_tensor(name).float().numpy()
        if cfg is None:
            from types import SimpleNamespace
            from rngd_pd.engines.model import ModelConfig
            with open(os.path.join(model_dir, "config.json")) as f:
                c = json.load(f)
            c.setdefault("head_dim", None)
            cfg = ModelConfig(SimpleNamespace(**c))
        self.cfg = cfg
        self.inv = cfg.inv_freq()

    def step_inputs(self, toks, pos, S):
        pos = np.asarray(pos)
        ang = pos.astype(np.float32)[:, None] * self.inv[None, :]
        mask = np.where(np.arange(S)[None, :] <= pos[:, None], 0.0, -1e4).astype(np.float32)
        oh = (np.arange(S)[None, :] == pos[:, None]).astype(np.float32)
        return (self.embed[list(toks)].astype(np.float32), np.cos(ang).astype(np.float32),
                np.sin(ang).astype(np.float32), mask, oh)


class AtomSession:
    """One loaded session (.rbln with several buckets) on one ATOM device."""

    def __init__(self, path, tables, device=0, compiled=None, variants=None):
        import rebel
        self.path, self.tables, self.device = path, tables, device
        t0 = _now()
        self.compiled = compiled or rebel.RBLNCompiledModel(path)
        t1 = _now()
        self.rt = self.compiled.create_runtime(device=device, tensor_type="np")
        t2 = _now()
        meta = {}
        if os.path.exists(path + ".json"):
            with open(path + ".json") as f:
                meta = json.load(f)
        self.nl = meta.get("layers", 32)
        from rngd_pd.swap.variants import Variant
        self.variants = variants or [Variant.parse(n, 0) for n in meta.get("variants", [])]
        self.active = self.variants[0] if self.variants else None
        self.timings = dict(parse_s=t1 - t0, create_runtime_s=t2 - t1,
                            parsed_before=compiled is not None)

    def close(self):
        self.rt = None
        self.compiled = None

    def has(self, v):
        return any(x.B == v.B and x.S == v.S for x in self.variants)

    def swap(self, v):
        """Select v's graph: the next call has its shapes. Nothing is loaded."""
        t0 = _now()
        if not self.has(v):
            raise KeyError(f"{v.name} is not in this session ({[x.name for x in self.variants]})")
        self.active = v
        return dict(swap_s=_now() - t0, variant=v.name, in_session=True)

    def blank_kv(self):
        c, v = self.tables.cfg, self.active
        return (np.zeros((self.nl, v.B, c.n_kv, c.head_dim, v.S), np.float32),
                np.zeros((self.nl, v.B, c.n_kv, v.S, c.head_dim), np.float32))

    def step(self, toks, pos, K, V):
        v = self.active
        x, cos, sin, mask, oh = self.tables.step_inputs(toks, pos, v.S)
        out = self.rt.run(x, K, V, cos, sin, mask, oh)
        logits, k, vv = (np.asarray(o) for o in out[:3])
        for b, p in enumerate(pos):
            K[:, b, :, :, p] = k[:, b]
            V[:, b, :, p, :] = vv[:, b]
        return [int(t) for t in np.argmax(logits, axis=-1)], K, V

    def first_step(self):
        v = self.active
        t0 = _now()
        K, V = self.blank_kv()
        self.step([1] * v.B, [0] * v.B, K, V)
        return _now() - t0

    def generate(self, prompts, n_new):
        v = self.active
        assert len(prompts) == v.B and len({len(p) for p in prompts}) == 1
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


def warm_page_cache(path, block=64 << 20):
    """Read a file once so the next open is served from the page cache."""
    t0 = _now()
    with open(path, "rb", buffering=0) as f:
        while f.read(block):
            pass
    return _now() - t0


class HostPool:
    """Prepare the next session on the host while the current one serves.

    prepare(path) reads and parses the .rbln in a background thread
    (RBLNCompiledModel: host memory only, the card is not touched). switch()
    then closes the running session and creates the runtime from the parsed
    model: the device part of a reload only. The card never holds both, so the
    cost is host memory (about the .rbln size), not device memory.
    """

    def __init__(self):
        self._thread = None
        self._result = None

    def prepare(self, path):
        def work():
            import rebel
            t0 = _now()
            try:
                cm = rebel.RBLNCompiledModel(path)
                self._result = dict(path=path, compiled=cm, parse_s=_now() - t0)
            except Exception as e:  # noqa: BLE001 - reported by switch()
                self._result = dict(path=path, error=e)
        self._result = None
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def ready(self):
        return self._thread is not None and not self._thread.is_alive()

    def switch(self, current, tables, device=0):
        """Close `current` and start the prepared session. Returns (session, timings)."""
        t0 = _now()
        self._thread.join()
        waited = _now() - t0
        r = self._result
        if "error" in r:
            raise r["error"]
        t1 = _now()
        if current is not None:
            current.close()
            import gc
            gc.collect()
        t2 = _now()
        sess = AtomSession(r["path"], tables, device=device, compiled=r["compiled"])
        t3 = _now()
        return sess, dict(wait_prepare_s=waited, close_s=t2 - t1, create_s=t3 - t2,
                          parse_in_background_s=r["parse_s"])

# Optimum-rbln bundle API retained alongside native rebel sessions.
from .atom_bundles import AtomBundle, SwappableAtom  # noqa: F401,E402
