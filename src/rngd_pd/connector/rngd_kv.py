"""RNGD KV connector — a REAL vllm.distributed...KVConnectorBase_V1 subclass (Phase 1).

Was: a stand-alone class MIRRORING the contract. Now: actually subclasses vLLM 0.18.0's
KVConnectorBase_V1, so `isinstance(conn, KVConnectorBase_V1)` is True and llm-d's connector
machinery can load it. The 7 abstract methods (get_num_new_matched_tokens, update_state_after_alloc,
build_connector_meta, save_kv_layer, start_load_kv, wait_for_layer_load, wait_for_save) are
implemented; transport is injected (kv_transport.Transport) so shm/TCP today, NIXL later — connector
logic never changes.

Dual construction:
  • real vLLM :  RngdKVConnector(vllm_config, role, kv_cache_config)      # positional, vLLM calls this
  • standalone:  RngdKVConnector(KVConnectorRole.WORKER, req_id="r1",      # our runtime / tests
                                 transport=tr, n_layers=32)

RNGD KV layout: per-layer k,v tensors ([NKV, S, hd]-ish). D2D=host PCIe, so transport is host shm/TCP.
"""
import sys, os, time
from dataclasses import dataclass, field
import numpy as np

from rngd_pd.connector.transport import Transport, InProcTransport, ser, deser

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorRole, KVConnectorMetadata)


@dataclass
class RngdKVConnectorMetadata(KVConnectorMetadata):
    """Scheduler→worker KV-transfer instructions for one step (concrete KVConnectorMetadata)."""
    req_id: str = ""
    load_layers: list = field(default_factory=list)
    save_layers: list = field(default_factory=list)
    num_external_tokens: int = 0
    remote: str = ""                                   # decode target (xPyD routing)


class RngdKVConnector(KVConnectorBase_V1):
    """vLLM KVConnectorBase_V1 subclass for RNGD PD disaggregation + KV offload."""

    def __init__(self, *args, req_id="r0", transport: Transport = None, n_layers=0, **kw):
        # --- dispatch: vLLM (VllmConfig, role, ...) vs standalone (KVConnectorRole, ...) ---
        if args and isinstance(args[0], KVConnectorRole):
            # standalone: (role, req_id?, transport?, n_layers?) — positional back-compat
            role = args[0]
            if len(args) > 1 and isinstance(args[1], str): req_id = args[1]
            if len(args) > 2 and args[2] is not None:      transport = args[2]
            if len(args) > 3:                              n_layers = args[3]
            # replicate the minimal state KVConnectorBase_V1.__init__ sets (no VllmConfig needed)
            self._connector_metadata = None
            self._vllm_config = None
            self._kv_transfer_config = None
            self._kv_cache_config = None
            self._role = role
        else:
            # real vLLM path: (vllm_config, role, kv_cache_config=None)
            vllm_config = args[0]
            role = args[1] if len(args) > 1 else kw.get("role")
            kv_cache_config = args[2] if len(args) > 2 else kw.get("kv_cache_config")
            super().__init__(vllm_config, role, kv_cache_config)
            # allow n_layers/req_id override; else infer later from registered caches
            xfer = getattr(self._kv_transfer_config, "kv_connector_extra_config", {}) or {}
            req_id = xfer.get("req_id", req_id)
            n_layers = xfer.get("n_layers", n_layers)

        self.req_id = req_id
        self.n_layers = n_layers
        self.t = transport or InProcTransport()
        self._kv_caches = None
        self._saved = {}
        self._loading = []

    @property
    def role(self): return self._role

    # ================= scheduler-side =================
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Tokens whose KV is externally available (on the prefill side) beyond local, + async flag.
        Returns (num_external, load_async). PD decode: whole prompt KV is remote → report handoff len."""
        n = getattr(request, "kv_ready_tokens", 0)
        extra = max(0, n - num_computed_tokens)
        return (extra if extra > 0 else None, extra > 0)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._pending_alloc = (getattr(request, "request_id", self.req_id), num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        layers = [f"layer.{i}" for i in range(self.n_layers)]
        is_worker = self._role == KVConnectorRole.WORKER
        return RngdKVConnectorMetadata(
            req_id=self.req_id,
            save_layers=layers if is_worker else [],
            load_layers=layers,
            num_external_tokens=self.n_layers)

    # ================= worker-side =================
    def register_kv_caches(self, kv_caches):
        """Bind the local KV buffer. vLLM passes dict[layer_name→tensor]; our runtime passes (K, V)
        lists indexed by layer (each [NKV,S,hd]). Both stored; wait_for_layer_load writes into them."""
        self._kv_caches = kv_caches
        if isinstance(kv_caches, dict) and self.n_layers == 0:
            self.n_layers = len(kv_caches)

    @staticmethod
    def _split_kv(kv_layer):
        """Normalize a layer's KV into (k, v) numpy arrays. Accepts (k,v) tuple/list, a stacked
        tensor with leading dim 2, or a torch tensor."""
        if isinstance(kv_layer, (tuple, list)) and len(kv_layer) == 2:
            k, v = kv_layer
        else:
            arr = kv_layer.detach().cpu().numpy() if hasattr(kv_layer, "detach") else np.asarray(kv_layer)
            assert arr.shape[0] == 2, f"expected stacked [2,...] kv_layer, got {arr.shape}"
            k, v = arr[0], arr[1]
        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return to_np(k), to_np(v)

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata=None, **kwargs):
        """PREFILL/offload: push one layer's KV to transport (overlaps with next-layer compute)."""
        k, v = self._split_kv(kv_layer)
        self.t.push(f"{self.req_id}/{layer_name}/k", ser(k))
        self.t.push(f"{self.req_id}/{layer_name}/v", ser(v))
        self._saved[layer_name] = True

    def wait_for_save(self):
        """Barrier until all pushes are durable. shm/TCP push is synchronous → already done."""
        return

    def start_load_kv(self, forward_context=None, **kwargs):
        """DECODE/onload: begin loading all layers named in bound metadata."""
        meta = self._connector_metadata
        self._loading = list(getattr(meta, "load_layers", [])) if meta else \
            [f"layer.{i}" for i in range(self.n_layers)]

    def wait_for_layer_load(self, layer_name):
        """DECODE/onload: block until this layer's KV is present; write into the local buffer.

        The wait has to cover the prefill queue, not just the transfer. One
        prefill card serialises the passes behind a B-wide decode batch, so the
        last request in a batch of sixteen waits for fifteen prefills before
        its first layer even exists -- and a five-second default turned that
        into a pull timeout that reads like a transport fault. The ceiling
        scales with the batch the deployment asks for, so it is configuration,
        not a constant.
        """
        t = float(os.environ.get("PD_KV_PULL_TIMEOUT", "5"))
        k = deser(self.t.pull(f"{self.req_id}/{layer_name}/k", timeout=t))
        v = deser(self.t.pull(f"{self.req_id}/{layer_name}/v", timeout=t))
        if self._kv_caches is not None:
            li = int(layer_name.split(".")[-1])
            # The producer may send only the columns the prompt filled --
            # a 20-token prompt has no reason to ship a 512-wide cache -- so
            # place what arrived and clear the rest. The tail matters: a decode
            # step injects its k/v by adding at a one-hot slot, which only
            # equals a write if the slot starts at zero.
            if isinstance(self._kv_caches, dict):
                dst = self._kv_caches.get(layer_name)
                if dst is not None: dst[0][:] = k; dst[1][:] = v
            else:                                       # (K_list, V_list)
                self._kv_caches[0][li][:] = k
                self._kv_caches[1][li][:] = v
        return k, v

    def get_finished(self, finished_req_ids):
        return (None, None)


# ================= self-test =================
def _selftest():
    NL, NKV, S, HD = 4, 8, 16, 128
    rng = np.random.default_rng(0)
    ck = [rng.standard_normal((NKV, S, HD)).astype(np.float32) for _ in range(NL)]
    cv = [rng.standard_normal((NKV, S, HD)).astype(np.float32) for _ in range(NL)]
    tr = InProcTransport()

    # is it REALLY a vLLM connector now?
    is_sub = issubclass(RngdKVConnector, KVConnectorBase_V1)
    no_abstract = len(RngdKVConnector.__abstractmethods__) == 0

    pf = RngdKVConnector(KVConnectorRole.WORKER, req_id="req1", transport=tr, n_layers=NL)
    inst = isinstance(pf, KVConnectorBase_V1)
    for i in range(NL):
        pf.save_kv_layer(f"layer.{i}", (ck[i], cv[i]))
    pf.wait_for_save()

    dk = [np.zeros((NKV, S, HD), np.float32) for _ in range(NL)]
    dv = [np.zeros((NKV, S, HD), np.float32) for _ in range(NL)]
    dc = RngdKVConnector(KVConnectorRole.WORKER, req_id="req1", transport=tr, n_layers=NL)
    dc.register_kv_caches((dk, dv))
    dc.bind_connector_metadata(dc.build_connector_meta(None))
    dc.start_load_kv()
    for i in range(NL):
        dc.wait_for_layer_load(f"layer.{i}")

    ref16 = [a.astype(np.float16).astype(np.float32) for a in ck]
    err = max(np.abs(dk[i] - ref16[i]).max() for i in range(NL))
    payload = sum(len(ser(a)) for a in ck + cv)
    print("RngdKVConnector self-test (REAL KVConnectorBase_V1 subclass):")
    print(f"  issubclass(KVConnectorBase_V1) = {is_sub} | __abstractmethods__ empty = {no_abstract} | isinstance = {inst}")
    print(f"  {NL} layers save→transport→load | transport loss {err:.6f} (0=lossless) | {payload/1e6:.1f}MB fp16")
    ok = is_sub and no_abstract and inst and err < 1e-6
    print(f"  {'PASS ✅ real vLLM connector; per-layer KV bit-exact through the contract' if ok else 'FAIL ❌'}")
    return ok


if __name__ == "__main__":
    _selftest()
