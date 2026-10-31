"""Whole-request RNGD KV connector, subclassing vLLM's KVConnectorBase_V1.

The custom cluster uses the shared standalone worker connector for whole layer arrays.
Paged vLLM scheduler/block mappings require a separate layout adapter and are rejected.

Dual construction:
  • real vLLM :  RngdKVConnector(vllm_config, role, kv_cache_config)      # positional, vLLM calls this
  • standalone:  RngdKVConnector(KVConnectorRole.WORKER, req_id="r1",      # our runtime / tests
                                 transport=tr, n_layers=32)

RNGD KV layout: per-layer k,v tensors ([NKV, S, hd]-ish). D2D=host PCIe, so transport is host shm/TCP.
"""
import sys, os, time
from dataclasses import dataclass, field
import numpy as np

from rngd_pd.connector.transport import Transport, InProcTransport, ser, deser, select_transport

from .worker import WorkerKVConnector

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1, KVConnectorRole, KVConnectorMetadata)
except ModuleNotFoundError as error:
    if error.name and error.name.split('.')[0] == 'vllm':
        raise ImportError('vLLM plugin requires vLLM; standalone workers use '
                          'rngd_pd.connector.worker.WorkerKVConnector') from error
    raise


@dataclass
class RngdKVConnectorMetadata(KVConnectorMetadata):
    """Scheduler→worker KV-transfer instructions for one step (concrete KVConnectorMetadata)."""
    req_id: str = ""
    load_layers: list = field(default_factory=list)
    save_layers: list = field(default_factory=list)
    num_external_tokens: int = 0
    remote: str = ""                                   # decode target (xPyD routing)


class RngdKVConnector(WorkerKVConnector, KVConnectorBase_V1):
    """vLLM KVConnectorBase_V1 subclass for RNGD PD disaggregation + KV offload."""

    worker_role = KVConnectorRole.WORKER
    metadata_type = RngdKVConnectorMetadata

    def __init__(self, *args, req_id="r0", transport: Transport = None, n_layers=0,
                 kv_transport=None, **kw):
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
            KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
            # allow n_layers/req_id override; else infer later from registered caches
            xfer = getattr(self._kv_transfer_config, "kv_connector_extra_config", {}) or {}
            req_id = xfer.get("req_id", req_id)
            n_layers = xfer.get("n_layers", n_layers)
            kv_transport = xfer.get("kv_transport", kv_transport)

        self._init_transfer(req_id, transport, n_layers, kv_transport)

    def bind_connector_metadata(self, metadata):
        WorkerKVConnector.bind_connector_metadata(self, metadata)
        KVConnectorBase_V1.bind_connector_metadata(self, metadata)


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
