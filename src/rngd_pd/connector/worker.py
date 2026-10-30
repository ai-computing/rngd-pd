"""Whole-request KV transfer for standalone workers, independent of vLLM."""
import os
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .transport import InProcTransport, ser, deser, select_transport


class WorkerRole(Enum):
    WORKER = 'worker'


@dataclass
class WorkerMetadata:
    req_id: str = ''
    load_layers: list = field(default_factory=list)
    save_layers: list = field(default_factory=list)
    num_external_tokens: int = 0
    remote: str = ''


class WorkerKVConnector:
    """Transfer whole layer arrays; the optional vLLM adapter reuses this code."""
    worker_role = WorkerRole.WORKER
    metadata_type = WorkerMetadata

    def __init__(self, role=WorkerRole.WORKER, req_id='r0', transport=None,
                 n_layers=0, kv_transport=None):
        if role is not WorkerRole.WORKER:
            raise ValueError('standalone connector requires WorkerRole.WORKER')
        self._role = role
        self._connector_metadata = None
        self._init_transfer(req_id, transport, n_layers, kv_transport)

    def _init_transfer(self, req_id, transport, n_layers, kv_transport):
        self.req_id, self.n_layers = req_id, n_layers
        if transport is None and (kv_transport or os.environ.get('AIRM_KV_TRANSPORT')):
            transport = select_transport(kv_transport)
        self.t = transport or InProcTransport()
        self._kv_caches = None
        self._saved, self._loading = {}, []
        self._pending_alloc = None
        self._loading_req_id = req_id

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
        if blocks is not None:
            raise NotImplementedError("rngd connector: paged block mappings require a layout adapter")
        req_id = getattr(request, "request_id", None)
        if not isinstance(req_id, str) or not req_id:
            raise ValueError("rngd connector: request_id must be a nonempty string")
        if type(num_external_tokens) is not int or num_external_tokens < 0:
            raise ValueError("rngd connector: num_external_tokens must be a nonnegative integer")
        if self._pending_alloc is not None:
            raise NotImplementedError("rngd connector: only one request per metadata step is supported")
        self._pending_alloc = (req_id, num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        if scheduler_output is not None:
            raise NotImplementedError("rngd connector: vLLM paged scheduler output requires a layout adapter")
        layers = [f"layer.{i}" for i in range(self.n_layers)]
        if self._pending_alloc is not None:
            req_id, num_external_tokens = self._pending_alloc
            self._pending_alloc = None
            load_layers = layers if num_external_tokens else []
            save_layers = [] if num_external_tokens else layers
        elif self._role == self.worker_role:
            req_id, num_external_tokens = self.req_id, 0
            load_layers = save_layers = layers
        else:
            req_id, num_external_tokens = "", 0
            load_layers, save_layers = [], []
        return self.metadata_type(
            req_id=req_id, save_layers=save_layers, load_layers=load_layers,
            num_external_tokens=num_external_tokens)

    # ================= worker-side =================
    def bind_connector_metadata(self, metadata):
        if not isinstance(metadata, self.metadata_type):
            raise TypeError("rngd connector: expected whole-request RngdKVConnectorMetadata")
        if (not isinstance(metadata.req_id, str) or
                (not metadata.req_id and (metadata.load_layers or metadata.save_layers))):
            raise ValueError("rngd connector: metadata requires a single request_id")
        if type(metadata.num_external_tokens) is not int or metadata.num_external_tokens < 0:
            raise ValueError("rngd connector: num_external_tokens must be a nonnegative integer")
        self._connector_metadata = metadata

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
        if attn_metadata is not None:
            raise NotImplementedError("rngd connector: paged attention metadata requires a layout adapter")
        meta = self._connector_metadata
        if meta is not None and layer_name not in meta.save_layers:
            return
        req_id = meta.req_id if meta is not None else self.req_id
        k, v = self._split_kv(kv_layer)
        self.t.push(f"{req_id}/{layer_name}/k", ser(k))
        self.t.push(f"{req_id}/{layer_name}/v", ser(v))
        self._saved[layer_name] = True

    def wait_for_save(self):
        """Barrier until all pushes are durable. shm/TCP push is synchronous → already done."""
        return

    def start_load_kv(self, forward_context=None, **kwargs):
        """DECODE/onload: begin loading all layers named in bound metadata."""
        if forward_context is not None:
            raise NotImplementedError("rngd connector: paged forward context requires a layout adapter")
        meta = self._connector_metadata
        self._loading_req_id = meta.req_id if meta is not None else self.req_id
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
        if self._connector_metadata is not None and layer_name not in self._loading:
            return
        t = float(os.environ.get("PD_KV_PULL_TIMEOUT", "5"))
        k = deser(self.t.pull(f"{self._loading_req_id}/{layer_name}/k", timeout=t))
        v = deser(self.t.pull(f"{self._loading_req_id}/{layer_name}/v", timeout=t))
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
