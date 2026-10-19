"""`nixl` KV transport: per-request KV through NIXL and hetero-npu-p2p's HETERO_NPU backend.

Entry point `airm.kv_transports` name `nixl` (rm-core KVTransport). The bytes move between live
allocations the owning process binds per request, on one host, through one NIXL agent:

  route            source             destination                 data path
  rngd -> dmabuf   RNGD device DRAM   exported ATOM DMA-BUF       direct PCIe P2P (PDMA DMABUF_READ)
  rngd -> host     RNGD device DRAM   pinned host memory          explicit staging leg (SCATTER_READ)
  host -> rngd     pinned host memory RNGD device DRAM            explicit staging leg (SCATTER_WRITE)

RNGD -> RNGD is refused: the backend has no validated device-to-device route between two RNGD
cards, so a caller that wants it plans the two staging legs itself (rngd -> host, then host -> rngd)
and sees both in its timings. Nothing here picks a route on its own or falls back to another path
(rm-core PROTOCOL §4.5): an unsupported pair raises TransportError at plan().

The layout is rngd-pd's: K [layers, kv_heads, head_dim, S] and V [layers, kv_heads, S, head_dim] in
bf16, S-wide (the decode side keeps an S-wide cache, so the producer ships the padded columns too).
The transport copies bytes and never converts; both sides must hold the same layout.

Limits, by construction of the backend (docs/nixl-backend.md in hetero-npu-p2p): one agent, one
host. The process that owns the agent must see the RNGD allocation (`tensor.data_ptr()` of a
Furiosa tensor, the card's PE0 device path) and hold the DMA-BUF fds of the ATOM destination
(passed with SCM_RIGHTS when another process allocated them). After a fence the ATOM owner must call
`rebel._C.vmem.mark_device_memory_is_updated(ptr)` before it reads the cache.
"""
import collections
import os
import threading
import time
import uuid
from dataclasses import dataclass

from rm_core import (ContractError, Measurement, TransferHandle, TransferPlan, TransferResult,
                     TransportError)

BACKEND = "HETERO_NPU"
COMPONENT = "rngd-pd"
DEVICE_KINDS = ("device-dram", "device-hbm")
ROUTES = {("rngd", "dmabuf"): "direct", ("rngd", "host"): "staged", ("host", "rngd"): "staged"}
WIRE_DTYPES = ("bf16",)


def _vendor(tier):
    """'rngd' | 'dmabuf' | 'host' for a tier, from its kind and accelerator id (`node/vendor:i`)."""
    if tier.kind == "host-dram":
        return "host", ""
    if tier.kind not in DEVICE_KINDS:
        return "", ""
    node, _, dev = tier.accelerator_id.partition("/")
    vendor = dev.split(":", 1)[0]
    return {"furiosa": "rngd", "rebellions": "dmabuf"}.get(vendor, ""), node


@dataclass(frozen=True)
class Region:
    """One registered allocation. `kind` rngd | dmabuf | host.

    rngd:   address = raw DRAM offset (Furiosa tensor data_ptr()), device = /dev/rngd/npuNch0
    dmabuf: address = the transfer base (e.g. the ATOM DVA + offset), fd = the exported DMA-BUF,
            offset = byte offset of the tensor inside the export
    host:   address = virtual address (page aligned)
    """
    kind: str
    address: int
    size: int
    device: str = ""
    fd: int = -1
    offset: int = 0

    def validate(self, align):
        if self.kind not in ("rngd", "dmabuf", "host"):
            raise TransportError(f"nixl: unknown region kind {self.kind!r}")
        if self.size <= 0 or self.address <= 0 and self.kind != "dmabuf":
            raise TransportError(f"nixl: {self.kind} region needs a positive address and size")
        if self.address % align:
            raise TransportError(f"nixl: {self.kind} region address {self.address:#x} is not {align}-byte aligned")
        if self.kind == "rngd" and not (self.device.startswith("/dev/rngd/npu") and self.device.endswith("ch0")):
            raise TransportError("nixl: an RNGD region needs its PE0 device path /dev/rngd/npuNch0")
        if self.kind == "dmabuf" and self.fd < 0:
            raise TransportError("nixl: a DMA-BUF region needs its fd")
        return self


def rngd_region(tensor_or_address, size, device):
    """Region of a live Furiosa allocation (a tensor, or its data_ptr())."""
    addr = tensor_or_address.data_ptr() if hasattr(tensor_or_address, "data_ptr") else int(tensor_or_address)
    return Region("rngd", addr, int(size), device=device)


def dmabuf_region(fd, size, base, offset=0):
    return Region("dmabuf", int(base), int(size), fd=int(fd), offset=int(offset))


def host_region(address, size):
    return Region("host", int(address), int(size))


@dataclass(frozen=True)
class NixlBinding:
    """The live allocations of one request: source and destination regions, each side viewed as one
    buffer (regions concatenated), and the context S the cache is laid out for."""
    src: tuple
    dst: tuple
    context: int


def _pieces(regions, length):
    """[(address, length, region index)] covering the first `length` bytes of the concatenation."""
    out, left = [], length
    for i, r in enumerate(regions):
        if left <= 0:
            break
        n = min(r.size, left)
        out.append((r.address, n, i))
        left -= n
    if left > 0:
        raise TransportError(f"nixl: regions hold {length - left} bytes, plan moves {length}")
    return out


def paired(src_regions, dst_regions, length):
    """Equal-length (src, dst) descriptor lists: NIXL pairs the i-th of each, so cut wherever
    either side crosses a region boundary."""
    src, dst = _pieces(src_regions, length), _pieces(dst_regions, length)
    out_s, out_d = [], []
    while src:
        (sa, sn, si), (da, dn, di) = src[0], dst[0]
        n = min(sn, dn)
        out_s.append((sa, n, si)); out_d.append((da, n, di))
        src[0], dst[0] = (sa + n, sn - n, si), (da + n, dn - n, di)
        if src[0][1] == 0:
            src.pop(0)
        if dst[0][1] == 0:
            dst.pop(0)
    return out_s, out_d


def _default_agent(params):
    from nixl._api import nixl_agent, nixl_agent_config
    agent = nixl_agent(f"rngd-pd-{os.getpid()}-{uuid.uuid4().hex[:6]}", nixl_agent_config(backends=[]))
    agent.create_backend(BACKEND, params)
    return agent


class NixlTransport:
    """RNGD KV to an ATOM DMA-BUF (direct) or to/from host memory (staging legs), fenced per request.

    Keyword arguments (all optional): `agent` a nixl_agent with HETERO_NPU created (tests pass a
    fake), `queue_depth` (4), `desc_bytes` (2 MiB), `descs_per_cmd` (8), `align` (4096): the
    backend's parameters; `poll_s` sleep between completion polls (0: spin); `metric_sink` a
    callable receiving each Measurement. The NIXL agent is created on first use, so constructing
    the transport (as the conformance runner does) touches no device.
    """

    name = "nixl"

    def __init__(self, agent=None, queue_depth=4, desc_bytes=2 << 20, descs_per_cmd=8, align=4096,
                 poll_s=0.0, metric_sink=None):
        if not 1 <= int(queue_depth) <= 64 or int(desc_bytes) % int(align):
            raise ContractError("nixl: queue_depth 1..64 and desc_bytes a multiple of align")
        self.params = {"queue_depth": str(queue_depth), "desc_bytes": str(desc_bytes),
                       "descs_per_cmd": str(descs_per_cmd), "align": str(align), "allow_host": "1"}
        self.align, self.poll_s, self.metric_sink = int(align), float(poll_s), metric_sink
        self._agent = agent
        self._regs = {}                     # Region -> (registration handle, mem type, dev_id)
        self._dev_ids = {}                  # (kind, device or fd) -> dev_id
        self._bindings, self._plans, self._pending = {}, {}, {}
        self._lock = threading.Lock()
        self.measurements = collections.deque(maxlen=1024)

    # -- rm-core KVTransport ----------------------------------------------
    def supports(self, src, dst):
        (sv, sn), (dv, dn) = _vendor(src), _vendor(dst)
        if (sv, dv) not in ROUTES:
            return False
        nodes = {n for n in (sn, dn) if n}
        return len(nodes) <= 1                  # same host; host tiers carry no node

    def route(self, src, dst):
        return ROUTES.get((_vendor(src)[0], _vendor(dst)[0]))

    def bind(self, request_id, binding):
        if not isinstance(binding, NixlBinding):
            raise TransportError("nixl: bind() needs a NixlBinding")
        with self._lock:
            self._bindings[request_id] = binding

    def plan(self, request_id, src, dst, src_layout, dst_layout, tokens):
        with self._lock:                        # a binding is one-shot, even for a rejected plan
            binding = self._bindings.pop(request_id, None)
        sv, dv = _vendor(src)[0], _vendor(dst)[0]
        if not self.supports(src, dst):
            hint = (" (RNGD -> RNGD has no validated device route: plan rngd -> host and host -> rngd)"
                    if (sv, dv) == ("rngd", "rngd") else "")
            raise TransportError(f"nixl: no HETERO_NPU route {src.name} ({src.kind}) -> {dst.name} ({dst.kind}){hint}")
        src.validate(); dst.validate(); src_layout.validate(); dst_layout.validate()
        if src_layout != dst_layout:
            raise TransportError("nixl: src and dst layouts differ; the transport copies bytes, convert in the engine")
        if src_layout.dtype not in WIRE_DTYPES:
            raise TransportError(f"nixl: rngd-pd caches are {WIRE_DTYPES}, not {src_layout.dtype}")
        start, end = tokens
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise TransportError(f"nixl: bad token range {tokens}")
        if start != 0:
            raise TransportError("nixl: only prefix transfers [0, end) (S-wide caches are shipped whole)")
        if binding is None:
            raise TransportError(f"nixl: no live allocation bound for {request_id!r}; call bind() first")
        if end > binding.context:
            raise TransportError(f"nixl: {end} tokens exceed the bound context {binding.context}")
        nbytes = binding.context * src_layout.bytes_per_token()
        kinds = {"src": sv, "dst": dv}
        for side, regions in (("src", binding.src), ("dst", binding.dst)):
            if not regions:
                raise TransportError(f"nixl: no {side} regions bound")
            for r in regions:
                r.validate(self.align)
                if r.kind != kinds[side]:
                    raise TransportError(f"nixl: {side} tier is {kinds[side]}, bound region is {r.kind}")
            if sum(r.size for r in regions) < nbytes:
                raise TransportError(f"nixl: {side} regions hold {sum(r.size for r in regions)} bytes, "
                                     f"the S={binding.context} cache is {nbytes}")
        if nbytes % self.align:
            raise TransportError(f"nixl: cache size {nbytes} is not a multiple of {self.align}")
        plan_id = uuid.uuid4().hex
        with self._lock:
            self._plans[plan_id] = binding
        return TransferPlan(request_id, src, dst, src_layout, dst_layout, (start, end), nbytes,
                            {"plan_id": plan_id, "route": ROUTES[(sv, dv)], "backend": BACKEND,
                             "src": sv, "dst": dv})

    def submit(self, plan):
        with self._lock:
            binding = self._plans.pop(plan.detail.get("plan_id", ""), None)
        if binding is None:
            raise TransportError(f"nixl: unknown or already-submitted plan for {plan.request_id!r}")
        agent = self._get_agent()
        sv, dv = plan.detail["src"], plan.detail["dst"]
        src_d, dst_d = paired(binding.src, binding.dst, plan.nbytes)
        src_d = [(a, n, self._register(binding.src[i])[2]) for a, n, i in src_d]
        dst_d = [(a, n, self._register(binding.dst[i])[2]) for a, n, i in dst_d]
        # The local list holds the side NIXL calls local: the source for WRITE (RNGD -> DMA-BUF,
        # host -> RNGD); for RNGD -> host the backend reads the device into local host memory.
        if (sv, dv) == ("rngd", "host"):
            op, local, ltype, remote, rtype = "READ", dst_d, "DRAM", src_d, "VRAM"
        else:
            op, local, remote = "WRITE", src_d, dst_d
            ltype, rtype = ("DRAM" if sv == "host" else "VRAM"), "VRAM"
        started = time.perf_counter()
        try:
            h = agent.initialize_xfer(op, agent.get_xfer_descs(local, ltype),
                                      agent.get_xfer_descs(remote, rtype), agent.name,
                                      plan.request_id.encode(), backends=[BACKEND])
            state = agent.transfer(h)
        except Exception as e:                  # refused in prepXfer/postXfer: nothing in flight
            raise TransportError(f"nixl: {type(e).__name__}: {e}") from e
        if state not in ("PROC", "DONE"):
            h.release()
            raise TransportError(f"nixl: transfer of {plan.request_id!r} not posted ({state})")
        token = uuid.uuid4().hex
        with self._lock:
            self._pending[token] = (plan, h, started)
        return TransferHandle(plan.request_id, self.name, token)

    def wait(self, handle, timeout_s):
        """The completion fence: returns once every DMA command of the request completed, or the
        request failed (then the destination is undefined and must not be read)."""
        with self._lock:
            entry = self._pending.pop(handle.token, None)
        if entry is None:
            raise TransportError(f"nixl: unknown or already-waited handle {handle.token}")
        plan, h, started = entry
        agent, deadline = self._get_agent(), time.perf_counter() + max(0.0, timeout_s)
        state = "PROC"
        try:
            while True:
                state = agent.check_xfer_state(h)
                if state != "PROC" or time.perf_counter() > deadline:
                    break
                if self.poll_s:
                    time.sleep(self.poll_s)
        finally:
            seconds = time.perf_counter() - started
            h.release()                          # waits for commands already submitted
        if state == "DONE":
            res = TransferResult(plan.request_id, True, plan.nbytes, seconds)
            self._report(res, plan.detail["route"])
            return res
        why = "timed out (released after the in-flight DMA drained)" if state == "PROC" else f"state {state}"
        return TransferResult(plan.request_id, False, 0, seconds, f"nixl: {why}")

    def close(self):
        with self._lock:
            pending, self._pending = list(self._pending.values()), {}
            regs, self._regs = list(self._regs.items()), {}
            self._bindings.clear(); self._plans.clear()
        for _plan, h, _t in pending:
            try:
                h.release()
            except Exception:
                pass
        if self._agent is not None:
            for _r, (reg, _mem, _dev) in regs:
                try:
                    self._agent.deregister_memory(reg, [BACKEND])
                except Exception:
                    pass

    # -- internals -----------------------------------------------------------
    def _get_agent(self):
        if self._agent is None:
            try:
                self._agent = _default_agent(self.params)
            except Exception as e:
                raise TransportError(f"nixl: no NIXL agent with {BACKEND} ({type(e).__name__}: {e}); "
                                     "install nixl and set NIXL_PLUGIN_DIR to the built plugin") from e
        return self._agent

    def _register(self, r):
        """Register a region once; returns (handle, mem type, dev_id). Two VRAM kinds never share a
        dev_id, so overlapping RNGD offsets and ATOM DVAs stay apart (NIXL looks up by dev_id + range)."""
        with self._lock:
            got = self._regs.get(r)
            if got is not None:
                return got
            key = (r.kind, r.device if r.kind == "rngd" else r.fd if r.kind == "dmabuf" else "")
            dev = self._dev_ids.setdefault(key, len(self._dev_ids))
        if r.kind == "rngd":
            desc, mem = (r.address, r.size, dev, f"rngd:{r.device}"), "VRAM"
        elif r.kind == "dmabuf":
            desc, mem = (r.address, r.size, dev, f"dmabuf:{r.fd}:{r.offset}"), "VRAM"
        else:
            desc, mem = (r.address, r.size, dev, ""), "DRAM"
        try:
            reg = self._get_agent().register_memory([desc], mem, [BACKEND])
        except Exception as e:
            raise TransportError(f"nixl: registering a {r.kind} region failed: {type(e).__name__}: {e}") from e
        with self._lock:
            self._regs[r] = (reg, mem, dev)
        return reg, mem, dev

    def _report(self, res, route):
        tags = {"component": COMPONENT, "kv_transport": self.name}
        rows = [Measurement("kv_transfer_ms", res.seconds * 1e3, tags)]
        if res.seconds > 0:
            rows.append(Measurement("kv_transfer_gbps", res.nbytes / res.seconds / 1e9, tags))
        for m in rows:
            self.measurements.append(m.validate())
            if self.metric_sink:
                try:
                    self.metric_sink(m)
                except Exception:
                    pass                         # a metrics outage never fails a fenced transfer

    def conformance_case(self):
        """Live tiers from a fixture, or NoHardware. RNGD_PD_NIXL_FIXTURE=`module:callable` receives
        this transport and returns (src_tier, dst_tier, layout, binding), owners kept alive by it."""
        from rm_core.conformance import NoHardware
        fixture = os.environ.get("RNGD_PD_NIXL_FIXTURE")
        if not fixture:
            raise NoHardware("nixl: needs live RNGD/ATOM allocations; set RNGD_PD_NIXL_FIXTURE")
        try:
            self._get_agent()
        except TransportError as e:
            raise NoHardware(str(e)) from e
        import importlib
        module, _, attr = fixture.partition(":")
        src, dst, layout, binding = getattr(importlib.import_module(module), attr)(self)
        self.bind("conf-1", binding)
        return src, dst, layout
