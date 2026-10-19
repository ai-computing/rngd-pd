"""ATOM landing buffers for bench/nixl_kv_check.py (ATOM Python environment, under hetero-npu-p2p's
examples/koni/capture.gdb so the live allocations can be exported as DMA-BUFs).

Allocates one eager uint8 tensor per region on rbln:0, exports each with
hetero_p2p.atom_experimental (pinned thunk ABI), sends the fds to the RNGD process over the Unix
socket named in the config (SCM_RIGHTS) and answers, one JSON message each:
  {"cmd": "fill", "value": v}        write v everywhere (host -> ATOM through the SDK)
  {"cmd": "dump", "path": p}         after the RNGD side's fence: drop the runtime's host view
                                     (mark_device_memory_is_updated), read every region back and
                                     write the concatenated bytes to p
  {"cmd": "stop"}
"""
import array
import ctypes
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch_rbln  # noqa: F401  registers the rbln device
import rebel

from hetero_p2p.atom_experimental import export_live_caches

config = json.loads(Path(sys.argv[1]).read_text())
out = Path(config["out"])
vmem = rebel._C.vmem
tensors, exports = [], []
conn = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
try:
    regions = []
    for size in config["regions"]:
        tensors.append(torch.zeros(size, dtype=torch.uint8).to("rbln:0"))
    capture = json.loads((out / "live-capture.json").read_text())
    for i, t in enumerate(tensors):
        ex = export_live_caches(t, [vmem.get_device_addrs(t.data_ptr())], t.numel(), capture, experimental=True)
        exports.append(ex)
        rec, m = ex.records[0], ex.mappings[0]
        regions.append(dict(block=i, export_size=rec["size"], offset=m["offset"], dva=rec["dva"], size=t.numel()))
    (out / "capture-complete").write_text("exports done")
    ctypes.CDLL(None).getppid()                 # capture.gdb disables its breakpoints here
    conn.connect(config["socket"])
    conn.sendmsg([json.dumps(dict(regions=regions)).encode()],
                 [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [ex.records[0]["fd"] for ex in exports]))])
    while True:
        cmd = json.loads(conn.recv(65536))
        try:
            t0 = time.perf_counter()
            if cmd["cmd"] == "stop":
                conn.send(json.dumps(dict(ok=True)).encode())
                break
            if cmd["cmd"] == "fill":
                for t in tensors:
                    t.copy_(torch.full((t.numel(),), int(cmd["value"]), dtype=torch.uint8))
                    vmem.sync_to_device(t.data_ptr())
                reply = dict(ok=True)
            elif cmd["cmd"] == "dump":
                with open(cmd["path"], "wb") as f:
                    for t in tensors:
                        vmem.mark_device_memory_is_updated(t.data_ptr())
                        f.write(np.ascontiguousarray(t.cpu().numpy()).tobytes())
                reply = dict(ok=True)
            else:
                reply = dict(ok=False, error=f"unknown command {cmd}")
            reply["seconds"] = time.perf_counter() - t0
        except Exception as e:
            reply = dict(ok=False, error=repr(e), traceback=traceback.format_exc())
        conn.send(json.dumps(reply).encode())
except Exception:
    traceback.print_exc()
    raise
finally:
    conn.close()
    for ex in exports:
        ex.close()
