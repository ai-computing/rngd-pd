"""The `nixl` KV transport on the node: real prefill KV, moved by NIXL + HETERO_NPU, checked byte for byte.

For each prompt, the RNGD prefill engine (the serving FullPrefillWorker) runs twice:

  shm   the serving path: prefill_stream -> device download -> connector -> /dev/shm (bf16 wire) ->
        connector load on the decode side; timed save and load
  nixl  prefill_device leaves K [nl,NKV,HD,S] and V [nl,NKV,S,HD] on the card (bf16); the `nixl`
        transport moves those allocations:
          rngd -> atom   direct, into DMA-BUFs exported from live ATOM allocations (bench/nixl_atom_owner.py)
          rngd -> host   staging leg into pinned host pages
          host -> rngd   staging leg onto a second RNGD card (with the leg above: RNGD -> RNGD)

Every destination is read back (ATOM through its SDK after mark_device_memory_is_updated, the second
RNGD through the Furiosa SDK) and compared with the source bytes and with the shm payload. The
landed caches are saved for bench/nixl_decode_tokens.py, which decodes on ATOM from the shm KV and
from the KV that crossed RNGD -> ATOM and compares the tokens.

Run in the prefill environment (Furiosa torch + nixl + NIXL_PLUGIN_DIR with libplugin_HETERO_NPU.so),
on cards no pod holds:

    PYTHONPATH=src:<nixl pylib>:<hetero-npu-p2p>/src python bench/nixl_kv_check.py \
        --chip 2 --rngd-index 3 --stage-chip 1 --stage-index 1 \
        --atom-python <ATOM env python> --hnp <hetero-npu-p2p checkout> --out runs/nixl-check
"""
import _bootstrap  # noqa: F401
import argparse
import array
import json
import mmap
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROMPTS = ["The capital of France is",
           "Explain disaggregated prefill and decode in two sentences.",
           " ".join(["word"] * 120),
           "Write a Python function that returns the n-th Fibonacci number. " * 6,
           " ".join(["token"] * 470)]


def bf16_bits(t):
    """uint16 bit pattern of a bf16 torch tensor (host copy)."""
    import torch
    return t.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)


def f32_to_bf16_bits(a):
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


class AtomOwner:
    """bench/nixl_atom_owner.py under hetero-npu-p2p's capture.gdb; receives the exported DMA-BUF fds."""

    def __init__(self, atom_python, hnp, out, sizes, timeout=600):
        self.out = Path(out)
        for name in ("live-capture.json", "capture-complete", "breakpoints-disabled"):
            (self.out / name).unlink(missing_ok=True)
        sock_path = str(self.out / "atom.sock")
        Path(sock_path).unlink(missing_ok=True)
        lst = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        lst.bind(sock_path); lst.listen(1); lst.settimeout(1)
        cfg = self.out / "atom-config.json"
        cfg.write_text(json.dumps(dict(socket=sock_path, out=str(self.out), regions=list(sizes))))
        env = dict(os.environ, HETERO_CAPTURE_DIR=str(self.out), TORCH_RBLN_EAGER_MALLOC="1",
                   PYTHONPATH=os.pathsep.join(p for p in (str(Path(hnp) / "src"), os.environ.get("PYTHONPATH")) if p))
        for k in ("RBLN_DEVICE_MAP", "RBLN_DUMMY_DEVICE", "PYTHONHOME"):
            env.pop(k, None)
        self.log = (self.out / "atom-owner.log").open("w")
        self.proc = subprocess.Popen(
            ["gdb", "-q", "-batch", "-x", str(Path(hnp) / "examples/koni/capture.gdb"), "--args", atom_python,
             str(Path(__file__).resolve().parent / "nixl_atom_owner.py"), str(cfg)],
            env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ATOM owner exited; see {self.out}/atom-owner.log")
            if time.monotonic() > deadline:
                self.proc.kill(); raise RuntimeError("ATOM owner did not export in time")
            try:
                self.conn, _ = lst.accept(); break
            except socket.timeout:
                pass
        lst.close()
        data, anc, _flags, _ = self.conn.recvmsg(65536, socket.CMSG_SPACE(16 * 4))
        fds = array.array("i")
        for level, kind, payload in anc:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(payload)
        self.regions = [dict(r, fd=fds[r["block"]]) for r in json.loads(data)["regions"]]

    def call(self, **cmd):
        self.conn.send(json.dumps(cmd).encode())
        reply = json.loads(self.conn.recv(65536))
        if not reply.get("ok"):
            raise RuntimeError(f"ATOM owner: {reply}")
        return reply

    def close(self):
        try:
            self.call(cmd="stop")
        except Exception:
            pass
        self.conn.close()
        for r in self.regions:
            os.close(r["fd"])
        self.proc.wait(timeout=120)
        self.log.close()


def pinned(nbytes):
    buf = mmap.mmap(-1, nbytes)                  # page aligned, as the scatter path requires
    arr = np.frombuffer(buf, dtype=np.uint8)
    arr[:] = 0                                   # touch every page before the DMA
    return buf, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True, help="prefill card: torch ordinal among visible cards")
    ap.add_argument("--rngd-index", type=int, default=None, help="prefill card: host index (/dev/rngd/npuN); "
                    "default: the visible card at that ordinal")
    ap.add_argument("--stage-chip", type=int, default=None, help="second RNGD (RNGD -> RNGD over two legs)")
    ap.add_argument("--stage-index", type=int, default=None)
    ap.add_argument("--atom-python", default=None, help="ATOM environment python (RNGD -> ATOM)")
    ap.add_argument("--hnp", default=None, help="hetero-npu-p2p checkout (capture.gdb, atom_experimental)")
    ap.add_argument("--S", type=int, default=512)
    ap.add_argument("--C", type=int, default=512)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--queue-depth", type=int, default=4)
    ap.add_argument("--align", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=3, help="timed transfers per prompt and route")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from rngd_pd.devices import visible_cards
    cards = visible_cards()                      # host indices, in torch ordinal order
    if a.rngd_index is None:
        a.rngd_index = cards[a.chip]
    if a.stage_chip is not None and a.stage_index is None:
        a.stage_index = cards[a.stage_chip]
    print(f"prefill card npu{a.rngd_index} (ordinal {a.chip}), stage card "
          f"{'npu%d' % a.stage_index if a.stage_index is not None else '-'}", flush=True)

    os.environ.setdefault("AIRM_KV_TRANSPORT", "shm")
    os.environ.setdefault("PD_PREFETCH", "0")
    import torch
    from rm_core import KVLayout, MemoryTier
    from rngd_pd.serving import cluster
    from rngd_pd.kv_nixl import NixlBinding, NixlTransport, dmabuf_region, host_region, rngd_region
    from transformers import AutoTokenizer
    from rngd_pd.engines.fsdk import sync

    tok = AutoTokenizer.from_pretrained(cluster.MODEL)
    w = cluster.FullPrefillWorker(a.chip, a.S, C=a.C, group=a.group)
    nl, S = w.nl, a.S

    class _Loader(cluster.DecodeWorkerBase):    # the decode side's connector load, without a card
        pass
    loader = _Loader()
    loader.nl, loader.NKV, loader.HD, loader.S = nl, 8, 128, S

    layout = KVLayout(num_layers=nl, num_kv_heads=8, head_dim=128, dtype="bf16", order="vendor:rngd-pd")
    node = "local"
    T_RNGD = MemoryTier(f"rngd{a.rngd_index}", "device-dram", 48 << 30, f"{node}/furiosa:{a.rngd_index}")
    T_STAGE = MemoryTier(f"rngd{a.stage_index}", "device-dram", 48 << 30, f"{node}/furiosa:{a.stage_index}")
    T_ATOM = MemoryTier("atom0", "device-hbm", 16 << 30, f"{node}/rebellions:0")
    T_HOST = MemoryTier("host", "host-dram", 64 << 30)
    dev = f"/dev/rngd/npu{a.rngd_index}ch0"
    nkv = S * layout.bytes_per_token() // 2      # bytes of K (= of V)
    tr = NixlTransport(queue_depth=a.queue_depth, align=a.align)

    def move(req, src_t, dst_t, src_regions, dst_regions):
        tr.bind(req, NixlBinding(tuple(src_regions), tuple(dst_regions), S))
        plan = tr.plan(req, src_t, dst_t, layout, layout, (0, S))
        res = tr.wait(tr.submit(plan), 30.0)
        if not res.ok:
            raise RuntimeError(f"{req}: {res.error}")
        return res.seconds * 1e3

    host_buf, host = pinned(2 * nkv)
    stage = None
    if a.stage_chip is not None:
        from rngd_pd.engines.fsdk import dev_str
        sdev = dev_str(a.stage_chip)
        try:                                     # SDK 2026.4: an explicitly flat allocation
            from furiosa.torch._C.ir import FlatDramShape
            from furiosa.torch._C.tensor import copy_from_tensor
            stage = copy_from_tensor(torch.zeros(2 * nkv, dtype=torch.uint8),
                                     FlatDramShape.new_as_broadcast([2 * nkv], torch.uint8), sdev)
        except ImportError:                      # older SDKs: a 1-D byte tensor (checked by the readback)
            stage = torch.zeros(2 * nkv, dtype=torch.uint8).to(sdev)
    atom = None
    if a.atom_python:
        atom = AtomOwner(a.atom_python, a.hnp, out, [nkv, nkv])
        atom_regions = [dmabuf_region(r["fd"], r["size"], base=r["dva"] + r["offset"], offset=r["offset"])
                        for r in atom.regions]

    report = dict(S=S, layers=nl, kv_bytes=2 * nkv, queue_depth=a.queue_depth, prompts=[])
    for i, text in enumerate(PROMPTS):
        ids = tok(text)["input_ids"][: a.C]
        rid = f"nixlchk{i}-{os.getpid()}"
        # shm: the serving path end to end (prefill_stream, download, bf16 wire, /dev/shm, load)
        pf = w.run(rid, ids)
        K, V, load_ms = loader._load_kv_now(rid)
        shm_bits = np.concatenate([f32_to_bf16_bits(K).reshape(-1), f32_to_bf16_bits(V).reshape(-1)])
        # nixl: the same prompt left on the card
        with cluster.DEVICE:
            Kp, Vp, nxt = w.eng.prefill_device(ids)
            sync(a.chip)                         # 2026.3+ queues work: the DMA must not race it
        ref = np.concatenate([bf16_bits(Kp).reshape(-1), bf16_bits(Vp).reshape(-1)])
        row = dict(prompt_tokens=len(ids), next=int(nxt), shm_next=int(pf["next"]),
                   shm_save_ms=pf["save_ms"], shm_download_ms=pf["download_ms"], shm_load_ms=round(load_ms, 1),
                   device_equals_shm=bool(np.array_equal(ref, shm_bits)),
                   kp_ptr=hex(Kp.data_ptr()), vp_ptr=hex(Vp.data_ptr()))
        src = [rngd_region(Kp, nkv, dev), rngd_region(Vp, nkv, dev)]
        # RNGD -> host (staging leg)
        t = [move(f"{rid}-h{r}", T_RNGD, T_HOST, src, [host_region(host.ctypes.data, 2 * nkv)]) for r in range(a.reps)]
        row["rngd_host_ms"] = [round(x, 2) for x in t]
        row["rngd_host_bytes_equal"] = bool(np.array_equal(host.view(np.uint16), ref))
        # host -> RNGD on the second card: with the leg above, RNGD -> RNGD
        if stage is not None:
            sdev_path = f"/dev/rngd/npu{a.stage_index}ch0"
            t = [move(f"{rid}-s{r}", T_HOST, T_STAGE, [host_region(host.ctypes.data, 2 * nkv)],
                      [rngd_region(stage, 2 * nkv, sdev_path)]) for r in range(a.reps)]
            row["host_rngd_ms"] = [round(x, 2) for x in t]
            row["rngd_rngd_bytes_equal"] = bool(np.array_equal(stage.cpu().numpy().view(np.uint16), ref))
        # RNGD -> ATOM (direct)
        if atom is not None:
            times = []
            for r in range(a.reps):
                atom.call(cmd="fill", value=0x5A)
                times.append(move(f"{rid}-a{r}", T_RNGD, T_ATOM, src, atom_regions))
            row["rngd_atom_ms"] = [round(x, 2) for x in times]
            dump = out / f"atom-{i}.bin"
            atom.call(cmd="dump", path=str(dump))
            landed = np.fromfile(dump, dtype=np.uint16)
            row["rngd_atom_bytes_equal"] = bool(np.array_equal(landed, ref))
            n = nkv // 2
            np.savez(out / f"kv-{i}.npz", first=int(nxt), pos=len(ids),
                     shm_k=f32_to_bf16_bits(K), shm_v=f32_to_bf16_bits(V),
                     atom_k=landed[:n].reshape(K.shape), atom_v=landed[n:].reshape(V.shape))
            dump.unlink()
        report["prompts"].append(row)
        print(json.dumps(row), flush=True)
        del Kp, Vp
    report["measurements"] = [dict(name=m.name, value=m.value, tags=m.tags) for m in tr.measurements]
    tr.close()
    if atom is not None:
        atom.close()
    (out / "report.json").write_text(json.dumps(report, indent=1))
    ok = all(p["rngd_host_bytes_equal"] and p.get("rngd_atom_bytes_equal", True) and p.get("rngd_rngd_bytes_equal", True)
             for p in report["prompts"])
    print("NIXL-CHECK", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
