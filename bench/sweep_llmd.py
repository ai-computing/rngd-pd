#!/usr/bin/env python3
"""Sweep the four card assignments through the llm-d gateway.

Every request in this sweep enters at the cluster gateway and is routed by
model name, so what it measures is the served system — Envoy, InferencePool,
EPP, our router, then two NPU pods and a KV handoff over /dev/shm — not a
library call. The four configurations differ only in which card prefills and
which decodes:

  rngd_alone   RNGD kv-free prefill -> RNGD decode
  pd_forward   RNGD kv-free prefill -> ATOM decode   (B = 1, 2, 4)
  pd_reverse   ATOM chunked prefill -> RNGD decode
  atom_alone   ATOM chunked prefill -> ATOM decode   (B = 1, 2, 4)

Batch means B requests in flight at once; the ATOM decode pod holds them until
its B-wide graph can carry them together, which is the only way one weight
stream serves more than one stream. RNGD now decodes batched too: a 32-layer batched graph still does not compile,
but a two-layer one does, and reusing it sixteen times gives a step that is
very nearly B-invariant.

Power and utilisation come from each vendor's own tool, sampled while the cell
runs and attributed to the cards that cell actually used; the pod-to-card map
is read out of the device plugin's env so an idle card's baseline never lands
in another configuration's number.

  python3 bench/sweep_llmd.py --out bench/sweep_llmd.jsonl
"""
import argparse
import getpass
import json
import subprocess
import threading
import time
import urllib.request

import numpy as np

import cluster_env

GATEWAY = cluster_env.gateway()
NS = "llm-serving"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
KUBECTL = cluster_env.kubectl()

RNGD_WEIGHT_BYTES = 8.03e9 * 2      # bf16, resident on the card
ATOM_WEIGHT_BYTES = 6.98e9 * 2      # 32 layers fp16; embed/lm_head stay on host

# The deployment is no longer one fixed shape: decode width and the number of
# prefill cards are both knobs now, and the sweep has to describe whatever was
# actually applied or it will sample the wrong cards' power. These mirror
# gen_sweep.py's RNGD_WIDTHS / PD_PREFILL_N and must be passed to match.
def _rngd_decode_name(B):
    return "sw-rngd-decode" if B == 1 else f"sw-rngd-decode-b{B}"


def build_configs(rngd_widths, atom_widths, prefill_n, solo_b=0):
    """(name, route, widths, prefill pods, decode pods by width) per config.

    `solo` is the control the whole study needs and it does not fit the shape:
    one pod prefills and decodes with the cache never leaving the card, so it
    has no prefill/decode pair and no handoff. It is listed with an empty
    prefill list and itself as the decode pod, which is literally true and
    keeps the power sampler counting exactly the one card it occupies.
    """
    rp = [f"sw-rngd-prefill{'' if i == 0 else i}" for i in range(prefill_n)]
    rd = {b: _rngd_decode_name(b) for b in rngd_widths}
    ad = {b: f"sw-atom-decode-b{b}" for b in atom_widths}
    if solo_b:
        return [("solo", "llama8b-solo", [solo_b], [], {solo_b: "sw-rngd-solo"})]
    return [
        ("rngd_alone", "llama8b-rngd-alone", sorted(rd), rp, rd),
        ("pd_forward", "llama8b-pd-forward", sorted(ad), rp, ad),
        ("pd_reverse", "llama8b-pd-reverse", sorted(rd), ["sw-atom-prefill"], rd),
        ("atom_alone", "llama8b-atom-alone", sorted(ad), ["sw-atom-prefill"], ad),
    ]


def build_chipmap(rngd_widths, prefill_n, solo_b=0, solo_extra=False):
    """Pod -> physical RNGD index, the same assignment gen_sweep.py emits.

    Two layouts put solo on different cards. PD_SOLO replaces the batched
    decoder, so solo lands on the card that pod would have had; PD_SOLO_EXTRA
    keeps both and puts solo after them. Getting this wrong does not fail --
    it samples an idle card and reports its power, which is how a solo run
    came back at 40 W and 0% utilisation and doubled its GFLOPS/W.
    """
    m = {f"sw-rngd-prefill{'' if i == 0 else i}": i for i in range(prefill_n)}
    if solo_b:
        m["sw-rngd-solo"] = prefill_n + (len(rngd_widths) if solo_extra else 0)
        return m
    for i, b in enumerate(rngd_widths):
        m[_rngd_decode_name(b)] = prefill_n + i
    return m


CONFIGS = build_configs([1, 4], [1, 2, 4], 1)
RNGD_CHIP = build_chipmap([1, 4], 1)


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------
class Sampler(threading.Thread):
    """Polls both vendors' tools, keeping every device separate."""

    def __init__(self, period=0.2):
        super().__init__(daemon=True)
        self.period = period
        self.stop_flag = threading.Event()
        self.rngd_w, self.rngd_u = {}, {}
        self.atom_w, self.atom_u, self.atom_m = {}, {}, {}
        self.load1, self.foreign_cpu = [], []

    def _rngd(self):
        try:
            out = subprocess.run(["furiosa-smi", "info"], capture_output=True,
                                 text=True, timeout=4).stdout
            for line in out.splitlines():
                for idx in RNGD_CHIP.values():
                    if f"npu{idx}" in line:
                        try:
                            w = float(line.split("|")[5].strip().rstrip(" W"))
                        except (IndexError, ValueError):
                            continue
                        self.rngd_w.setdefault(idx, []).append(w)
            out = subprocess.run(["furiosa-smi", "status"], capture_output=True,
                                 text=True, timeout=4).stdout
            blocks, cur = [], []
            for line in out.splitlines():
                if line.startswith("+"):
                    if cur:
                        blocks.append(cur)
                    cur = []
                else:
                    cur.append(line)
            if cur:
                blocks.append(cur)
            for b in blocks:
                text = "\n".join(b)
                for idx in RNGD_CHIP.values():
                    if f"npu{idx}" not in text:
                        continue
                    vals = []
                    for line in b:
                        if "Core" in line:
                            v = line.split(":")[-1].replace("|", "").replace("%", "").strip()
                            if v not in ("-", ""):
                                vals.append(float(v))
                    if vals:
                        self.rngd_u.setdefault(idx, []).append(sum(vals) / len(vals))
        except Exception:
            pass

    def _host(self):
        """CPU this host is spending on anything that is not ours.

        The node is shared, and our path does real host-side work -- KV
        marshalling, HTTP, the argmax on some paths -- so a neighbour's job
        shows up in our latencies without touching a single NPU. Recording it
        per cell is the difference between "this number is slow" and "this
        number was measured while someone else had the machine".
        """
        try:
            with open("/proc/loadavg") as fh:
                self.load1.append(float(fh.read().split()[0]))
            out = subprocess.run(
                ["ps", "-eo", "user:32,pcpu", "--no-headers"],
                capture_output=True, text=True, timeout=4).stdout
            foreign = 0.0
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] not in (getpass.getuser(), "root"):
                    foreign += float(parts[1])
            self.foreign_cpu.append(foreign)
        except Exception:
            pass

    def _atom(self):
        try:
            js = json.loads(subprocess.run(["rbln-stat", "--json"],
                                           capture_output=True, text=True,
                                           timeout=4).stdout)
            for d in js.get("devices", []):
                dev = d.get("device")
                self.atom_w.setdefault(dev, []).append(
                    float(str(d["card_power"]).rstrip("uW")) / 1e6)
                self.atom_u.setdefault(dev, []).append(float(d.get("util", 0)))
                self.atom_m.setdefault(dev, []).append(
                    float(d["memory"]["used"]) / 1e9)
        except Exception:
            pass

    def run(self):
        while not self.stop_flag.is_set():
            self._rngd()
            self._atom()
            self._host()
            time.sleep(self.period)

    def report(self, rngd_idxs, atom_devs):
        """Mean power and peak utilisation over just the cards this cell used."""
        def agg(store, keys, fn):
            vals = [v for k in keys for v in store.get(k, [])]
            return round(float(fn(vals)), 1) if vals else None
        return {
            "rngd_power_w": agg(self.rngd_w, rngd_idxs, np.mean),
            "rngd_util_pct": agg(self.rngd_u, rngd_idxs, np.max),
            "atom_power_w": agg(self.atom_w, atom_devs, np.mean),
            "atom_util_pct": agg(self.atom_u, atom_devs, np.max),
            "atom_mem_gb": agg(self.atom_m, atom_devs, np.max),
            "host_load1": round(float(np.mean(self.load1)), 2) if self.load1 else None,
            "foreign_cpu_pct": round(float(np.mean(self.foreign_cpu)), 1)
                               if self.foreign_cpu else None,
        }


# --------------------------------------------------------------------------
# cluster introspection
# --------------------------------------------------------------------------
def atom_device_map():
    """pod app label -> /dev/rblnN, straight out of the device plugin's env."""
    out = subprocess.run(KUBECTL + ["get", "pods", "-n", NS, "-o", "json"],
                         capture_output=True, text=True, timeout=60).stdout
    pods = json.loads(out)["items"]
    m = {}
    for p in pods:
        app = p["metadata"].get("labels", {}).get("app", "")
        if not app.startswith("sw-atom"):
            continue
        name = p["metadata"]["name"]
        # printenv, not `bash -c "echo $VAR"`: kubectl is reached over ssh, and
        # a remote shell would expand the variable on the wrong machine.
        env = subprocess.run(
            KUBECTL + ["exec", "-n", NS, name, "--", "printenv",
                       "PCIDEVICE_REBELLIONS_AI_ATOM_INFO"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        try:
            info = json.loads(env)
            m[app] = list(info.values())[0]["rebellions"]["mount"].split("/")[-1]
        except Exception:
            print(f"  ! could not read device for {app}: {env[:80]!r}")
    return m


def wait_ready(names, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = subprocess.run(KUBECTL + ["get", "pods", "-n", NS, "-o", "json"],
                             capture_output=True, text=True, timeout=60).stdout
        st = {}
        for p in json.loads(out)["items"]:
            app = p["metadata"].get("labels", {}).get("app", "")
            if app in names:
                cs = p["status"].get("containerStatuses") or [{}]
                st[app] = bool(cs[0].get("ready"))
        missing = [n for n in names if not st.get(n)]
        if not missing:
            print(f"[ready] all {len(names)} pods up after {time.time()-t0:.0f}s",
                  flush=True)
            return True
        print(f"  waiting {int(time.time()-t0)}s — not ready: {missing}", flush=True)
        time.sleep(20)
    raise TimeoutError(f"pods never became ready: {missing}")


# --------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------
class Prompts:
    """Length-exact prompts, each with a random head so nothing can be cached."""

    def __init__(self, seed=0):
        from transformers import AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained(MODEL)
        self.rng = np.random.default_rng(seed)
        body = ("Paris is the capital and most populous city of France, "
                "situated on the river Seine in the north of the country. ")
        self.ids = self.tk.encode(body * 40, add_special_tokens=False)

    def make(self, P):
        head = " ".join(str(int(v)) for v in
                        self.rng.integers(10 ** 6, 10 ** 7, size=6))
        hlen = len(self.tk.encode(head, add_special_tokens=False))
        text = head + " " + self.tk.decode(self.ids[:max(P - hlen, 1)])
        # trim to exactly P tokens
        ids = self.tk.encode(text, add_special_tokens=False)[:P]
        return self.tk.decode(ids)


def post(model, prompt, max_new, batch):
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_new, "batch": batch}).encode()
    req = urllib.request.Request(GATEWAY, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    d["_wall_ms"] = (time.perf_counter() - t0) * 1e3
    return d


def fire(model, prompts, max_new, batch):
    """B requests in flight; returns their responses and the wall clock."""
    out, err = [None] * len(prompts), [None] * len(prompts)

    def one(i):
        try:
            out[i] = post(model, prompts[i], max_new, batch)
        except Exception as e:                       # keep the other lanes
            err[i] = e

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = (time.perf_counter() - t0) * 1e3
    live = [o for o in out if o]
    if not live:
        raise RuntimeError(f"all {len(prompts)} requests failed: {err[0]}")
    return live, wall, [e for e in err if e]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/sweep_llmd.jsonl")
    ap.add_argument("--P", default="32,64,128,224")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3, help="timed reps per cell")
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--rngd-widths", default="1,4",
                    help="must match gen_sweep.py's RNGD_WIDTHS")
    ap.add_argument("--atom-widths", default="1,2,4")
    ap.add_argument("--prefill-n", type=int, default=1,
                    help="must match gen_sweep.py's PD_PREFILL_N")
    ap.add_argument("--atom-C", type=int, default=0,
                    help="ATOM prefill chunk width; 0 = the context width S")
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--solo-b", type=int, default=0,
                    help="measure the one-card control at this batch width")
    ap.add_argument("--solo-extra", action="store_true",
                    help="solo runs alongside the decode pods (PD_SOLO_EXTRA), "
                         "not in place of one — it changes which card it is on")
    a = ap.parse_args()
    P_LIST = [int(v) for v in a.P.split(",")]
    rw = [int(v) for v in a.rngd_widths.split(",")]
    aw = [int(v) for v in a.atom_widths.split(",")]
    a.atom_C = a.atom_C or a.S
    global CONFIGS, RNGD_CHIP
    CONFIGS = build_configs(rw, aw, a.prefill_n, a.solo_b)
    RNGD_CHIP = build_chipmap(rw, a.prefill_n, a.solo_b, a.solo_extra)
    keep = set(a.only.split(",")) if a.only else None
    cfgs = [c for c in CONFIGS if keep is None or c[0] in keep]
    print(f"[sweep] S={a.S} RNGD widths={rw} ATOM widths={aw} "
          f"prefill_n={a.prefill_n} chips={RNGD_CHIP}", flush=True)

    pods = sorted({p for c in cfgs for p in c[3]} |
                  {p for c in cfgs for p in c[4].values()})
    print(f"[sweep] pods needed: {pods}", flush=True)
    wait_ready(pods)
    amap = atom_device_map()
    print(f"[sweep] ATOM pod -> card: {amap}", flush=True)

    pr = Prompts()
    rows = []
    fh = open(a.out, "w")
    for name, model, widths, ppods, dpods in cfgs:
        for B in widths:
            dpod = dpods[B]
            # Every card the cell occupies, prefill pool included: with two
            # prefill pods behind one decode pod, sampling only the first would
            # halve the configuration's reported power.
            used = list(ppods) + [dpod]
            rngd_idxs = [RNGD_CHIP[p] for p in used if p in RNGD_CHIP]
            atom_devs = [amap[p] for p in used if p in amap]
            for P in P_LIST:
                prompts = [pr.make(P) for _ in range(B)]
                try:
                    fire(model, prompts, a.D, B)          # warm the path
                except Exception as e:
                    print(f"[{name} B={B} P={P}] warmup failed: {e}", flush=True)
                    continue
                s = Sampler(); s.start()
                best = None
                for _ in range(a.reps):
                    prompts = [pr.make(P) for _ in range(B)]
                    try:
                        live, wall, errs = fire(model, prompts, a.D, B)
                    except Exception as e:
                        print(f"[{name} B={B} P={P}] failed: {e}", flush=True)
                        best = None
                        break
                    if best is None or wall < best[1]:
                        best = (live, wall, errs)
                s.stop_flag.set(); s.join(timeout=3)
                if best is None:
                    continue
                live, wall, errs = best
                pd = [o["pd"] for o in live]
                med = lambda k: float(np.median([p[k] for p in pd]))
                prefill_ms = med("prefill_ms")
                # Each worker times its own request, and the device lock
                # serialises them, so a per-request median understates how long
                # the prefill card was actually occupied by the batch. The
                # slowest request's prefill window is the closest thing we have
                # to that span, and dividing B*P tokens by the median instead
                # would report a prefill rate the card never sustained.
                prefill_span_ms = float(np.max([p["prefill_ms"] for p in pd]))
                ttft_span_ms = float(np.max([p["ttft_ms"] for p in pd]))
                tpot_ms = med("tpot_ms")
                n_out = sum(o["usage"]["completion_tokens"] for o in live)
                # weights are streamed once per decode step regardless of B, and
                # once per prefill pass — ATOM prefills in chunks of 16.
                dec_bytes = (RNGD_WEIGHT_BYTES if dpod in RNGD_CHIP
                             else ATOM_WEIGHT_BYTES)
                # solo has no separate prefill pod: the one card that decodes
                # also prefilled, so its weight stream is the decode pod's.
                pre_pod = ppods[0] if ppods else dpod
                if pre_pod in RNGD_CHIP:
                    pre_bytes = RNGD_WEIGHT_BYTES          # one whole-prompt pass
                else:
                    # ATOM prefills in chunks of C, and C is the context width
                    # now, not the 16 an earlier build used. Leaving 16 here
                    # inflated ATOM's prefill bandwidth by P/16 -- a prompt of
                    # 224 tokens was credited with fourteen weight passes it
                    # never made.
                    pre_bytes = ATOM_WEIGHT_BYTES * int(np.ceil(P / a.atom_C))
                row = {"config": name, "P": P, "D": a.D, "B": B, "S": a.S,
                       "prefill_n": a.prefill_n,
                       "ttft_ms": round(med("ttft_ms"), 1),
                       "tpot_ms": round(tpot_ms, 2),
                       "tokens_s": round(n_out / (wall / 1e3), 2),
                       "wall_ms": round(wall, 1),
                       "prefill_ms": round(prefill_ms, 1),
                       "prefill_span_ms": round(prefill_span_ms, 1),
                       "ttft_span_ms": round(ttft_span_ms, 1),
                       "prefill_ms_per_token": round(prefill_ms / P, 3),
                       "handoff_ms": round(med("kv_save_ms") + med("kv_load_ms"), 1),
                       "kv_mb": pd[0]["kv_mb"],
                       "decode_bw_gbs": round(dec_bytes / (tpot_ms / 1e3) / 1e9, 1),
                       "prefill_bw_gbs": round(pre_bytes / (prefill_ms / 1e3) / 1e9, 1),
                       "errors": len(errs),
                       "text": live[0]["choices"][0]["text"][:40]}
                row.update(s.report(rngd_idxs, atom_devs))
                rows.append(row)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{name:11s} B={B} P={P:3d}] TTFT {row['ttft_ms']:7.1f}  "
                      f"TPOT {row['tpot_ms']:6.2f}  {row['tokens_s']:6.2f} tok/s  "
                      f"RNGD {row['rngd_power_w']}W/{row['rngd_util_pct']}%  "
                      f"ATOM {row['atom_power_w']}W/{row['atom_util_pct']}%",
                      flush=True)
    fh.close()
    print(f"\n[sweep] {len(rows)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
