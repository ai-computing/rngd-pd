"""ATOM delta load: in-session swap, full reload, and the accumulating session.

  # card-free: one compile per session (bucketing), weights once per session
  python -m rngd_pd.swap.bench_atom build --ckpt <Llama-3.1-8B> --out s1.rbln --variants B1,S256 B4,S256
  python -m rngd_pd.swap.bench_atom build --ckpt <Llama-3.1-8B> --out s2.rbln --variants B1,S256 B4,S256 B2,S512
  # one ATOM: everything below, each fresh load in its own process
  python -m rngd_pd.swap.bench_atom all --ckpt <Llama-3.1-8B> --session s1.rbln --next s2.rbln \\
      --outside B2,S512 --out atom.json

Phases of `all`:
  fresh    process start -> first token for each variant of the first session
           (full reload, page cache as found and after a warm read), tokens;
  swap     one process: load the session, swap() between its variants
           (request -> first token), tokens compared with `fresh`;
  outside  a variant only in --next: the running session keeps serving while
           HostPool parses --next on the host, then switch() and the first
           token on it; tokens compared with a fresh load of --next.
"""
import argparse
import json
import os
import subprocess
import sys
import time

T_START = time.perf_counter()

import numpy as np  # noqa: E402

from rngd_pd.swap.variants import Variant  # noqa: E402


def prompts(n, P, seed=0):
    rng = np.random.default_rng(seed)
    return [list(map(int, rng.integers(100, 120000, P))) for _ in range(n)]


def mem(device=0):
    """Device memory in use, GiB, from `rbln-stat -j` (rebel 0.11 has no Python call for it)."""
    try:
        out = subprocess.run(["rbln-stat", "-j"], capture_output=True, text=True, timeout=30).stdout
        devs = json.loads(out)["devices"]
        d = devs[device] if device < len(devs) else devs[0]
        return round(int(d["memory"]["used"]) / 2 ** 30, 2)
    except Exception as e:  # noqa: BLE001 - a measurement must not stop the run
        return repr(e)


def _tables(a):
    from rngd_pd.swap.atom import Tables
    return Tables(a.ckpt)


def cmd_build(a):
    from rngd_pd.swap.atom import build_session
    vs = [Variant.parse(x, 0) for x in a.variants]
    meta = build_session(a.ckpt, a.session_out, vs, layers=a.layers)
    print(json.dumps(meta, indent=1))
    _write(a, dict(mode="build", **meta))


def cmd_fresh(a):
    from rngd_pd.swap.atom import AtomSession, warm_page_cache
    v = Variant.parse(a.variant, 0)
    warm = warm_page_cache(a.session) if a.warm else None
    t0 = time.perf_counter()
    tab = _tables(a)
    t1 = time.perf_counter()
    s = AtomSession(a.session, tab, device=a.device)
    s.swap(v)
    first = s.first_step()
    t_first = time.perf_counter()
    wall_first = time.time()
    m = mem(a.device)
    toks, ttft = s.generate(prompts(v.B, a.P), a.n)
    _write(a, dict(mode="fresh", session=os.path.basename(a.session), variant=v.name,
                   warm_read_s=warm, tables_s=t1 - t0, session_timings=s.timings,
                   first_step_s=first, load_to_first_token_s=t_first - t0,
                   launch_to_first_token_s=(wall_first - a.t_launch - (warm or 0))
                   if a.t_launch else None,
                   device_mem=m, prompt_ttft_s=ttft, step_s=s.last_step_s, tokens=toks))


def cmd_swap(a):
    from rngd_pd.swap.atom import AtomSession
    tab = _tables(a)
    s = AtomSession(a.session, tab, device=a.device)
    m0 = mem(a.device)
    rows = []
    for rnd in range(2):
        for v in s.variants:
            t0 = time.perf_counter()
            info = s.swap(v)
            info["first_step_s"] = s.first_step()
            info["request_to_first_token_s"] = time.perf_counter() - t0
            toks, ttft = s.generate(prompts(v.B, a.P), a.n)
            info.update(round=rnd, tokens=toks, prompt_ttft_s=ttft, step_s=s.last_step_s)
            rows.append(info)
            print(f"[swap] round {rnd} {v.name}: {info['request_to_first_token_s']*1e3:.0f} ms",
                  flush=True)
    _write(a, dict(mode="swap", session=os.path.basename(a.session),
                   session_timings=s.timings, device_mem=m0, rows=rows))


def cmd_outside(a):
    """Serve on --session, need --outside (only in --next): prepare, switch, first token."""
    from rngd_pd.swap.atom import AtomSession, HostPool
    tab = _tables(a)
    s = AtomSession(a.session, tab, device=a.device)
    v0, v = s.variants[0], Variant.parse(a.outside, 0)
    assert not s.has(v), f"{v.name} is already in {a.session}"
    s.swap(v0)
    m_before = mem(a.device)
    pool = HostPool()
    t_req = time.perf_counter()
    pool.prepare(a.next)
    # the running session keeps serving while the next one is parsed
    served = 0
    while not pool.ready():
        s.generate(prompts(v0.B, a.P), 2)
        served += 1
    t_ready = time.perf_counter()
    s2, sw = pool.switch(s, tab, device=a.device)
    s2.swap(v)
    first = s2.first_step()
    t_first = time.perf_counter()
    m_after = mem(a.device)
    toks, ttft = s2.generate(prompts(v.B, a.P), a.n)
    _write(a, dict(mode="outside", session=os.path.basename(a.session),
                   next=os.path.basename(a.next), variant=v.name,
                   request_to_first_token_s=t_first - t_req,
                   prepare_while_serving_s=t_ready - t_req, served_batches_meanwhile=served,
                   switch=sw, first_step_s=first,
                   switch_to_first_token_s=t_first - t_ready,
                   device_mem_before=m_before, device_mem_after=m_after,
                   tokens=toks, prompt_ttft_s=ttft))


def _write(a, obj):
    obj["argv"] = sys.argv[1:]
    with open(a.out, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    print(f"[bench] wrote {a.out}", flush=True)


def _child(a, mode, out, extra):
    cmd = [sys.executable, "-m", "rngd_pd.swap.bench_atom", mode, "--ckpt", a.ckpt,
           "--device", str(a.device), "--P", str(a.P), "--n", str(a.n), "--out", out,
           "--t-launch", repr(time.time())] + extra
    print("[all] " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)
    with open(out) as f:
        return json.load(f)


def cmd_all(a):
    from rngd_pd.swap.atom import AtomSession  # noqa: F401  (fail early without rebel)
    d = a.out + ".d"
    os.makedirs(d, exist_ok=True)
    with open(a.session + ".json") as f:
        names = json.load(f)["variants"]
    res = dict(fresh={}, fresh_warm={})
    for n in names:
        res["fresh"][n] = _child(a, "fresh", f"{d}/fresh_{n}.json",
                                 ["--session", a.session, "--variant", n])
    res["fresh_warm"][names[0]] = _child(a, "fresh", f"{d}/fresh_warm_{names[0]}.json",
                                         ["--session", a.session, "--variant", names[0], "--warm"])
    res["swap"] = _child(a, "swap", f"{d}/swap.json", ["--session", a.session])
    for r in res["swap"]["rows"]:
        r["tokens_equal_fresh"] = r["tokens"] == res["fresh"][r["variant"]]["tokens"]
    if a.next and a.outside:
        vname = Variant.parse(a.outside, 0).name
        res["fresh_next"] = _child(a, "fresh", f"{d}/fresh_next_{vname}.json",
                                   ["--session", a.next, "--variant", a.outside])
        res["outside"] = _child(a, "outside", f"{d}/outside.json",
                                ["--session", a.session, "--next", a.next,
                                 "--outside", a.outside])
        res["outside"]["tokens_equal_fresh"] = \
            res["outside"]["tokens"] == res["fresh_next"]["tokens"]
    _write(a, res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "fresh", "swap", "outside", "all"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--session", help=".rbln of the running session")
    ap.add_argument("--session-out", help="build: where to write the session .rbln")
    ap.add_argument("--next", help=".rbln of the session with the outside variant added")
    ap.add_argument("--outside", help="variant not in --session (in --next)")
    ap.add_argument("--variant")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warm", action="store_true", help="fresh: read the file into the page cache first")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--t-launch", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"build": cmd_build, "fresh": cmd_fresh, "swap": cmd_swap, "outside": cmd_outside,
     "all": cmd_all}[a.mode](a)


if __name__ == "__main__":
    main()
