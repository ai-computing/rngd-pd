"""Swap against full reload on one RNGD card, with greedy tokens compared.

  python -m rngd_pd.swap.bench_rngd all --ckpt <Llama-3.1-8B dir> --chip 0 \\
      --variants B4,S512 B8,S512 B16,S512 B4,S256 --out swap_rngd.json

`all` runs, each in a fresh process so the card is released in between:
  fresh   one variant loaded from nothing: process start -> first token
          (the full reload), then greedy tokens;
  swap    one engine, first variant loaded, then swap() to each other
          variant: request -> first token on it, then greedy tokens; a second
          round swaps back between already-prepared variants;
  delta   (with --delta-new) the engine on --ckpt switches its weights to the
          checkpoint in --delta-new by uploading only the tensors whose bytes
          changed, against a fresh load of --delta-new.

`make-delta` writes a checkpoint that differs from --ckpt in a few tensors
(same files, the bytes of those tensors replaced in place), for the delta run.
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


def _engine(a, ckpt, group):
    from rngd_pd.swap.rngd import SwappableDecode
    return SwappableDecode(ckpt, group=group, chip=a.chip, cache_dir=a.cache)


def _mem(a, eng):
    if not a.mem:
        return None
    from rngd_pd.swap.rngd import free_capacity_gib
    return free_capacity_gib(eng.dev)


def cmd_fresh(a):
    v = Variant.parse(a.variant, a.group)
    ckpt = a.ckpt
    t_import = time.perf_counter()
    eng = _engine(a, ckpt, v.group)
    t_w = time.perf_counter()
    info = eng.swap(v)
    first_s, _ = eng.first_step()
    t_first = time.perf_counter()
    wall_first = time.time()
    toks, ttft = eng.generate(prompts(v.B, a.P), a.n)
    out = dict(mode="fresh", variant=v.name, ckpt=ckpt,
               launch_to_first_token_s=(wall_first - a.t_launch) if a.t_launch else None,
               process_to_first_token_s=t_first - T_START,
               imports_s=t_import - T_START, weights=eng.timings,
               engine_init_s=t_w - t_import, graph=info, first_step_s=first_s,
               prompt_ttft_s=ttft, tokens=toks, free_gib=_mem(a, eng))
    _write(a, out)


def cmd_swap(a):
    vs = [Variant.parse(x, a.group) for x in a.variants]
    eng = _engine(a, a.ckpt, vs[0].group)
    rows = []
    free0 = _mem(a, eng)
    for rnd in range(2):
        for v in vs:
            if rnd == 0 and v == vs[0]:
                info = eng.swap(v)
                first_s, _ = eng.first_step()
                info["first_step_s"] = first_s
                info["request_to_first_token_s"] = info["swap_s"] + first_s
                info["initial"] = True
            else:
                t0 = time.perf_counter()
                info = eng.swap(v)
                first_s, _ = eng.first_step()
                info["first_step_s"] = first_s
                info["request_to_first_token_s"] = time.perf_counter() - t0
            toks, ttft = eng.generate(prompts(v.B, a.P), a.n)
            info.update(round=rnd, tokens=toks, prompt_ttft_s=ttft)
            if rnd == 0:
                info["free_gib_after"] = _mem(a, eng)
            rows.append(info)
            print(f"[swap] round {rnd} {v.name}: request->first token "
                  f"{info['request_to_first_token_s']*1e3:.0f} ms", flush=True)
    _write(a, dict(mode="swap", ckpt=a.ckpt, weights=eng.timings, free_gib_start=free0,
                   rows=rows))


def cmd_delta(a):
    from rngd_pd.swap import delta
    v = Variant.parse(a.variant, a.group)
    eng = _engine(a, a.ckpt, v.group)
    eng.swap(v)
    base_toks, _ = eng.generate(prompts(v.B, a.P), a.n)
    t0 = time.perf_counter()
    if a.ranges and os.path.exists(a.ranges):
        rng = delta.load_onggi_ranges(a.ranges)
        source = "onggi"
    else:
        rng = None
        source = "local-compare"
    plan = delta.plan_update(a.ckpt, a.delta_new, rng)
    t_plan = time.perf_counter() - t0
    t1 = time.perf_counter()
    res = eng.apply_delta(a.delta_new, plan, mode=a.delta_mode)
    first_s, _ = eng.first_step()
    t_first = time.perf_counter() - t1
    toks, ttft = eng.generate(prompts(v.B, a.P), a.n)
    _write(a, dict(mode="delta", variant=v.name, ranges_source=source, plan_s=t_plan,
                   plan=plan.summary(), apply=res,
                   request_to_first_token_s=t_first, base_tokens=base_toks,
                   tokens=toks))


def cmd_make_delta(a):
    """Copy of --ckpt with --tensors replaced in place (same dtype, shape, offsets)."""
    import shutil
    from rngd_pd.swap.delta import read_header
    os.makedirs(a.out, exist_ok=False)
    names = a.tensors.split(",")
    with open(os.path.join(a.ckpt, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    touched = {where[n] for n in names}
    for fn in os.listdir(a.ckpt):
        src = os.path.join(a.ckpt, fn)
        if os.path.isdir(src):
            continue
        dst = os.path.join(a.out, fn)
        if fn in touched:
            shutil.copyfile(src, dst)
        else:
            os.symlink(os.path.realpath(src), dst)
    rng = np.random.default_rng(1)
    for fn in touched:
        path = os.path.join(a.out, fn)
        hdr = read_header(path)
        with open(path, "r+b") as f:
            for n in names:
                if where[n] != fn:
                    continue
                info = hdr[n]
                f.seek(info["start"])
                raw = np.frombuffer(f.read(info["end"] - info["start"]), np.uint16)
                # bf16 -> f32 by shifting, perturb, back by truncation: a fine-tune-like change
                w = (raw.astype(np.uint32) << 16).view(np.float32)
                w = w * (1 + a.scale * rng.standard_normal(w.shape).astype(np.float32))
                new = (w.view(np.uint32) >> 16).astype(np.uint16)
                f.seek(info["start"])
                f.write(new.tobytes())
    print(json.dumps(dict(out=a.out, files=sorted(touched), tensors=names)))


def _write(a, obj):
    obj["argv"] = sys.argv[1:]
    with open(a.out, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[bench] wrote {a.out}", flush=True)


def _child(a, mode, out, extra):
    cmd = [sys.executable, "-m", "rngd_pd.swap.bench_rngd", mode, "--chip", str(a.chip),
           "--P", str(a.P), "--n", str(a.n), "--group", str(a.group), "--out", out,
           "--t-launch", repr(time.time())] + (["--cache", a.cache] if a.cache else []) + \
        (["--mem"] if a.mem else []) + extra
    print("[all] " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)
    with open(out) as f:
        return json.load(f)


def cmd_all(a):
    d = a.out + ".d"
    os.makedirs(d, exist_ok=True)
    res = dict(fresh={}, chip=a.chip)
    for x in a.variants:
        v = Variant.parse(x, a.group)
        res["fresh"][v.name] = _child(a, "fresh", f"{d}/fresh_{v.name}.json",
                                      ["--ckpt", a.ckpt, "--variant", x])
    res["swap"] = _child(a, "swap", f"{d}/swap.json",
                         ["--ckpt", a.ckpt, "--variants", *a.variants])
    for r in res["swap"]["rows"]:
        r["tokens_equal_fresh"] = r["tokens"] == res["fresh"][r["variant"]]["tokens"]
    if a.delta_new:
        v0 = a.variants[0]
        name = Variant.parse(v0, a.group).name
        res["fresh_new"] = _child(a, "fresh", f"{d}/fresh_new.json",
                                  ["--ckpt", a.delta_new, "--variant", v0])
        extra = ["--ckpt", a.ckpt, "--delta-new", a.delta_new, "--variant", v0,
                 "--delta-mode", a.delta_mode]
        if a.ranges:
            extra += ["--ranges", a.ranges]
        res["delta"] = _child(a, "delta", f"{d}/delta.json", extra)
        res["delta"]["tokens_equal_fresh_new"] = res["delta"]["tokens"] == res["fresh_new"]["tokens"]
        res["delta"]["base_tokens_equal_fresh"] = res["delta"]["base_tokens"] == res["fresh"][name]["tokens"]
        res["delta"]["tokens_differ_from_base"] = res["delta"]["tokens"] != res["delta"]["base_tokens"]
    _write(a, res)
    summarize(res)


def summarize(res):
    print("\nvariant        full reload (s)  swap (ms, 1st / again)  tokens equal")
    rows = {}
    for r in res["swap"]["rows"]:
        rows.setdefault(r["variant"], []).append(r)
    for name, fr in res["fresh"].items():
        rs = rows.get(name, [])
        sw = [f"{r['request_to_first_token_s']*1e3:.0f}" for r in rs if not r.get("initial")]
        eq = all(r.get("tokens_equal_fresh") for r in rs)
        print(f"{name:14s} {fr['launch_to_first_token_s'] or fr['process_to_first_token_s']:15.1f}"
              f"  {' / '.join(sw) or '-':22s}  {'yes' if eq else 'NO'}")
    if "delta" in res:
        dl = res["delta"]
        print(f"delta: {dl['plan']['tensors']} tensors, {dl['apply']['uploaded_bytes']/1e6:.0f} MB "
              f"uploaded, request->first token {dl['request_to_first_token_s']*1e3:.0f} ms, "
              f"tokens equal fresh load of the new checkpoint: {dl['tokens_equal_fresh_new']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fresh", "swap", "delta", "all", "make-delta"])
    ap.add_argument("--ckpt")
    ap.add_argument("--variant")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--chip", type=int, default=-1,
                    help="torch device ordinal; -1: the card this pod was given (devices.resolve_chip)")
    ap.add_argument("--cache")
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--mem", action="store_true", help="probe free device capacity")
    ap.add_argument("--delta-new")
    ap.add_argument("--delta-mode", default="inplace", choices=["inplace", "group"])
    ap.add_argument("--ranges", help="onggi diff --ranges --json output (else compare locally)")
    ap.add_argument("--tensors", help="make-delta: comma-separated tensor names")
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--t-launch", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.chip < 0 and a.mode in ("fresh", "swap", "delta"):
        from rngd_pd.devices import resolve_chip
        a.chip = resolve_chip(a.chip)
    {"fresh": cmd_fresh, "swap": cmd_swap, "delta": cmd_delta, "all": cmd_all,
     "make-delta": cmd_make_delta}[a.mode](a)


if __name__ == "__main__":
    main()
