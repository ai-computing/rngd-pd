"""Phase 5 verify: does a compiled CompileModule round-trip through the artifact cache — reload in a
FRESH process, produce identical output, and load in seconds (not recompile)?

Phase 'compile': compile a small module, run it, SAVE via artifact_cache, print output checksum + compile time.
Phase 'load'   : in a fresh subprocess, LOAD the artifact, run same input, print checksum + load time.
Parent compares checksums and asserts load << compile → cold-start removal is real.

  python3 probe_artifact_cache.py
"""
import sys, os, time, hashlib, subprocess
import numpy as np

CACHE = "/dev/shm/rngd_artifact_probe"
KEY = "probe_small_v1"
NKV, HD, S = 8, 128, 16


def _build_and_inputs():
    import torch, furiosa.torch
    from furiosa.torch import native_device as nd
    nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
    from furiosa.torch.custom_ops import CompileModule
    BF = torch.bfloat16
    class Inject(torch.nn.Module):
        def forward(self, Kp, k, oh):
            return Kp + k.view(NKV, HD, 1) * oh.view(1, 1, S)
    sample = (torch.zeros(NKV, HD, S, dtype=BF), torch.zeros(NKV, HD, dtype=BF), torch.zeros(S, dtype=BF))
    def _d(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to("rngd:0")
    rng = np.random.default_rng(7)
    inp = (_d(rng.standard_normal((NKV, HD, S))), _d(rng.standard_normal((NKV, HD))),
           _d(np.eye(S)[3].astype(np.float32)))
    return Inject, sample, inp


def _ck(t): return hashlib.blake2b(np.ascontiguousarray(t.detach().float().cpu().numpy()).tobytes(), digest_size=8).hexdigest()


def phase_compile():
    import torch, furiosa.torch  # noqa
    from furiosa.torch.custom_ops import CompileModule
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving"))
    from rngd_pd.engines import artifact_cache as AC
    Inject, sample, inp = _build_and_inputs()
    def compile_fn():
        m = CompileModule.from_module(Inject(), sample); m.to("rngd:0"); return m
    m, mode, secs = AC.get_or_compile(CACHE, KEY, compile_fn, device="rngd:0")
    out = m(*inp)
    p = AC.path_for(CACHE, KEY)
    size = os.path.getsize(p + ".edf") + os.path.getsize(p + ".w") if AC.exists(p) else -1
    print(f"COMPILE mode={mode} secs={secs:.1f} ck={_ck(out)} size={size}")


def phase_load():
    import torch, furiosa.torch  # noqa
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving"))
    from rngd_pd.engines import artifact_cache as AC
    _, _, inp = _build_and_inputs()
    m, mode, secs = AC.get_or_compile(CACHE, KEY, lambda: (_ for _ in ()).throw(RuntimeError("should not compile!")), device="rngd:0")
    out = m(*inp)
    print(f"LOAD mode={mode} secs={secs:.2f} ck={_ck(out)}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("compile", "load"):
        {"compile": phase_compile, "load": phase_load}[sys.argv[1]](); return
    env = dict(os.environ)
    def run(ph):
        r = subprocess.run([sys.executable, __file__, ph], capture_output=True, text=True, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith(ph.upper())]
        if not line:
            print(r.stdout[-2000:]); print(r.stderr[-2000:]); raise SystemExit(f"{ph} phase failed")
        return line[-1]
    c = run("compile"); l = run("load")
    cparts = dict(p.split("=") for p in c.split()[1:]); lparts = dict(p.split("=") for p in l.split()[1:])
    same = cparts["ck"] == lparts["ck"]
    fast = float(lparts["secs"]) < float(cparts["secs"])
    print("=== Phase 5 artifact-cache probe ===")
    print(f"  {c}")
    print(f"  {l}")
    print(f"  identical output: {'✅' if same else '❌'} | load faster than compile: {'✅' if fast else '❌'} "
          f"({float(cparts['secs']):.1f}s → {float(lparts['secs']):.2f}s)")
    ok = same and fast and lparts["mode"] == "load"
    print(f"  {'PASS ✅ compiled EDF persists & reloads in a fresh process — cold-start removed' if ok else 'FAIL ❌ EDF not serializable; see fallback in artifact_cache'}")


if __name__ == "__main__":
    main()
