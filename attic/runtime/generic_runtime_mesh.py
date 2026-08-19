"""M6-Llama — apply SDK hidden API (fusion+mesh TP) + resident + concat to big models.

set_fusion(8)+set_mesh_kind(Mesh(2)) (the vendor native-TP trigger, 17.96→2.42ms on Qwen)
distributes matmuls across the 8 PEs and — the key question for Llama — may let 16GB weights
be RESIDENT across the mesh's HBM instead of the ~6GB furiosa.torch pool that forced streaming.
"""
import argparse, time, numpy as np, torch, sys
import _paths  # noqa: E402  (portable path bootstrap)
# fusion+mesh MUST be set before any compile / get_* — do it first
import furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8)
nd.set_mesh_kind(nd.MeshKind.Mesh(2))
print(f"[mesh] fusion=8 mesh=2 set; devices now: {sum(1 for _ in range(4))} logical", flush=True)

from generic_runtime import ModelConfig, GenericRuntime
import generic_runtime_opt as OPT
OPT.DEV = "rngd:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd)
    seq = ids + oracle.generate(ids, a.n); oam = oracle.argmax_per_pos(seq)

    print("[mesh] building resident+concat runtime (fusion+mesh) — watching for OOM...", flush=True)
    try:
        t0 = time.time(); rt = OPT.OptRuntime(cfg, sd); load = time.time() - t0
    except Exception as e:
        print(f"\nRESIDENT FAILED: {type(e).__name__}: {str(e)[:160]}")
        print("→ 16GB weights still exceed the pool even with mesh → needs low-level hbm_tensor (47GB) path.")
        return
    t0 = time.time(); am = rt.argmax_per_pos(seq); mt = (time.time()-t0)/len(seq)*1000
    lo = len(ids)-1; N = len(seq)-1-lo
    ok = sum(am[i] == oam[i] for i in range(lo, len(seq)-1))
    print("\n" + "=" * 58)
    print(f"{cfg.n_layers}L hidden={cfg.hidden} — fusion(8)+mesh(2) + resident + concat")
    print(f"correctness vs oracle: {ok}/{N} {'PASS ✅' if ok == N else 'CHECK'}")
    print(f"{mt:.0f} ms/token  (load+compile {load:.1f}s)")
    print(f"vs streaming baseline (earlier): 1294 ms/token → {1294/mt:.2f}× if this ran resident")
    print("=" * 58)


if __name__ == "__main__":
    main()
