"""Phase 3 de-risk probe: can we keep the KV cache RESIDENT on device and feed the module's updated
cache output back as its own next input (device→device), with no host round-trip and no NaN?

The fullwire runtime re-uploads the whole KV (16.8MB×2) every token via _d(Kp). If instead the module
RETURNS the injected cache Kf = Kp + k·onehot and we feed Kf back as next-token Kp — all device
tensors — the per-token KV upload disappears. The summary warned device-output chaining can NaN;
this tiny/fast probe tests it before the ~30min full compile.

  python3 probe_resident_kv.py
"""
import torch, furiosa.torch
from furiosa.torch import native_device as nd
nd.set_fusion(8); nd.set_mesh_kind(nd.MeshKind.Mesh(2))
import numpy as np
from furiosa.torch.custom_ops import CompileModule

DEV = "rngd:0"; BF = torch.bfloat16
NKV, HD, S = 8, 128, 16


class Inject(torch.nn.Module):
    """One decode step's KV inject: Kf[NKV,HD,S] = Kp + k[NKV,HD,1]·oh[1,1,S]."""
    def forward(self, Kp, k, oh):
        return Kp + k.view(NKV, HD, 1) * oh.view(1, 1, S)


def _d(a): return torch.from_numpy(np.ascontiguousarray(a)).to(BF).to(DEV)


def main():
    m = CompileModule.from_module(Inject(), (torch.zeros(NKV, HD, S, dtype=BF),
                                             torch.zeros(NKV, HD, dtype=BF),
                                             torch.zeros(S, dtype=BF)))
    m.to(DEV)
    rng = np.random.default_rng(0)
    ks = [rng.standard_normal((NKV, HD)).astype(np.float32) for _ in range(S)]

    # --- RESIDENT path: Kp stays on device, fed back from the module's own output ---
    Kp_dev = _d(np.zeros((NKV, HD, S), np.float32))     # uploaded ONCE
    uploads = 1
    for pos in range(S):
        oh = np.zeros(S, np.float32); oh[pos] = 1.0
        Kf = m(Kp_dev, _d(ks[pos]), _d(oh))             # k,oh are tiny per-token uploads
        Kp_dev = Kf                                     # device→device feedback, NO re-upload
    got = Kp_dev.detach().float().cpu().numpy()

    # --- reference: numpy accumulation (bf16-rounded) ---
    ref = np.zeros((NKV, HD, S), np.float32)
    bf = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(BF).float().numpy()
    for pos in range(S):
        ref[:, :, pos] = bf(ks[pos])
    err = np.abs(got - ref).max()
    nan = np.isnan(got).any()
    print("=== Phase 3 resident-KV chaining probe ===")
    print(f"  {S} steps, KV resident on device, output fed back as input ({uploads} upload vs {S} in the re-upload path)")
    print(f"  max err vs numpy: {err:.5f} | NaN: {nan}")
    ok = (not nan) and err < 0.05
    print(f"  {'PASS ✅ device→device KV feedback works — resident KV is viable' if ok else 'FAIL ❌ chaining breaks (NaN/err) — need in-place or host merge'}")


if __name__ == "__main__":
    main()
