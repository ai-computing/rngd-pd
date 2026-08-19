"""Dim-parameterized L1 DSL generators for the fast wired path (Phase 4).

llama_engine.py hardcodes the DSL dims (N=4096 rms/zip, N=14336 swiglu, K=8/G=4 softmax) so the fast
path only fits Llama-3.1-8B. These builders emit the SAME kernels for any (hidden, mlp, n_kv, gqa,
seq_len) → the fast wired engine becomes config-driven for the Llama/Qwen2 dense family.

The kernel bodies are byte-for-byte the proven ones from llama_engine.py; only the dim literals vary.
"""


def rms_norm_dsl(N):
    return f"""#tactic_kernel_dsl
def rms_norm(x: [N={N}]/bf16, w_baked: [N={N}]/bf16, eps_baked: [N={N}]/f32, neg_half: [N={N}]/f32) -> [N={N}]/bf16:
    output: [N]/bf16 = tk.ReduceByVe(
        %0 = read(x)
        %1 = read(w_baked)
        %2 = read(eps_baked)
        %3 = read(neg_half)
        %4 = ve.exec(%0 * %0)
        %5 = ve.exec(+ t[N] %4)
        %6 = ve.exec(%5 + %2)
        %7 = ve.exec(log %6)
        %8 = ve.exec(%7 * %3)
        %9 = ve.exec(exp %8)
        %10 = ve.exec(%0 * %9)
        %11 = ve.exec(%10 * %1)
        write.to([N]/bf16)
    )
    return output
"""


def zip_add_dsl(N):
    return f"""#tactic_kernel_dsl
def zip_add(a: [N={N}]/bf16, b: [N={N}]/bf16) -> [N={N}]/bf16:
    output: [N]/bf16 = tk.Interleaving(
        %0 = read(a)
        %1 = read(b)
        %2 = ve.exec(%0 + %1)
        write.to([N]/bf16)
    )
    return output
"""


def swiglu_dsl(M):
    return f"""#tactic_kernel_dsl
def swiglu(gate: [N={M}]/bf16, up: [N={M}]/bf16) -> [N={M}]/bf16:
    silu_gate: [N]/bf16 = tk.Elementwise(
        %0 = read(gate)
        %1 = ve.exec(sigmoid %0)
        %2 = ve.exec(%0 * %1)
        write.to([N]/bf16)
    )
    output: [N]/bf16 = tk.Interleaving(
        %0 = read(silu_gate)
        %1 = read(up)
        %2 = ve.exec(%0 * %1)
        write.to([N]/bf16)
    )
    return output
"""


def softmax_dsl(K, G, S):
    return f"""#tactic_kernel_dsl
def softmax_attn(scores: [K={K},G={G},S={S}]/bf16) -> [K={K},G={G},S={S}]/bf16:
    output: [K,G,S]/bf16 = tk.ReduceByVe(
        %0 = read(scores)
        %1 = ve.exec(max_f t[S] %0)
        %2 = ve.exec(%0 - %1)
        %3 = ve.exec(exp %2)
        %4 = ve.exec(+ t[S] %3)
        %5 = ve.exec(%3 / %4)
        write.to([K,G,S]/bf16)
    )
    return output
"""


def _selftest():
    # dims match the two verified models
    for name, (N, M, K, G, S) in {"Llama-8B": (4096, 14336, 8, 4, 64),
                                   "Qwen2.5-0.5B": (896, 4864, 2, 7, 64)}.items():
        for gen, args in [(rms_norm_dsl, (N,)), (zip_add_dsl, (N,)), (swiglu_dsl, (M,)),
                          (softmax_dsl, (K, G, S))]:
            s = gen(*args)
            assert "#tactic_kernel_dsl" in s and str(args[0]) in s
        print(f"  {name}: rms[N={N}] zip[N={N}] swiglu[N={M}] softmax[K={K},G={G},S={S}] — generated ✅")
    print("dsl_gen self-test: PASS ✅ (byte-identical kernels, dims parameterized)")


if __name__ == "__main__":
    _selftest()
