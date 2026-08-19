"""Fast RNGD 2-chip PD disagg THROUGH the llm-d/vLLM-shaped KV connector — performance + generality.

Same real cross-chip flow as llama_pd_2chip.py (prefill@rngd:0 produces KV → decode@rngd:1 consumes),
but the KV now crosses the boundary via the standard KVConnectorBase_V1-shaped interface
(RngdKVConnector.save_kv_layer on the prefill worker → transport → wait_for_layer_load on the decode
worker). This is the llm-d protocol path: our runtime becomes a drop-in KV-connector backend while
keeping RNGD-native performance. Verified == monolithic.

  python3 llama_pd_llmd.py --model meta-llama/Llama-3.1-8B-Instruct --n 4 --split 3
"""
import argparse, sys, numpy as np, torch
import _paths  # noqa: E402  (portable path bootstrap)
torch.set_grad_enabled(False)
from llama_pd_2chip import ChipRuntime
from generic_runtime_wired import _bucket
from generic_runtime import ModelConfig, GenericRuntime
from rngd_kv_connector import RngdKVConnector, KVConnectorRole
from kv_transport import make_transport


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--split", type=int, default=3)
    ap.add_argument("--transport", default="shm", choices=["inproc", "shm"])
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n)
    S = _bucket(len(seq)); P = min(a.split, len(seq)-1); NL = cfg.n_layers
    print(f"[pd-llmd] compiling prefill@rngd:0 + decode@rngd:1 (S={S})...", flush=True)
    rt0 = ChipRuntime(cfg, sd, S, chip=0); rt1 = ChipRuntime(cfg, sd, S, chip=1)
    print(f"  compiled: chip0 {rt0.compile_s:.0f}s, chip1 {rt1.compile_s:.0f}s", flush=True)

    mono = None
    ckM, cvM = rt0.new_kv(); mono = [rt0.step(seq[p], p, ckM, cvM) for p in range(len(seq))]

    # ---- PD via the REAL vLLM KVConnectorBase_V1 subclass ----
    transport = make_transport("shm", root="/dev/shm/rngd_kv_pd") if a.transport == "shm" else make_transport("inproc")
    pf_conn = RngdKVConnector(KVConnectorRole.WORKER, req_id="req1", transport=transport, n_layers=NL)
    dc_conn = RngdKVConnector(KVConnectorRole.WORKER, req_id="req1", transport=transport, n_layers=NL)

    # PREFILL @chip0 → produce KV, then SAVE each layer via the connector (save_kv_layer)
    ck0, cv0 = rt0.new_kv()
    pre = [rt0.step(seq[p], p, ck0, cv0) for p in range(P)]
    for i in range(NL):
        pf_conn.save_kv_layer(f"layer.{i}", (ck0[i], cv0[i]))
    pf_conn.wait_for_save()

    # DECODE @chip1 → LOAD each layer via the connector (start_load_kv / wait_for_layer_load)
    ck1, cv1 = rt1.new_kv()
    dc_conn.register_kv_caches((ck1, cv1))
    dc_conn.bind_connector_metadata(dc_conn.build_connector_meta(None))
    dc_conn.start_load_kv()
    for i in range(NL):
        dc_conn.wait_for_layer_load(f"layer.{i}")   # fills ck1/cv1 from transport
    dec = [rt1.step(seq[p], p, ck1, cv1) for p in range(P, len(seq))]
    pd = pre + dec

    ok = pd == mono
    kv_mb = sum(len(transport.pull(f"req1/layer.{i}/k")) + len(transport.pull(f"req1/layer.{i}/v")) for i in range(NL))/1e6
    print("\n" + "=" * 66)
    print(f"RNGD 2-chip PD via llm-d/vLLM KVConnectorBase_V1-shaped interface")
    print(f"  prefill@rngd:0 (0..{P-1}) --save_kv_layer×{NL}--> transport --wait_for_layer_load×{NL}--> decode@rngd:1 ({P}..)")
    print(f"  KV over connector: {kv_mb:.1f} MB (fp16, shm transport)")
    print(f"  monolithic: {mono}")
    print(f"  PD (connector): {pd}")
    print(f"  {'PASS ✅ standard-interface KV handoff, 2 physical chips, output identical' if ok else 'FAIL ❌'}")
    print(f"  → performance: RNGD-native wired path.  generality: llm-d/vLLM KV-connector protocol.")
    print("=" * 66)


if __name__ == "__main__":
    main()
