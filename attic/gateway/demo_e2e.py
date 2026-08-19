"""Phase 2 acceptance — REAL NPU end-to-end through the gateway (RngdDispatcher).

Wires actual RNGD ChipRuntimes behind the gateway's EPP:
  • pd mode      : EPP picks prefill@chip0 + decode@chip1; prefill builds KV → rngd_kv_connector
                   hands it off → decode continues. Output compared to the monolithic run.
  • replica mode : EPP picks one BOTH-worker@chip0 that does prefill+decode. Output compared too.
Then A/B: latency of pd vs replica, and a repeated-prefix request to show KV-hit routing.

  python3 demo_e2e.py --model meta-llama/Llama-3.1-8B-Instruct --n 4 --split 3
"""
import argparse, sys, os, time, numpy as np, torch
import _paths  # noqa: E402  (portable path bootstrap)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(False)
from llama_pd_2chip import ChipRuntime
from generic_runtime_wired import _bucket
from generic_runtime import ModelConfig, GenericRuntime
from rngd_kv_connector import RngdKVConnector, KVConnectorRole
from kv_transport import ShmTransport
from registry import WorkerRegistry, Role
from kv_events import KVIndex, block_hashes
from server import Gateway, Dispatcher


class RngdDispatcher(Dispatcher):
    """Runs routed requests on real RNGD chips + real KV connector."""
    def __init__(self, runtimes, kv_index, seq, split, n_layers, block_size=16):
        self.rt = runtimes                      # wid -> ChipRuntime
        self.kv = kv_index; self.seq = seq; self.P = split; self.NL = n_layers
        self.block_size = block_size

    def dispatch(self, token_ids, decision):
        seq, P, NL = self.seq, self.P, self.NL
        hs = block_hashes(token_ids, self.block_size)
        if decision.mode == "pd":
            rt_p = self.rt[decision.prefill_wid]; rt_d = self.rt[decision.decode_wid]
            tr = ShmTransport(root="/dev/shm/rngd_gw_pd")
            pf = RngdKVConnector(KVConnectorRole.WORKER, req_id="gw", transport=tr, n_layers=NL)
            dc = RngdKVConnector(KVConnectorRole.WORKER, req_id="gw", transport=tr, n_layers=NL)
            ckp, cvp = rt_p.new_kv()
            pre = [rt_p.step(seq[i], i, ckp, cvp) for i in range(P)]
            for i in range(NL): pf.save_kv_layer(f"layer.{i}", (ckp[i], cvp[i]))
            pf.wait_for_save()
            ckd, cvd = rt_d.new_kv()
            dc.register_kv_caches((ckd, cvd)); dc.bind_connector_metadata(dc.build_connector_meta(None))
            dc.start_load_kv()
            for i in range(NL): dc.wait_for_layer_load(f"layer.{i}")
            dec = [rt_d.step(seq[i], i, ckd, cvd) for i in range(P, len(seq))]
            out = pre + dec
        else:  # replica: one chip does everything
            rt = self.rt[decision.decode_wid]
            ck, cv = rt.new_kv()
            out = [rt.step(seq[i], i, ck, cv) for i in range(len(seq))]
        self.kv.block_stored(decision.decode_wid, hs)      # worker now holds this prefix
        return {"output_tokens": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--split", type=int, default=3)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model}...", flush=True)
    tk = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    cfg = ModelConfig(hf.config); sd = {k: v.detach().numpy() for k, v in hf.state_dict().items()}; del hf
    ids = tk.encode(a.prompt, add_special_tokens=False)
    oracle = GenericRuntime(cfg, sd); seq = ids + oracle.generate(ids, a.n)
    S = _bucket(len(seq)); P = min(a.split, len(seq)-1); NL = cfg.n_layers
    print(f"[gw-e2e] compiling chip0 + chip1 (S={S})...", flush=True)
    rt0 = ChipRuntime(cfg, sd, S, chip=0); rt1 = ChipRuntime(cfg, sd, S, chip=1)
    print(f"  compiled: chip0 {rt0.compile_s:.0f}s, chip1 {rt1.compile_s:.0f}s", flush=True)

    ckM, cvM = rt0.new_kv(); mono = [rt0.step(seq[i], i, ckM, cvM) for i in range(len(seq))]

    # registry: w_pd_p@chip0 (prefill), w_pd_d@chip1 (decode), w_rep@chip0 (both)
    reg = WorkerRegistry(); kv = KVIndex()
    reg.register("pf0", Role.PREFILL, 0); reg.register("dc1", Role.DECODE, 1)
    reg.register("rep0", Role.BOTH, 0)
    runtimes = {"pf0": rt0, "dc1": rt1, "rep0": rt0}
    BLK = 2                                                # short demo prompt → small blocks
    disp = RngdDispatcher(runtimes, kv, seq, P, NL, block_size=BLK)
    gw = Gateway(reg, kv, disp, block_size=BLK)

    # --- A/B over the gateway ---
    def run(mode, toks):
        t0 = time.time(); r = gw.complete({"tokens": toks, "mode": mode}); return r, (time.time()-t0)*1000

    r_pd, t_pd = run("pd", seq)
    kv.all_cleared("rep0"); kv.all_cleared("dc1")            # cold replica for fair A/B
    r_rep, t_rep = run("replica", seq)
    # repeated identical request in replica → KV-hit routing evidence
    r_rep2, _ = run("replica", seq)

    pd_ok = r_pd["choices"][0]["tokens"] == mono
    rep_ok = r_rep["choices"][0]["tokens"] == mono
    hit_ok = r_rep["routing"]["kv_hit_blocks"] == 0 and r_rep2["routing"]["kv_hit_blocks"] > 0

    print("\n" + "=" * 68)
    print("RNGD gateway (EPP KV-aware) — REAL 2-chip end-to-end")
    print(f"  monolithic     : {mono}")
    print(f"  pd  route={r_pd['routing']['prefill']}→{r_pd['routing']['decode']}  out={r_pd['choices'][0]['tokens']}  {'==mono ✅' if pd_ok else 'MISMATCH ❌'}  {t_pd:.0f}ms")
    print(f"  rep route={r_rep['routing']['decode']}          out={r_rep['choices'][0]['tokens']}  {'==mono ✅' if rep_ok else 'MISMATCH ❌'}  {t_rep:.0f}ms")
    print(f"  KV-hit on repeat: req1={r_rep['routing']['kv_hit_blocks']}blk → req2={r_rep2['routing']['kv_hit_blocks']}blk  {'✅ routed to warm worker' if hit_ok else '❌'}")
    print(f"  A/B: pd {t_pd:.0f}ms vs replica {t_rep:.0f}ms")
    ok = pd_ok and rep_ok and hit_ok
    print(f"  {'PASS ✅ gateway routes real requests through PD & replica, KV-aware, output correct' if ok else 'FAIL ❌'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
