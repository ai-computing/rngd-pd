"""Phase 1 acceptance: RngdKVConnector over TcpTransport across SEPARATE PROCESSES, bit-exact.

Proves the real KVConnectorBase_V1 subclass moves per-layer KV between two OS processes (prefill
worker → decode worker) via the cross-node TCP transport — the xPyD path where prefill and decode
run on different nodes. No NPU needed (numpy KV); this isolates the connector+transport contract.

  python3 test_connector_tcp.py
"""
import sys, os, time, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402  (portable path bootstrap — resolves connector/ etc.)

NL, NKV, S, HD, PORT = 6, 8, 16, 128, 5577
SEED = 1234


def _ref():
    rng = np.random.default_rng(SEED)
    ck = [rng.standard_normal((NKV, S, HD)).astype(np.float32) for _ in range(NL)]
    cv = [rng.standard_normal((NKV, S, HD)).astype(np.float32) for _ in range(NL)]
    return ck, cv


def prefill_proc(ready):
    from rngd_pd.connector.rngd_kv import RngdKVConnector, KVConnectorRole
    from rngd_pd.connector.transport import TcpTransport
    ready.wait()                                            # wait until server is up
    tr = TcpTransport(remote=("127.0.0.1", PORT))
    conn = RngdKVConnector(KVConnectorRole.WORKER, req_id="reqT", transport=tr, n_layers=NL)
    ck, cv = _ref()
    for i in range(NL):
        conn.save_kv_layer(f"layer.{i}", (ck[i], cv[i]))
    conn.wait_for_save()
    tr.close()


def decode_proc(ready, out_path):
    from rngd_pd.connector.rngd_kv import RngdKVConnector, KVConnectorRole
    from rngd_pd.connector.transport import TcpTransport
    ready.wait()
    tr = TcpTransport(remote=("127.0.0.1", PORT))
    dk = [np.zeros((NKV, S, HD), np.float32) for _ in range(NL)]
    dv = [np.zeros((NKV, S, HD), np.float32) for _ in range(NL)]
    conn = RngdKVConnector(KVConnectorRole.WORKER, req_id="reqT", transport=tr, n_layers=NL)
    conn.register_kv_caches((dk, dv))
    conn.bind_connector_metadata(conn.build_connector_meta(None))
    conn.start_load_kv()
    for i in range(NL):
        conn.wait_for_layer_load(f"layer.{i}")             # blocks until prefill has pushed it
    np.savez(out_path, **{f"k{i}": dk[i] for i in range(NL)}, **{f"v{i}": dv[i] for i in range(NL)})
    tr.close()


def main():
    from rngd_pd.connector.base import TcpKVServer
    server = TcpKVServer(PORT); server.start()             # decode-node KV server
    time.sleep(0.2)
    ctx = mp.get_context("spawn")
    ready = ctx.Event(); ready.set()
    out = "/dev/shm/rngd_tcp_decode.npz"
    if os.path.exists(out): os.remove(out)
    # start decode first (it blocks pulling), then prefill pushes
    dp = ctx.Process(target=decode_proc, args=(ready, out)); dp.start()
    pp = ctx.Process(target=prefill_proc, args=(ready,)); pp.start()
    pp.join(120); dp.join(120)
    server.stop()

    ck, cv = _ref()
    z = np.load(out)
    ref16 = lambda a: a.astype(np.float16).astype(np.float32)
    errk = max(np.abs(z[f"k{i}"] - ref16(ck[i])).max() for i in range(NL))
    errv = max(np.abs(z[f"v{i}"] - ref16(cv[i])).max() for i in range(NL))
    err = max(errk, errv)
    print("=== Phase 1 TCP cross-process connector test ===")
    print(f"  2 OS processes (prefill→decode) · TcpTransport · {NL} layers · {NKV}×{S}×{HD}")
    print(f"  transport loss {err:.6f} (0 = bit-exact for fp16 storage)")
    print(f"  {'PASS ✅ real KVConnectorBase_V1 subclass moves KV across processes via TCP' if err < 1e-6 else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
