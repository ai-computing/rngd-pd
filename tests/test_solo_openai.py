"""The solo worker's OpenAI front and the benchmark client, with a fake engine.

Streams must carry one chunk per generated token as it is produced, refuse
prompts that do not fit, and read the same way through bench/vendor_sweep.py
as any other OpenAI server.

    PYTHONPATH=src python -m pytest -q tests/test_solo_openai.py
"""
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

from rngd_pd.serving import cluster  # noqa: E402
import vendor_sweep  # noqa: E402


class FakeTokenizer:
    """Token t decodes to the letter chr(97 + t % 26); text encodes per char."""
    bos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens else []) + [ord(c) % 26 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + t % 26) for t in ids)


class FakeEngine:
    def __init__(self, B):
        self.B = B

    def blank_kv(self):
        return object(), object()

    def write_slot_kv(self, Kd, Vd, idx, K, V):
        return Kd, Vd

    def step(self, cur, pos, Kd, Vd):
        time.sleep(0.02)                        # a visible gap between tokens
        return [c + 1 for c in cur], Kd, Vd


class FakeWorker:
    """What SoloOpenAI needs from SoloWorker, over the real scheduler."""

    def __init__(self, B=2):
        self.pre = type("P", (), {"buckets": {16: None, 32: None}})()
        self.batcher = cluster._ContinuousScheduler(
            B, FakeEngine(B), lambda it: (None, None, 3, len(it["tokens"]), 1.0))

    run = cluster.SoloWorker.run


class SoloOpenAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cluster.STATE["ready"] = True
        cls.front = cluster.SoloOpenAI(FakeWorker(), S=48, served_model="m",
                                       tokenizer=FakeTokenizer())
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), cluster.make_handler(
            {"/v1/completions": cls.front.complete}))
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_stream_one_chunk_per_token(self):
        out = []
        vendor_sweep.stream_once(self.url, "m", [0, 5, 6, 7], 6, out)
        r = out[0]
        self.assertIsNone(r["err"])
        self.assertEqual(r["ids"], [3, 4, 5, 6, 7, 8])      # first, then +1 per step
        self.assertEqual(r["text"], "defghi")
        self.assertEqual((r["n"], r["events"], r["prompt_tokens"]), (6, 6, 4))
        self.assertGreater(r["tpot"], 10)                    # tokens arrived spread out
        self.assertLess(r["ttft"], r["wall"])

    def test_text_prompt_and_non_streaming(self):
        body = json.dumps({"model": "m", "prompt": "abc", "max_tokens": 3}).encode()
        req = urllib.request.Request(self.url + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            got = json.loads(r.read())
        self.assertEqual(got["token_ids"], [3, 4, 5])
        self.assertEqual(got["usage"]["prompt_tokens"], 4)  # BOS added
        self.assertEqual(got["choices"][0]["finish_reason"], "length")

    def test_too_long_is_refused(self):
        out = []
        vendor_sweep.stream_once(self.url, "m", list(range(40)), 4, out)
        self.assertIn("400", out[0]["err"])

    def test_concurrent_streams_all_complete(self):
        outs, ths = [], []
        for i in range(5):
            th = threading.Thread(target=vendor_sweep.stream_once,
                                  args=(self.url, "m", [0, i, i], 4, outs))
            th.start(); ths.append(th)
        for th in ths:
            th.join(20)
        self.assertEqual(len(outs), 5)
        self.assertTrue(all(o["err"] is None and o["n"] == 4 for o in outs))


if __name__ == "__main__":
    unittest.main()
