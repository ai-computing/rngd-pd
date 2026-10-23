from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import pytest

from rngd_pd.swap.atom import AtomBundle, SwappableAtom
from rngd_pd.swap.variants import Variant


V1, V4 = Variant(1, 32), Variant(4, 32)


class Runtime:
    def __init__(self, name):
        self.name = name

    def run(self, *inputs):
        return self.name, inputs


def model(path, device):
    return SimpleNamespace(
        rbln_config=SimpleNamespace(max_seq_len=32, decoder_batch_sizes=[1, 4],
            expected_compiled_model_names=["prefill", "decoder_batch_1", "decoder_batch_4"]),
        model=[Runtime(path + n) for n in ("prefill", "B1", "B4")])


def test_switch_keeps_weight_owner_and_never_reloads():
    calls = []
    def load(path, device):
        calls.append((path, device))
        return model(path, device)
    with SwappableAtom({"a": AtomBundle("a", (V1, V4))}, loader=load) as e:
        assert e.swap(V1)["strategy"] == "full-load"
        owner = e._loaded["a"]["model"]
        assert e.swap(V4)["strategy"] == "shared-session"
        assert e.run(7) == ("aB4", (7,))
        e.swap(V1)
        assert e._loaded["a"]["model"] is owner
        assert calls == [("a", 0)]
        with pytest.raises(ValueError):
            e.release("a")


def test_warm_bundle_loading_does_not_block_current_requests():
    entered, finish = Event(), Event()
    def load(path, device):
        if path == "b":
            entered.set()
            assert finish.wait(5)
        return model(path, device)
    bundles = {n: AtomBundle(n, (V1, V4)) for n in ("a", "b")}
    with SwappableAtom(bundles, loader=load) as e:
        e.swap(V1, "a")
        future = e.prepare_async(V1, "b")
        try:
            assert entered.wait(2)
            assert e.run(9) == ("aB1", (9,))
            with pytest.raises(RuntimeError, match="not prepared"):
                e.swap(V4, "b", require_ready=True)
            assert e.active_bundle == "a"
        finally:
            finish.set()
        future.result(5)
        info = e.swap(V4, "b", require_ready=True)
        assert info["strategy"] == "warm-pool"
        assert not info["weights_uploaded_on_swap"]
        e.release("a")
        assert "a" not in e._loaded


def test_failed_prepare_leaves_active_model_usable():
    def load(path, device):
        if path == "b":
            raise MemoryError("HBM full")
        return model(path, device)
    with SwappableAtom({n: AtomBundle(n, (V1,)) for n in ("a", "b")}, loader=load) as e:
        e.swap(V1, "a")
        with pytest.raises(MemoryError):
            e.swap(V1, "b")
        assert e.run() == ("aB1", ())


@pytest.mark.parametrize("action", ["release", "close"])
def test_swap_revalidates_bundle_after_preparing(monkeypatch, action):
    prepared, activate = Event(), Event()
    with SwappableAtom({n: AtomBundle(n, (V1,)) for n in ("a", "b")}, loader=model) as e:
        e.swap(V1, "a")
        prepare = e.prepare

        def pause_after_prepare(variant, bundle=None):
            info = prepare(variant, bundle)
            prepared.set()
            assert activate.wait(5)
            return info

        monkeypatch.setattr(e, "prepare", pause_after_prepare)
        with ThreadPoolExecutor(max_workers=1) as pool:
            swap = pool.submit(e.swap, V1, "b")
            try:
                assert prepared.wait(2)
                if action == "release":
                    e.release("b")
                else:
                    e.close()
            finally:
                activate.set()
            with pytest.raises(RuntimeError, match="released|closed"):
                swap.result(5)
        if action == "release":
            assert e.active_bundle == "a" and e.active == V1
            assert e.run() == ("aB1", ())
        else:
            assert e.active_bundle is None and e.active is None


def test_wrong_compiled_shape_is_rejected_before_activation():
    with SwappableAtom({"a": AtomBundle("a", (Variant(2, 32),))}, loader=model) as e:
        with pytest.raises(ValueError, match="does not contain"):
            e.swap(Variant(2, 32))
        assert e.active is None


def test_ambiguous_variant_requires_bundle_and_close_is_final():
    e = SwappableAtom({n: AtomBundle(n, (V1,)) for n in ("a", "b")}, loader=model)
    try:
        with pytest.raises(ValueError, match="ambiguous"):
            e.swap(V1)
    finally:
        e.close()
    with pytest.raises(RuntimeError, match="closed"):
        e.swap(V1, "a")


def test_generation_timing_records_first_token_and_selector_routes_batch():
    import torch
    from rngd_pd.swap import BatchSelector
    def load(path, device):
        m = model(path, device)
        def generate(input_ids, attention_mask, max_new_tokens, do_sample, streamer):
            streamer.put(input_ids)
            streamer.put(torch.full((input_ids.shape[0],), 7))
            streamer.end()
            return torch.cat([input_ids, torch.full((input_ids.shape[0], max_new_tokens), 7)], 1)
        m.generate = generate
        return m
    with SwappableAtom({"a": AtomBundle("a", (V1, V4))}, loader=load) as e:
        e.swap(V1)
        tokens, ttft = e.generate([[1, 2]], 3, timed=True)
        assert tokens == [[7, 7, 7]] and ttft >= 0
        selector = BatchSelector([1, 4], start=1, up_after=1)
        e.select(selector, depth=3, context=16)
        assert e.active == V4
        with pytest.raises(ValueError, match="fits"):
            e.select(selector, depth=3, context=64)
