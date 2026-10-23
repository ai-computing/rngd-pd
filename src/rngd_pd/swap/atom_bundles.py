"""Select co-compiled ATOM decoders without loading their weights again.

An optimum-rbln bundle is compiled with decoder_batch_sizes in ONE
CompileContext(use_weight_sharing=True). Its prefill and decoder runtimes
must be kept alive together. An independently compiled bundle has separate
weights: prepare_async() overlaps its full load with serving the active one,
provided both fit. No device addresses or compiled files are patched.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
import time

from .variants import Variant


@dataclass(frozen=True)
class AtomBundle:
    path: str
    variants: tuple

    def __post_init__(self):
        if not self.variants or any(not isinstance(v, Variant) for v in self.variants):
            raise ValueError("a bundle needs compiled Variant objects")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("duplicate variants in a bundle")


def _load_bundle(path, device):
    from optimum.rbln import RBLNAutoModelForCausalLM
    return RBLNAutoModelForCausalLM.from_pretrained(
        path, export=False, rbln_config={"device": device})


class SwappableAtom:
    """ATOM engine with swap(), raw run() and greedy generate().

    Use the request methods here so a swap waits for the current request to
    drain. A shape change starts a NEW request; KV from the old shape is not
    migrated. Background loading uses one thread, not an agent or process.
    """

    def __init__(self, bundles, device=0, loader=None):
        if not bundles or any(not isinstance(b, AtomBundle) for b in bundles.values()):
            raise ValueError("bundles must map names to AtomBundle objects")
        self.bundles = dict(bundles)
        self.device = device
        self._loader = loader or _load_bundle
        self._lock = RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atom-warm")
        self._pending = {}
        self._loaded = {}
        self.active = self.active_bundle = None
        self._closed = False

    def _resolve(self, variant, bundle=None):
        if self._closed:
            raise RuntimeError("engine is closed")
        names = [n for n, b in self.bundles.items() if variant in b.variants]
        if bundle is None:
            if len(names) != 1:
                raise ValueError("variant is missing or ambiguous; specify bundle")
            bundle = names[0]
        if bundle not in names:
            raise ValueError(f"{variant.name} is not compiled in bundle {bundle!r}")
        return bundle

    def _load(self, name):
        start = time.perf_counter()
        model = self._loader(self.bundles[name].path, self.device)
        config = model.rbln_config
        for v in self.bundles[name].variants:
            if v.S != config.max_seq_len or v.B not in config.decoder_batch_sizes:
                raise ValueError(f"bundle {name!r} does not contain {v.name}")
        names = list(config.expected_compiled_model_names)
        # Keep the model, including the prefill weight owner, alive. Keeping
        # only a decoder runtime can release the shared context too early.
        runtimes = {v: model.model[names.index(f"decoder_batch_{v.B}")]
                    for v in self.bundles[name].variants}
        return dict(model=model, runtimes=runtimes,
                    prepare_s=time.perf_counter() - start,
                    prepared_at_s=time.perf_counter())

    def prepare_async(self, variant, bundle=None):
        """Start a separate bundle's full load; failures stay on the Future.

        Loading does not change the active engine. A failed or memory-limited
        warm load can be retried explicitly with prepare_async().
        """
        with self._lock:
            name = self._resolve(variant, bundle)
            future = self._pending.get(name)
            if future is None or (future.done() and future.exception() is not None):
                if name in self._loaded:
                    return None
                future = self._pool.submit(self._load, name)
                self._pending[name] = future
            return future

    def prepare(self, variant, bundle=None):
        with self._lock:
            name = self._resolve(variant, bundle)
            if name in self._loaded:
                return dict(prepared=False, prepare_s=0.0)
            future = self.prepare_async(variant, name)
        # A slow load must not hold the request lock: current serving proceeds.
        loaded = future.result()
        with self._lock:
            self._resolve(variant, name)
            self._loaded[name] = loaded
            return dict(prepared=True, prepare_s=loaded["prepare_s"])

    def swap(self, variant, bundle=None, require_ready=False):
        start = time.perf_counter()
        with self._lock:
            name = self._resolve(variant, bundle)
            ready = name in self._loaded
            future = self._pending.get(name)
            background_ready = (future is not None and future.done()
                                and future.exception() is None)
            if require_ready and not ready and (future is None or not future.done()):
                raise RuntimeError(f"bundle {name!r} is not prepared")
            if require_ready and not ready:
                self._loaded[name] = future.result()  # propagate a failed warm load; do not retry
        info = self.prepare(variant, name)
        with self._lock:
            self._resolve(variant, name)
            if name not in self._loaded:
                raise RuntimeError(f"bundle {name!r} was released before activation")
            same = name == self.active_bundle
            strategy = "shared-session" if same else (
                "warm-pool" if ready or background_ready else "full-load")
            self.active_bundle, self.active = name, variant
            return dict(info, variant=variant.name, bundle=name, strategy=strategy,
                        weights_uploaded_on_swap=not ready and strategy == "full-load",
                        swap_s=time.perf_counter() - start)

    def _state(self):
        if self._closed or self.active is None:
            raise RuntimeError("no active ATOM variant")
        return self._loaded[self.active_bundle]

    def run(self, *inputs, **kwargs):
        """Run inputs for the selected compiled decoder; output is synchronous."""
        with self._lock:
            return self._state()["runtimes"][self.active].run(*inputs, **kwargs)

    def generate(self, prompts, n_new, timed=False):
        """Greedy tokens for B equal-length token-id prompts, with fresh KV."""
        start = time.perf_counter()
        import torch
        with self._lock:
            state = self._state()
            v = self.active
            if (len(prompts) != v.B or not prompts or not prompts[0]
                    or len({len(p) for p in prompts}) != 1 or n_new < 1
                    or len(prompts[0]) + n_new > v.S):
                raise ValueError("prompts/generation do not fit the selected variant")
            ids = torch.tensor(prompts, dtype=torch.int64)
            class FirstToken:
                def __init__(self):
                    self.prompt_seen = False
                    self.seconds = None

                def put(self, value):
                    if not self.prompt_seen:
                        self.prompt_seen = True
                    elif self.seconds is None:
                        self.seconds = time.perf_counter() - start

                def end(self):
                    pass

            timer = FirstToken()
            result = state["model"].generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                             max_new_tokens=n_new, do_sample=False,
                                             **({"streamer": timer} if timed else {}))
            tokens = result[:, ids.shape[1]:].tolist()
            if timed:
                if timer.seconds is None:
                    raise RuntimeError("runtime did not emit first-token timing")
                return tokens, timer.seconds
            return tokens

    def select(self, selector, depth, context, bundle=None):
        """Queue-depth hook; invoke at a request boundary after draining KV."""
        with self._lock:
            self._state()
            name = bundle or self.active_bundle
            candidates = [v for v in self.bundles[name].variants
                          if v.B == selector.fit(depth) and v.S >= context]
            if not candidates:
                raise ValueError("no compiled variant fits depth and context")
            batch = selector.observe(depth)
            candidates = [v for v in self.bundles[name].variants
                          if v.B == batch and v.S >= context]
            if not candidates:
                raise ValueError("selector batch is not in the bundle")
            return self.swap(min(candidates, key=lambda v: v.S), name)

    def release(self, bundle):
        with self._lock:
            if bundle == self.active_bundle:
                raise ValueError("cannot release the active bundle")
            future = self._pending.get(bundle)
            if future is not None and not future.done():
                raise RuntimeError("cannot release a bundle while it is loading")
            self._loaded.pop(bundle, None)
            self._pending.pop(bundle, None)

    def allocations(self):
        """SDK allocation evidence per bundle and graph, without filling HBM."""
        with self._lock:
            return {name: [rt.get_alloc_per_node_by_key() for rt in st["model"].model]
                    for name, st in self._loaded.items()}

    def close(self):
        self._pool.shutdown(wait=True)
        with self._lock:
            self._closed = True
            self.active = self.active_bundle = None
            self._loaded.clear()
            self._pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
