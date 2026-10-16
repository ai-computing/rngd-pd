"""SDK paged-attention decode; upload each request's KV once at admission."""
import importlib.metadata
import time
import numpy as np
from .atom import AtomDecoder
from .dlfloat import bf16_expanded_to_dlfloat


class AtomResidentDecoder(AtomDecoder):
    def new_buffers(self):
        if self.direct_admit:
            from rebel.kv_cache import aligned_tensor
            from .packed_kv import PackedKVView
            storage = aligned_tensor(64*self.batch*8*512*128, dtype=np.uint16,
                                     alignment=2*1024*1024, tensor_type="np")
            storage.fill(0)
            return PackedKVView(storage, self.batch, 0), PackedKVView(storage, self.batch, 1)
        if self.huge_host:
            raise ValueError('resident KV already has aligned admission staging; omit --huge-host')
        keys = (np.zeros((32, self.batch, 8, 512, 128), np.uint16).transpose(0, 1, 2, 4, 3)
                if self.token_major_host else np.zeros((32, self.batch, 8, 128, 512), np.uint16))
        return (keys,
                np.zeros((32, self.batch, 8, 512, 128), np.uint16))

    def __init__(self, *args, bulk_admit=False, token_major_host=False, skip_clear=False, direct_admit=False, **kwargs):
        if direct_admit and not bulk_admit:
            raise ValueError("direct admission requires bulk admission")
        self.direct_admit = direct_admit
        self.bulk_admit = bulk_admit
        self.token_major_host = token_major_host
        self.skip_clear = skip_clear
        if importlib.metadata.version('rebel-compiler') != '0.11.0':
            raise RuntimeError('DLFloat16 cache adapter validated only with Rebel 0.11.0')
        super().__init__(*args, **kwargs)
        from rebel.kv_cache import aligned_tensor
        self.block_elements = 8*512*128
        self.layer_bytes = self.runtime.get_kv_cache_size_for_layer('past_key_values_0')
        expected = self.batch*self.block_elements*2
        if self.layer_bytes != expected or self.runtime.get_kv_cache_size() != 64*expected:
            raise RuntimeError('unexpected native cache shape or precision')
        # Raw KV API requires aligned host storage; ordinary np.empty is rejected.
        self.staging = aligned_tensor(expected//2, dtype=np.uint16,
                                      alignment=2*1024*1024, tensor_type='np')
        self.staging.fill(0)
        if bulk_admit:
            # SDK 0.11.0 packs all-layer transfers in lexicographic name order.
            # Validated by per-layer writes followed by bulk read/write roundtrips.
            self.cache_order = sorted(range(64), key=lambda i: f'past_key_values_{i}')
            if not self.direct_admit:
                self.bulk_staging = aligned_tensor(64*expected//2, dtype=np.uint16,
                                                  alignment=2*1024*1024, tensor_type="np")
                self.bulk_staging.fill(0)
        self.blocks = np.arange(self.batch, dtype=np.int16).reshape(self.batch, 1)
        self.binary_mask = np.tril(np.ones((512, 512), np.float32))

    def clear(self, lane, buffers=None):
        # KVReceiver requires complete, nonduplicated prefix coverage before
        # admission. admit() overwrites the prefix and zeroes the device tail.
        # Clearing the entire host slot first is therefore redundant here.
        if not self.skip_clear:
            if self.direct_admit:
                for view in (buffers if buffers is not None else (self.keys, self.values)):
                    view.clear_lane(lane)
            else:
                super().clear(lane, buffers)

    def admit(self, jobs, buffers):
        keys, values = buffers
        lanes = [j.get("_lane", i) for i,j in enumerate(jobs)]
        if len(set(lanes)) != len(lanes) or any(type(i) is not int or not 0 <= i < self.batch for i in lanes):
            raise ValueError("invalid or duplicate admission slots")
        if not self.direct_admit and any(i != lane for i,lane in enumerate(lanes)):
            raise ValueError("sparse admission requires direct-admit")
        if self.direct_admit:
            if keys.storage is not values.storage:
                raise ValueError("direct KV views must share storage")
            for index, job in enumerate(jobs):
                lane = job.get("_lane", index)
                n = len(job["tokens"])-1
                keys.packed[:, lane, :, n:, :] = 0
                self.runtime.set_kv_cache(lane, 0, 512, keys.storage)
            return
        if self.bulk_admit:
            packed = self.bulk_staging.reshape(64, self.batch, 8, 512, 128)
            for index, job in enumerate(jobs):
                lane = job.get("_lane", index)
                n = len(job['tokens'])-1
                for slot, index in enumerate(self.cache_order):
                    layer, kind = divmod(index, 2)
                    source = (keys[layer, lane, :, :, :n].transpose(0, 2, 1)
                              if kind == 0 else values[layer, lane, :, :n, :])
                    packed[slot, lane, :, :n] = source
                    packed[slot, lane, :, n:] = 0
                self.runtime.set_kv_cache(lane, 0, 512, self.bulk_staging)
            return
        for lane, job in enumerate(jobs):
            # SDK host offsets mirror device block indices; lane 1 must not
            # read the host region for lane 0 (validated on a two-block graph).
            block = self.staging.reshape(self.batch, 8, 512, 128)[lane]
            n = len(job['tokens'])-1
            for layer in range(32):
                for kind in (0, 1):
                    block.fill(0)
                    source = keys[layer, lane, :, :, :n].transpose(0, 2, 1) if kind == 0 else values[layer, lane, :, :n, :]
                    block[:, :n, :] = source if source.dtype == np.uint16 else bf16_expanded_to_dlfloat(source)
                    self.runtime.set_kv_cache_for_layer(f'past_key_values_{2*layer+kind}',
                                                        lane, 0, 512, self.staging)

    def decode(self, jobs, diagnostics=False, buffers=None):
        if not jobs or len(jobs) > self.batch:
            raise ValueError('invalid batch occupancy')
        buffers = buffers if buffers is not None else (self.keys, self.values)
        start = time.perf_counter()
        self.admit(jobs, buffers)
        admit_seconds = time.perf_counter()-start
        current = np.zeros(self.batch, np.int64)
        positions = np.zeros(self.batch, np.int32)
        output, times, top = ([[] for _ in jobs] for _ in range(3))
        steps = []
        for lane, job in enumerate(jobs):
            current[lane] = job['tokens'][-1]
            positions[lane] = len(job['tokens'])-1
        for step in range(max(j['max_new_tokens'] for j in jobs)):
            active = [i for i, j in enumerate(jobs) if step < j['max_new_tokens']]
            t = time.perf_counter()
            logits = self.runtime.run(self.embed[current], self.cos[positions], self.sin[positions],
                                      self.binary_mask[positions], positions.reshape(self.batch, 1), self.blocks)
            if diagnostics and not np.isfinite(logits[active]).all():
                raise FloatingPointError(f'non-finite active logits at decode step {step}')
            nxt = np.argmax(logits, axis=-1)
            for i in active:
                output[i].append(int(nxt[i]))
                times[i].append(time.perf_counter()-start)
                if diagnostics:
                    indices = np.argpartition(logits[i], -5)[-5:]
                    indices = indices[np.argsort(logits[i][indices])[::-1]]
                    top[i].append([dict(token_id=int(x), logit=float(logits[i][x])) for x in indices])
                current[i] = nxt[i];positions[i] += 1
            for i in range(self.batch):
                if i >= len(jobs) or step+1 >= jobs[i]['max_new_tokens']:
                    positions[i] = 0;current[i] = 0
            steps.append(dict(seconds=time.perf_counter()-t, useful_lanes=len(active)))
        return [dict(id=j['id'], token_ids=output[i], token_ready_seconds=times[i],
                     admit_seconds=admit_seconds, device_resident_kv=True,
                     **({'top5': top[i]} if diagnostics else {})) for i, j in enumerate(jobs)], steps
