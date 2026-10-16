"""ATOM decode adapter with persistent host buffers (not resident device KV)."""
import time
import numpy as np


class AtomDecoder:
    def __init__(self, artifact, tables, device=0, batch=4, huge_host=False):
        import rebel
        self.batch = batch
        self.huge_host = huge_host
        self.host_owners = []
        self.runtime = rebel.RBLNCompiledModel(str(artifact)).create_runtime(device=device)
        with np.load(tables) as z:
            self.embed = z['embed'].astype(np.float32)
            inv = z['inv']
        angles = np.arange(512, dtype=np.float32)[:, None] * inv[None, :]
        self.cos, self.sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        self.mask = np.triu(np.full((512, 512), -1e4, np.float32), 1)
        self.eye = np.eye(512, dtype=np.float32)
        self.keys, self.values = self.new_buffers()

    def new_buffers(self):
        shapes = [(32, self.batch, 8, 128, 512), (32, self.batch, 8, 512, 128)]
        if not self.huge_host:
            return tuple(np.zeros(shape, np.float32) for shape in shapes)
        from .host_memory import HugeHostBuffer
        owners = [HugeHostBuffer(shape) for shape in shapes]
        self.host_owners.extend(owners)
        return tuple(owner.array for owner in owners)

    def close(self):
        # Runner calls this only after all decode futures have completed.
        self.runtime = None
        self.keys = self.values = None
        for owner in self.host_owners:
            owner.close()
        self.host_owners.clear()

    def clear(self, lane, buffers=None):
        keys, values = buffers if buffers is not None else (self.keys, self.values)
        keys[:, lane].fill(0)
        values[:, lane].fill(0)

    def decode(self, jobs, diagnostics=False, buffers=None):
        """All jobs already have transferred P-1 KV in their respective slots.

        Unused compiled lanes execute padding and are excluded from useful work.
        """
        if not jobs or len(jobs) > self.batch:
            raise ValueError('invalid batch occupancy')
        keys, values = buffers if buffers is not None else (self.keys, self.values)
        current = np.zeros(self.batch, dtype=np.int64)
        positions = np.zeros(self.batch, dtype=np.int64)
        output = [[] for _ in jobs]
        times = [[] for _ in jobs]
        top = [[] for _ in jobs]
        steps = []
        for i, job in enumerate(jobs):
            current[i] = job['tokens'][-1]
            positions[i] = len(job['tokens'])-1
        start = time.perf_counter()
        for step in range(max(j['max_new_tokens'] for j in jobs)):
            active = [i for i, j in enumerate(jobs) if step < j['max_new_tokens']]
            t = time.perf_counter()
            logits, key, value = self.runtime.run(self.embed[current], keys, values,
                self.cos[positions], self.sin[positions], self.mask[positions], self.eye[positions])
            nxt = np.argmax(logits, axis=-1)
            for i in active:
                keys[:, i, :, :, positions[i]] = key[:, i]
                values[:, i, :, positions[i], :] = value[:, i]
                output[i].append(int(nxt[i]))
                if diagnostics:
                    best = np.argpartition(logits[i], -5)[-5:]
                    best = best[np.argsort(logits[i][best])[::-1]]
                    top[i].append([dict(token_id=int(x), logit=float(logits[i][x])) for x in best])
                times[i].append(time.perf_counter()-start)
                current[i] = nxt[i]
                positions[i] += 1
            # Completed lanes must remain in bounds even for a full-context request.
            for i in range(self.batch):
                if i >= len(jobs) or step+1 >= jobs[i]['max_new_tokens']:
                    positions[i] = 0; current[i] = 0
            steps.append(dict(seconds=time.perf_counter()-t, useful_lanes=len(active)))
        return [dict(id=j['id'], token_ids=output[i], token_ready_seconds=times[i],
                     **({'top5': top[i]} if diagnostics else {}))
                for i, j in enumerate(jobs)], steps
