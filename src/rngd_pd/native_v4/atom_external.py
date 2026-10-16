"""Experimental KONI decoder with caller-owned device KV tensors.

Requires a graph-mode artifact, not the existing static-cache decode.rbln.
Uses the installed rebel 0.11 runtime ABI. Transport and cache admission are
deliberately separate: creating this decoder does not enable P2P transfer.
"""
from pathlib import Path

import numpy as np


class ExternalCacheDecoder:
    def __init__(self, artifact, tables, device=0):
        import torch
        import torch_rbln  # noqa: F401: registers the device backend
        import rebel
        from rebel.sync_runtime import DynamoRuntime

        if rebel.__version__ != '0.11.0':
            raise RuntimeError('External-cache adapter validated only with rebel 0.11.0')
        self.torch = torch
        self.rebel = rebel
        self.device = f'rbln:{device}'
        # Register torch's logical-device mapping before creating the SDK runtime.
        self._device_owner = torch.zeros(1).to(self.device)
        self.compiled = rebel.RBLNCompiledModel(str(Path(artifact)))
        self.runtime = self.compiled.create_runtime(device=device, tensor_type='pt')
        profiles = self.runtime._input_profile
        self.batch_size = int(profiles[0].shape[0])
        if self.batch_size not in (1, 2, 4, 8):
            raise ValueError('Unsupported compiled batch size')
        batch = self.batch_size
        cache_shape = (batch, 8, 512, 128)
        expected = [((batch, 4096), torch.float32), ((batch, 64), torch.float32),
                    ((batch, 64), torch.float32), ((batch, 512), torch.float32),
                    (cache_shape, torch.float32), (cache_shape, torch.float32),
                    ((batch, 1), torch.int32), ((batch, 1), torch.int16)]
        expected += [(cache_shape, torch.float32)] * 62
        actual = [(tuple(x.shape), x.dtype) for x in profiles]
        if actual != expected:
            raise ValueError('Artifact does not match the KONI external-KV input order')
        self.runner = DynamoRuntime(compiled_model=self.runtime._compiled_model,
                                    executor=self.runtime._executor,
                                    runtime_handle=self.runtime._runtime_handle)
        self.runner.cpu_threads = 8
        self.caches = [torch.zeros(cache_shape).to(self.device) for _ in range(64)]
        with np.load(tables) as data:
            self.embed = data['embed'].astype(np.float32)
            self.inv = data['inv'].copy()

    def cache_device_addresses(self):
        """Live addresses only; these do not establish exportability or lifetime."""
        return [self.rebel._C.vmem.get_device_addrs(t.data_ptr()) for t in self.caches]

    def notify_cache_write(self, indices):
        """Call only after the owning transport has confirmed DMA completion."""
        indices = list(indices)
        if any(not isinstance(i, int) or i < 0 or i >= 64 for i in indices):
            raise ValueError('KV tensor index must be in [0, 64)')
        for i in indices:
            self.rebel._C.vmem.mark_device_memory_is_updated(self.caches[i].data_ptr())

    def step(self, token_id, position):
        if self.batch_size != 1:
            raise ValueError('Use step_batch for a batched artifact')
        return self.step_batch([token_id], [position])

    def step_batch(self, token_ids, positions):
        """One independent sequence per cache block; no cross-card model split."""
        token_ids, positions = np.asarray(token_ids), np.asarray(positions)
        if (token_ids.shape != (self.batch_size,) or
                positions.shape != (self.batch_size,) or
                token_ids.dtype.kind not in 'iu' or positions.dtype.kind not in 'iu'):
            raise ValueError('Expected one integer token and position per batch slot')
        if (np.any(token_ids < 0) or np.any(token_ids >= len(self.embed)) or
                np.any(positions < 0) or np.any(positions >= 512)):
            raise ValueError('token or cache position out of range')
        torch = self.torch
        angle = positions.astype(np.float32)[:, None] * self.inv[None, :]
        mask = (np.arange(512)[None, :] <= positions[:, None]).astype(np.float32)
        host = [self.embed[token_ids],
                np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32), mask,
                positions.astype(np.int32)[:, None],
                np.arange(self.batch_size, dtype=np.int16)[:, None]]
        with torch.no_grad():
            inputs = [torch.from_numpy(x).to(self.device) for x in host]
            # Graph-mode tracing places first-layer K/V before position/block inputs.
            args = inputs[:4] + self.caches[:2] + inputs[4:] + self.caches[2:]
            outputs = self.runner(*args, out=None)
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                raise RuntimeError('Unexpected decoder output structure')
            return outputs[0].cpu().float().numpy()
