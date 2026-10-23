"""Experimental ATOM constant refresh through the weight-free runtime ABI.

The caller must drain requests and retain the original CPU parameter objects.
Compiled transform groups, rather than arbitrary file offsets, define the
smallest update unit. No installed SDK files or native binaries are modified.
"""
import os
import time
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4

import numpy as np

from .atom_inputs import DeviceWeightDecode, _KV, _serialized, weight_names, SUFFIXES, PER_LAYER


class ConstantWeightUpdater:
    def __init__(self, parameters, runtimes):
        self.parameters, self.runtimes = dict(parameters), runtimes
        self.failed = False

    @staticmethod
    def _write(handle, keys, host_keys, groups, values):
        if keys:
            # The native API otherwise skips transform graphs already applied.
            handle.reset_applied_transform_graphs()
            applied = handle.apply_extern_transforms({k: values[k] for k in keys})
            if applied != groups:
                raise RuntimeError(f"constant refresh applied {applied}/{groups} transforms")
        if host_keys:
            handle.apply_host_params({k: values[k] for k in host_keys})

    def apply(self, replacements):
        import torch

        if self.failed:
            raise RuntimeError("constant buffers unavailable; reload required")
        params = self.parameters
        if not replacements.keys() <= params.keys():
            raise ValueError("unknown constant parameter")
        before = {p.data_ptr(): p for p in params.values()}
        if len(before) != len(params):
            raise ValueError("aliased constant parameters require an explicit mapping")
        after, backup = dict(before), {}
        for name, value in replacements.items():
            param = params[name]
            if (tuple(value.shape) != tuple(param.shape) or value.dtype != param.dtype
                    or value.device.type != 'cpu' or param.device.type != 'cpu'):
                raise ValueError(f"constant layout mismatch: {name}")
            key = param.data_ptr()
            backup[key] = param.detach().clone()
            after[key] = value.detach().contiguous()
        changed = set(backup)
        old_values = dict(before)
        old_values.update(backup)
        jobs, covered, seen = [], set(), set()
        for runtime in self.runtimes:
            handle = runtime._runtime_handle
            if id(handle) in seen:
                continue
            seen.add(id(handle))
            groups = [set(g) for g in handle.extern_transform_key_groups()
                      if changed.intersection(g)]
            keys = set().union(*groups) if groups else set()
            host = changed.intersection(handle.get_host_op_keys())
            if not keys <= before.keys():
                raise ValueError("unmapped constant transform dependency")
            covered.update(changed.intersection(keys | host))
            if keys or host:
                jobs.append((handle, keys, host, len(groups)))
        if covered != changed:
            raise ValueError("changed constant is not bound to a loaded runtime")
        attempted = []
        try:
            for job in jobs:
                attempted.append(job)
                self._write(*job, after)
            with torch.no_grad():
                for key in changed:
                    before[key].copy_(after[key])
        except Exception:
            try:
                for job in reversed(attempted):
                    self._write(*job, old_values)
                with torch.no_grad():
                    for key in changed:
                        before[key].copy_(backup[key])
            except Exception as rollback_error:
                self.failed = True
                raise RuntimeError("constant update and rollback failed; reload required") from rollback_error
            raise
        return dict(changed_weight_bytes=sum(t.numel() * t.element_size() for t in backup.values()),
                    transform_input_bytes=sum(after[k].numel() * after[k].element_size()
                                              for _, keys, _, _ in jobs for k in keys),
                    transform_groups=sum(n for _, _, _, n in jobs),
                    host_parameters=sum(len(keys) for _, _, keys, _ in jobs))


class ConstantWeightDecode(DeviceWeightDecode):
    """Fast constant graphs with partial refresh, on the validated 0.11 ABI.

    Start the worker with RBLN_WEIGHT_FREE=1. Prepare shapes before admission.
    Each device may have one owning instance per process: this SDK's eager
    compile context is global. Restart the worker to replace the owning model.
    KV is host-resident; resident weights are updated in compiled constant pools.
    """
    _claimed_devices = set()
    _claim_lock = Lock()

    def __init__(self, model_dir, cache_dir, device=0, layers=None):
        import torch
        import rebel
        from rebel.compile_context import CompileContext
        from .atom import Tables, load_checkpoint, decode_module
        from .rngd import Checkpoint

        if rebel.__version__ != '0.11.0':
            raise RuntimeError('constant refresh is validated only with rebel 0.11.0')
        if os.environ.get('RBLN_WEIGHT_FREE') not in ('1', 'on'):
            raise ValueError('start the worker with RBLN_WEIGHT_FREE=1 before importing rebel')
        if not isinstance(device, int) or isinstance(device, bool) or device < 0:
            raise ValueError('an explicitly allocated device index is required')
        with self._claim_lock:
            if device in self._claimed_devices:
                raise RuntimeError('one constant engine per device per process; start a fresh worker')
            self._claimed_devices.add(device)
        torch.set_grad_enabled(False)
        self.torch, self.rebel, self.device = torch, rebel, device
        self._lock, self._closed, self._generation = RLock(), False, 0
        self._kv_owner = object()
        self.ck = Checkpoint(model_dir)
        t0 = time.perf_counter()
        cfg, sd, nl = load_checkpoint(model_dir, layers)
        self.module = decode_module(cfg, sd, nl)
        del sd
        self.tables = Tables(model_dir, cfg)
        self.d = dict(nl=nl, H=cfg.hidden, NKV=cfg.n_kv, HD=cfg.head_dim, G=cfg.gqa)
        self.weights = {}
        for i in range(nl):
            for suffix, attr in zip(SUFFIXES, PER_LAYER):
                attr = 'wo3' if attr == 'wo' else attr
                self.weights[f'model.layers.{i}.{suffix}'] = getattr(self.module, f'{attr}_{i}')
        self.weights.update({'model.norm.weight': self.module.normw, 'lm_head.weight': self.module.lmw})
        self.W = [self.weights[n] for n in weight_names(nl)]
        self.context = CompileContext(use_weight_sharing=True, is_eager_mode=True,
                                      use_global_ctx=True, device_id=device)
        self.cache_dir = Path(cache_dir) / ('constants-' + uuid4().hex)
        self.cache_dir.mkdir(parents=True)
        self.graphs, self.compiled, self.runtimes = {}, {}, []
        self.updater = ConstantWeightUpdater(self.weights, self.runtimes)
        self.active = None
        self.timings = dict(host_weights_s=time.perf_counter() - t0)

    @_serialized
    def prepare(self, variant):
        if variant in self.graphs:
            return
        from rebel.core.weight_free import apply_weight_free_weights
        from .atom import input_info

        cm = self.rebel.compile_from_torch(
            self.module, input_info=input_info(self.tables.cfg, self.d['nl'], variant.B, variant.S),
            compile_context=self.context, npu='RBLN-CA22', _intmd_mode='gen')
        rt = cm.create_runtime(device=self.device, tensor_type='np')
        handle = rt._runtime_handle
        if not self.runtimes and not handle.extern_transform_key_groups():
            raise RuntimeError('compiler emitted no weight-free transform metadata')
        apply_weight_free_weights(rt, self.module)
        cm.save(self.cache_dir / (variant.name + '.rbln'))
        self.compiled[variant], self.graphs[variant] = cm, rt
        self.runtimes.append(rt)

    @_serialized
    def swap(self, variant):
        started = time.perf_counter()
        prepared = variant not in self.graphs
        self.prepare(variant)
        if self.active != variant:
            self._generation += 1
        self.active = variant
        return dict(variant=variant.name, prepared=prepared, swap_s=time.perf_counter() - started)

    @_serialized
    def blank_kv(self):
        v, d = self.active, self.d
        if v is None:
            raise ValueError('select a variant first')
        return tuple(_KV(a, self._generation, v, self._kv_owner) for a in (
            np.zeros((d['nl'], v.B, d['NKV'], d['HD'], v.S), np.float32),
            np.zeros((d['nl'], v.B, d['NKV'], v.S, d['HD']), np.float32)))

    @_serialized
    def step(self, toks, pos, K, V):
        v = self.active
        if any(not isinstance(t, _KV) or t.owner is not self._kv_owner
               or t.generation != self._generation or t.variant != v for t in (K, V)):
            raise ValueError('weights or variant changed; start a fresh request')
        if (v is None or len(toks) != v.B or len(pos) != v.B
                or any(p < 0 or p >= v.S for p in pos)):
            raise ValueError('tokens/positions do not fit the active variant')
        x, cos, sin, mask, oh = self.tables.step_inputs(toks, pos, v.S)
        logits, k, vv = self.graphs[v].run(x, K.tensor, V.tensor, cos, sin, mask, oh)[:3]
        if not np.isfinite(logits).all():
            raise RuntimeError('non-finite NPU logits')
        for b, p in enumerate(pos):
            K.tensor[:, b, :, :, p] = k[:, b]
            V.tensor[:, b, :, p, :] = vv[:, b]
        return [int(t) for t in np.argmax(logits, axis=-1)], K, V

    @_serialized
    def apply_delta(self, new_checkpoint, plan):
        from .rngd import Checkpoint
        from .delta import CHECKPOINT_GRAPH_KEYS

        started = time.perf_counter()
        new = Checkpoint(new_checkpoint)
        try:
            old = self.ck
            if (plan.base_dir != os.path.realpath(old.path)
                    or plan.new_dir != os.path.realpath(new.path)):
                raise ValueError('delta plan does not describe the resident and target checkpoints')
            if any(old.config.get(k) != new.config.get(k) for k in CHECKPOINT_GRAPH_KEYS):
                raise ValueError('checkpoint graph/rotary configuration changed; reload required')
            if plan.removed or set(old.where) != set(new.where):
                raise ValueError('checkpoint tensor set changed; reload required')
            staged, embed = {}, self.tables.embed
            for name in plan.tensors:
                a, b = old.get(name), new.get(name)
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f'tensor layout changed; reload required: {name}')
                if name == 'model.embed_tokens.weight':
                    embed = b.float().numpy()
                elif name in self.weights:
                    if name.endswith('self_attn.o_proj.weight'):
                        b = b.t().reshape(self.weights[name].shape)
                    staged[name] = b.to(self.weights[name].dtype).contiguous()
                else:
                    raise ValueError(f'unsupported changed tensor: {name}')
            try:
                result = self.updater.apply(staged)
            except Exception:
                if self.updater.failed:
                    self._closed = True
                raise
            self.tables.embed, self.ck = embed, new
            self._generation += 1
            old.close()
            return dict(result, delta_s=time.perf_counter() - started,
                        uploaded_bytes=result['transform_input_bytes'],
                        mode='constant-transforms', tensors=len(plan.tensors))
        finally:
            if self.ck is not new:
                new.close()

    def close(self):
        with self._lock:
            self.graphs.clear()
            self.compiled.clear()
            self.runtimes.clear()
            self.updater.parameters.clear()
            self.weights.clear()
            self.W.clear()
            self.module = self.context = None
            self.ck.close()
            self.active, self._closed = None, True
