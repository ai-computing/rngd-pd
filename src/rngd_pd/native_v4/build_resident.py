"""Compile KONI decode with SDK-native paged attention and resident KV.

Keeps the validated BF16 projection/MLP arithmetic; replaces host KV graph
inputs with SDK static cache tensors. No hardware is opened while building.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
import numpy as np
import torch
import rebel
from rebel.compile_context import CompileContext
from safetensors import safe_open
from rngd_pd.vendors.rbln_decode_batch import LlamaDecodeBatchRBLN


def register_attention():
    path = Path(rebel.__file__).parent.parent/'optimum/rbln/ops/attn.py'
    spec = importlib.util.spec_from_file_location('native_pd_paged_ops', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path


class ResidentDecode(LlamaDecodeBatchRBLN):
    full_output_projection = False

    def forward(self, x, cos, sin, mask, seq, blocks, *cache):
        B, NH, NKV, HD, G = self.B, self.NH, self.NKV, self.HD, self.G
        for i in range(self.nl):
            h = self._rms(x, getattr(self, f'n1_{i}'))
            q = self._rope(self._mm(h, getattr(self, f'wq_{i}')), NH, cos, sin)
            k = self._rope(self._mm(h, getattr(self, f'wk_{i}')), NKV, cos, sin)
            v = self._mm(h, getattr(self, f'wv_{i}')).view(B, NKV, HD)
            ao = torch.ops.rbln_custom_ops.paged_attn_decode(
                q.view(B, NKV, G, 1, HD), k.view(B, NKV, 1, 1, HD),
                v.view(B, NKV, 1, 1, HD), mask.view(B, 1, 1, 1, self.S),
                cache[2*i].unsqueeze(2), cache[2*i+1].unsqueeze(2),
                seq, torch.tensor(self.scale), blocks, self.S)
            if self.full_output_projection:
                # wo3 stores the original o_proj transpose, split into heads.
                # One contraction avoids eight separately rounded partials.
                weight = getattr(self, f'wo3_{i}').reshape(self.H, self.H)
                x2 = x + torch.matmul(ao.reshape(B, self.H).to(self.wdt), weight).float()
            else:
                y = torch.matmul(ao.reshape(B, NKV, 1, G*HD).to(self.wdt),
                                 getattr(self, f'wo3_{i}').unsqueeze(0)).float()
                x2 = x
                for j in range(NKV):
                    x2 = x2+y[:, j, 0]
            h2 = self._rms(x2, getattr(self, f'n2_{i}'))
            gate = self._mm(h2, getattr(self, f'wg_{i}'))
            up = self._mm(h2, getattr(self, f'wu_{i}'))
            x = self._mm(torch.nn.functional.silu(gate)*up, getattr(self, f'wd_{i}'))+x2
        return self._mm(self._rms(x, self.normw), self.lmw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--layers', type=int, default=32)
    p.add_argument('--full-output-projection', action='store_true')
    p.add_argument('--weight-dtype', choices=['bf16', 'fp16'], default='bf16',
                   help='Projection/MLP precision; BF16 remains the validated default')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    try:
        torch.set_num_threads(8)
        op_path = register_attention()
        config = json.loads((a.model/'config.json').read_text())
        cfg = SimpleNamespace(hidden=config['hidden_size'], inter=config['intermediate_size'],
            n_heads=config['num_attention_heads'], n_kv=config['num_key_value_heads'],
            head_dim=config['hidden_size']//config['num_attention_heads'],
            gqa=config['num_attention_heads']//config['num_key_value_heads'],
            eps=config['rms_norm_eps'])
        sd = {}
        for shard in sorted(a.model.glob('model-*.safetensors')):
            with safe_open(shard, framework='pt') as f:
                for key in f.keys():
                    if key.startswith('model.layers.') and int(key.split('.')[2]) >= a.layers:
                        continue
                    if key == 'model.embed_tokens.weight':
                        continue
                    sd[key] = f.get_tensor(key).float().numpy()
        weight_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[a.weight_dtype]
        model = ResidentDecode(cfg, sd, 512, a.layers, a.batch, wdt=weight_dtype).eval()
        model.full_output_projection = a.full_output_projection
        del sd
        cache = [torch.zeros(a.batch, cfg.n_kv, 512, cfg.head_dim) for _ in range(a.layers*2)]
        context = CompileContext(use_weight_sharing=True)
        for i, tensor in enumerate(cache):
            context.mark_static_address(tensor, f'past_key_values_{i}')
        inputs = [torch.zeros(a.batch, cfg.hidden), torch.zeros(a.batch, cfg.head_dim//2),
                  torch.zeros(a.batch, cfg.head_dim//2), torch.ones(a.batch, 512),
                  torch.zeros(a.batch, 1, dtype=torch.int32),
                  torch.arange(a.batch, dtype=torch.int16).reshape(a.batch, 1), *cache]
        metadata = dict(model=str(a.model), revision=a.model.name, batch=a.batch, context=512,
            layers=a.layers, weight_dtype=a.weight_dtype.upper(), host_inputs='FP32 except int32 positions/int16 blocks',
            resident_cache=True, kv_storage='SDK DLFloat16',
            full_output_projection=a.full_output_projection,
            compiler=rebel.__version__, operator_source=str(op_path),
            operator_sha256=hashlib.sha256(op_path.read_bytes()).hexdigest())
        (a.out/'manifest.json').write_text(json.dumps(metadata, indent=2))
        compiled = rebel.compile_from_torch(model, example_inputs=inputs,
                                             compile_context=context, npu='RBLN-CA22')
        compiled.save(str(a.out/'decode.rbln'))
        (a.out/'completed.json').write_text(json.dumps(dict(status='compiled', seconds=time.time()-start)))
        print('COMPILED', a.out, time.time()-start, flush=True)
    except Exception as exc:
        (a.out/'failed.json').write_text(json.dumps(dict(error=repr(exc), traceback=traceback.format_exc(),
                                                       seconds=time.time()-start), indent=2))
        raise


if __name__ == '__main__':
    main()
