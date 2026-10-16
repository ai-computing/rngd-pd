"""Device-side BF16 KV encoding for the experimental RNGD→ATOM transport.

The FP16 output is a *bit container* for DLFloat16, not an FP16 KV tensor.
Valid only for finite inputs with abs(x) < 2**31. The caller must check the
returned maximum before submitting DMA. Underflowed signed zeros may become
positive zero; comparison against the host encoder canonicalizes both zeros.
"""
import torch


class DLFloatBitPayload(torch.nn.Module):
    def forward(self, x):
        x = x.to(torch.float32)
        a = torch.abs(x)
        scale = torch.full_like(a, 2.**14)
        bias = torch.full_like(a, 2.**-15)
        # For 4**n <= abs(x) < 4**(n+1), the FP16 number whose bits encode
        # DLFloat16(x) is abs(x)*2**(-n-1) + 2**n. These operations are exact
        # for BF16 inputs; no integer bit operators or floor are required.
        for n in range(-14, 16):
            select = a >= 4.**n
            scale = torch.where(select, torch.full_like(a, 2.**(-n-1)), scale)
            bias = torch.where(select, torch.full_like(a, 2.**n), bias)
        y = a * scale + bias
        # The FP16 carrier's subnormal range needs its own affine mapping.
        y = torch.where(a < 2.**-29, a * 2.**15, y)
        y = torch.where(a < 2.**-30, a * 2.**16 - 2.**-15, y)
        y = torch.where(a < 2.**-31, torch.zeros_like(y), y)
        y = torch.where(x < 0, -y, y)
        y = torch.where(a == 0, x, y)
        return y.to(torch.float16)


class PackKVPayload(torch.nn.Module):
    """K [layers,heads,dim,tokens], V [layers,heads,tokens,dim] → K0,V0,…"""
    def __init__(self):
        super().__init__()
        self.encode = DLFloatBitPayload()

    def forward(self, keys, values):
        kv = torch.stack((keys.transpose(-2, -1), values), dim=1).flatten(0, 1)
        return self.encode(kv), torch.amax(torch.abs(kv.to(torch.float32)))


class PackPrefixKVPayload(PackKVPayload):
    """Pack only a page-aligned prefix of the validated 512-token KV layout.

    Each head uses 256 bytes/token. Rounding to 16 tokens keeps every DMA
    segment 4 KiB aligned. The consumer must mask/overwrite the padded tail;
    this is for an exclusively owned B1 cache, not concurrent cache slots.
    """
    def __init__(self, valid_tokens):
        super().__init__()
        if not isinstance(valid_tokens, int) or not 1 <= valid_tokens <= 512:
            raise ValueError('valid_tokens must be in [1, 512]')
        self.transfer_tokens = ((valid_tokens + 15) // 16) * 16

    def forward(self, keys, values):
        return super().forward(keys[..., :self.transfer_tokens],
                               values[..., :self.transfer_tokens, :])


class BoundedPrefixPacker:
    """Encode a long prefix using one fixed 16-token device graph.

    Avoids compiling the elementwise DLFloat encoder over a large full prefix.
    KV slices, encoded chunks and concatenation stay on the selected RNGD.
    Only the scalar range checks are read on the host. This does not submit DMA.
    """
    def __init__(self, device='furiosa:1', layers=32, heads=8, head_dim=128):
        import furiosa.torch as ft
        if any(type(n) is not int or n <= 0 for n in (layers, heads, head_dim)):
            raise ValueError('positive integer model dimensions required')
        self.device, self.layers, self.heads, self.head_dim = device, layers, heads, head_dim
        sample = (torch.zeros((layers, heads, head_dim, 16), dtype=torch.bfloat16),
                  torch.zeros((layers, heads, 16, head_dim), dtype=torch.bfloat16))
        self.module = ft.CompileModule.from_module(PackKVPayload(), sample)
        self.module.to(device)

    def encode(self, keys, values, valid_tokens):
        import math
        if type(valid_tokens) is not int or not 1 <= valid_tokens <= 512:
            raise ValueError('valid_tokens must be in [1, 512]')
        if (tuple(keys.shape) != (self.layers, self.heads, self.head_dim, 512) or
                tuple(values.shape) != (self.layers, self.heads, 512, self.head_dim)):
            raise ValueError('expected the 512-token KONI cache layout')
        if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            raise ValueError('bounded encoder requires BF16 KV tensors')
        if str(keys.device) != self.device or str(values.device) != self.device:
            raise ValueError('KV must already reside on the selected RNGD')
        padded = ((valid_tokens + 15) // 16) * 16
        chunks, maximum = [], 0.0
        for offset in range(0, padded, 16):
            # The compiled input has dense strides. A narrow view still carries
            # the original 512-token strides; materialize it on-device first.
            key_window = keys[..., offset:offset+16].contiguous()
            value_window = values[..., offset:offset+16, :].contiguous()
            encoded, bound = self.module(key_window, value_window)
            scalar = float(bound.cpu().item())
            if not math.isfinite(scalar) or scalar >= 2**31:
                raise ValueError('KV outside the validated DLFloat encoding range')
            maximum = max(maximum, scalar)
            chunks.append(encoded)
        return torch.cat(chunks, dim=2), maximum
