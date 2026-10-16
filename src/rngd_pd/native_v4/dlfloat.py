"""BF16 KV -> Rebel 0.11 DLFloat16 cache representation.

Format/range follow installed tvm/target/datatype.py: sign/exp/mantissa=1/6/9,
float32 exponent range 96..159. BF16's seven mantissa bits need no rounding.
This converts a numerical format; it does not reinterpret BF16 as FP16.
"""
import numpy as np


def bf16_expanded_to_dlfloat(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    if np.any(bits & 0xffff):
        raise ValueError('admission expects exactly BF16-expanded FP32 KV')
    return bf16_bits_to_dlfloat(bits >> 16)


def _make_lookup():
    bf = np.arange(65536, dtype=np.uint32)
    exp = ((bf >> 7) & 255).astype(np.int32)
    sign = bf & 0x8000
    normal = sign | (((exp-96) << 9).astype(np.uint32)) | ((bf & 127) << 2)
    output = np.where(exp < 96, sign, np.where(exp > 159, sign | 0x7fff, normal))
    return output.astype(np.uint16)


_LOOKUP = _make_lookup()
_LOOKUP.flags.writeable = False


def bf16_bits_to_dlfloat(bf):
    # A 128 KiB table covers every BF16 bit pattern. Avoid repeatedly creating
    # several large FP32/exponent/mantissa temporaries for each KV tensor.
    bf = np.asarray(bf, dtype=np.uint16)
    if np.any((bf & 0x7f80) == 0x7f80):
        raise ValueError('non-finite KV cannot be admitted')
    return _LOOKUP[bf]
