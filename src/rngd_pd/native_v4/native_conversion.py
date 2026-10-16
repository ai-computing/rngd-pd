"""Optional CPU SIMD converter; same finite BF16 mapping as dlfloat.py."""
import ctypes
import numpy as np

class NativeConverter:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path.resolve()))
        self.function = self.library.bf16_to_dlfloat
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.function.restype = ctypes.c_int

    def __call__(self, raw):
        raw = np.ascontiguousarray(raw, dtype=np.uint16)
        result = np.empty_like(raw)
        if self.function(raw.ctypes.data, result.ctypes.data, raw.size):
            raise ValueError('non-finite KV cannot be admitted')
        return result
