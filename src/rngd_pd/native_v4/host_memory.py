"""Owned 2 MiB-aligned buffers using the installed Rebel allocator.

Uses a version-dependent SDK extension, not device memory or cross-vendor P2P.
The owner must outlive every NumPy view and every synchronous runtime call.
"""
import ctypes
import math
import numpy as np


class HugeHostBuffer:
    def __init__(self, shape):
        from rebel import _C
        self.nbytes = math.prod(shape)*4
        self.ptr = 0
        self._free = _C.free_huge_host_memory
        self.ptr = _C.allocate_huge_host_memory(self.nbytes)
        if not self.ptr:
            raise MemoryError('Rebel huge-host allocation failed')
        if self.ptr % (2*1024*1024):
            self.close()
            raise RuntimeError('Rebel allocator returned an unaligned address')
        self.storage = (ctypes.c_ubyte*self.nbytes).from_address(self.ptr)
        self.array = np.ndarray(shape, dtype=np.float32, buffer=self.storage)
        self.array.fill(0)

    def close(self):
        if getattr(self, 'ptr', 0):
            self._free(self.ptr, self.nbytes)
            self.ptr = 0

    def __del__(self):
        self.close()
