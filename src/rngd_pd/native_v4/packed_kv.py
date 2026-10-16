"""Views into SDK lexicographically packed host KV; no vendor/device P2P."""
import numpy as np

class PackedKVView:
    """Scalar-layer indexing returns writable views, including token reordering.

    Arbitrary multi-layer advanced indexing is intentionally unsupported: NumPy
    would silently copy, breaking writes to the DMA source allocation.
    """
    def __init__(self, storage, batch, kind):
        self.storage = storage
        self.kind = kind
        self.packed = storage.reshape(64, batch, 8, 512, 128)
        order = sorted(range(64), key=lambda i: f'past_key_values_{i}')
        self.slots = {index:slot for slot,index in enumerate(order)}
        self.shape = (32,batch,8,128,512) if kind == 0 else (32,batch,8,512,128)
        self.dtype = storage.dtype
        self.nbytes = storage.nbytes // 2
        self.ctypes = storage.ctypes

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        layer = index[0]
        if not isinstance(layer, (int, np.integer)) or not 0 <= layer < 32:
            raise IndexError('packed KV requires scalar layer 0..31')
        view = self.packed[self.slots[2*int(layer)+self.kind]]
        if self.kind == 0:
            view = view.transpose(0,1,3,2)
        return view[index[1:]]

    def __setitem__(self, index, value):
        self[index][...] = value

    def clear_lane(self, lane):
        for layer in range(32):
            self[layer, lane].fill(0)
