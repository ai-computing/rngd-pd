"""Validated, bounded host staging for the 2026.4 KONI connector.

This is host-staged DMA, not device-to-device P2P. Wire KV is interleaved
K0,V0,...,K31,V31, BF16 [token, head, channel].
"""
import numpy as np
from .wire import ProtocolError
from .dlfloat import bf16_bits_to_dlfloat


def host_layout(capacity=512):
    return {
        'attention_type': 'Global', 'unit_block_size': 2048, 'block_size': 2048,
        'num_chips': 1, 'element_type': 'Bfloat16',
        'dram_shape': {'inter_chip_axes': {'inner': []}, 'intra_chip_axes': {
            'axes': [{'tag': {'LabelStride': {'label': {'inner': str(i)}, 'stride': 1}},
                      'size': size, 'stride': stride}
                     for i, (size, stride) in enumerate(
                         [(capacity, 1024), (1, 1024), (8, 128), (128, 1)])],
            'offset': 0}}}


class KVReceiver:
    """Write each received block straight into a reusable ATOM host input slot.

    A single-tensor scratch avoids concatenating payloads or building a second
    full KV cache. Never acknowledge partial, duplicated, or out-of-range KV.
    """
    def __init__(self, request_id, prefix_tokens, keys, values, lane=0, converter=None):
        if not 1 <= prefix_tokens <= 511:
            raise ValueError('supported transferred prefix is 1..511 tokens')
        if keys.shape != (32, keys.shape[1], 8, 128, 512):
            raise ValueError('invalid key buffer')
        if values.shape != (32, keys.shape[1], 8, 512, 128):
            raise ValueError('invalid value buffer')
        if keys.dtype != values.dtype or keys.dtype not in (np.float32, np.uint16):
            raise ValueError('ATOM host KV must be FP32 or native DLFloat16 uint16')
        self.native_dl = keys.dtype == np.uint16
        self.converter = converter or bf16_bits_to_dlfloat
        if not 0 <= lane < keys.shape[1]:
            raise ValueError('invalid lane')
        self.request_id, self.n = request_id, prefix_tokens
        self.keys, self.values, self.lane = keys, values, lane
        self.seen = np.zeros((64, prefix_tokens), dtype=bool)
        self.scratch = None if self.native_dl else np.empty((prefix_tokens, 8, 128), dtype=np.uint32)
        self.complete = False
        self.payload_bytes = 0
        self.frame_count = 0
        self.entry_count = 0

    def accept(self, message, frames):
        if self.complete or message.get('kind') != 'kv' or message.get('request_id') != self.request_id:
            raise ProtocolError('unexpected KV message or completed request')
        entries = message['entries']
        arrays = isinstance(entries, np.ndarray)
        total = int(entries['bytes'].sum()) if arrays else sum(e['bytes'] for e in entries)
        if not len(entries) or total > 64 * self.n * 2048 or sum(len(f) for f in frames) != total:
            raise ProtocolError('invalid payload length')
        unit_blocks = bool(np.all(entries['bytes'] == 2048)) if arrays else all(e['bytes'] == 2048 for e in entries)
        if len(frames) == 1 and unit_blocks:
            return self._accept_unit_frame(message, frames[0])
        if arrays:
            entries = [{name: int(e[name]) for name in entries.dtype.names} for e in entries]
        # Validate the entire batch before writing any host inputs.
        covered = set()
        for e in entries:
            t, b, size = e['tensor'], e['block'], e['bytes']
            if e['tier'] != 0 or not 0 <= t < 64 or size <= 0 or size % 2048:
                raise ProtocolError('unsupported KV tensor, tier, or block size')
            n = size // 2048
            if b < 0 or b + n > self.n:
                raise ProtocolError('KV block outside admitted prefix')
            for j in range(b, b+n):
                if self.seen[t, j] or (t, j) in covered:
                    raise ProtocolError('duplicate KV block')
                covered.add((t, j))
        if message['is_final'] and int(self.seen.sum()) + len(covered) != 64 * self.n:
            raise ProtocolError('final batch has incomplete KV coverage')
        # Frames may split a block. Copy only such split blocks, never the full payload.
        views = [memoryview(f) for f in frames if len(f)]
        fi = off = 0
        runs = []
        for e in entries:
            if (runs and runs[-1]['tensor'] == e['tensor'] and
                    runs[-1]['block'] + runs[-1]['bytes']//2048 == e['block']):
                runs[-1]['bytes'] += e['bytes']
            else:
                runs.append(dict(e))
        for e in runs:
            b, end = e['block'], e['block'] + e['bytes']//2048
            while b < end:
                available = (len(views[fi])-off)//2048
                if available:
                    count_blocks = min(end-b, available)
                    count_bytes = count_blocks*2048
                    block = views[fi][off:off+count_bytes]; off += count_bytes
                else:
                    count_blocks = 1
                    block = bytearray(2048); dest = 0
                    while dest < 2048:
                        count = min(2048-dest, len(views[fi])-off)
                        block[dest:dest+count] = views[fi][off:off+count]
                        dest += count; off += count
                        if off == len(views[fi]) and dest < 2048:
                            fi += 1; off = 0
                if off == len(views[fi]):
                    fi += 1; off = 0
                value = self._convert(np.frombuffer(block, dtype='<u2').reshape(count_blocks, 8, 128))
                t = e['tensor']
                if t % 2 == 0:
                    self.keys[t//2, self.lane, :, :, b:b+count_blocks] = value.transpose(1, 2, 0)
                else:
                    self.values[t//2, self.lane, :, b:b+count_blocks, :] = value.transpose(1, 0, 2)
                self.seen[t, b:b+count_blocks] = True
                b += count_blocks
        self.payload_bytes += total
        self.frame_count += len(frames)
        self.entry_count += len(entries)
        self.complete = bool(message['is_final'])
        return self.complete

    def _accept_unit_frame(self, message, frame):
        """Native layer-group fast path: vectorize even interleaved block order."""
        entries = message['entries']
        if isinstance(entries, np.ndarray):
            tier, tensor, block = (entries[name] for name in ('tier', 'tensor', 'block'))
        else:
            meta = np.asarray([(e['tier'], e['tensor'], e['block']) for e in entries], dtype=np.int64)
            tier, tensor, block = meta.T
        if (np.any(tier != 0) or np.any(tensor < 0) or np.any(tensor >= 64) or
                np.any(block < 0) or np.any(block >= self.n)):
            raise ProtocolError('unsupported KV tensor, tier, or block index')
        flat = tensor*self.n + block
        if len(np.unique(flat)) != len(flat) or self.seen[tensor, block].any():
            raise ProtocolError('duplicate KV block')
        if message['is_final'] and int(self.seen.sum())+len(entries) != 64*self.n:
            raise ProtocolError('final batch has incomplete KV coverage')
        raw = np.frombuffer(frame, dtype='<u2').reshape(len(entries), 8, 128)
        for t in np.unique(tensor):
            rows = np.flatnonzero(tensor == t)
            positions = block[rows]
            selected = raw[rows[0]:rows[-1]+1] if rows[-1]-rows[0]+1 == len(rows) else raw[rows]
            value = self._convert(selected)
            contiguous = np.array_equal(positions, np.arange(positions[0], positions[0]+len(positions)))
            destination = slice(int(positions[0]), int(positions[-1])+1) if contiguous else positions
            if t % 2 == 0:
                target = self.keys[t//2, self.lane]
                target[:, :, destination] = value.transpose(1, 2, 0)
            else:
                target = self.values[t//2, self.lane]
                target[:, destination, :] = value.transpose(1, 0, 2)
        self.seen[tensor, block] = True
        self.payload_bytes += len(frame)
        self.frame_count += 1
        self.entry_count += len(entries)
        self.complete = bool(message['is_final'])
        return self.complete

    def _convert(self, raw):
        if self.native_dl:
            return self.converter(raw)
        scratch = self.scratch[:len(raw)]
        np.left_shift(raw, 16, out=scratch, dtype=np.uint32)
        return scratch.view(np.float32)
