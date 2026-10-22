"""Which tensors changed between two checkpoints, from changed byte ranges.

The byte ranges come from `onggi diff --ranges --json` (or, without onggi, from
comparing the two files block by block). They are mapped to tensors through the
safetensors header, an open format: 8 bytes of little-endian header length, a
JSON header with each tensor's dtype, shape and [begin, end) data offsets, then
the data. Nothing else about either file is interpreted.

Standard library only, so the planning runs on any host.
"""
import bisect
import json
import os
import struct

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
               "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path):
    """{tensor name: dict(dtype, shape, start, end)} with absolute file offsets."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        out[name] = dict(dtype=info["dtype"], shape=list(info["shape"]),
                         start=base + b, end=base + e)
    return out


def load_onggi_ranges(doc):
    """`onggi diff --ranges --json` output -> {file path: [(offset, length)]}.

    Accepts the parsed JSON, a JSON string or a path to a file holding it. A
    file without a paired base (`base` empty) has no "same" bytes: every byte
    of it counts as changed.
    """
    if isinstance(doc, (str, os.PathLike)) and os.path.exists(doc):
        with open(doc) as f:
            doc = json.load(f)
    elif isinstance(doc, str):
        doc = json.loads(doc)
    out = {}
    for fd in doc.get("files") or []:
        if not fd.get("base"):
            out[fd["path"]] = [(0, int(fd["size"]))]
        else:
            out[fd["path"]] = [(int(o), int(n)) for o, n in fd.get("changed") or []]
    return out


def diff_ranges_local(base_path, new_path, block=1 << 20):
    """Changed [offset, length] runs between two files, by direct comparison.

    The fallback when onggi is not installed. Coarser than onggi (whole blocks)
    and it reads both files, but it gives the same kind of answer.
    """
    runs = []
    size_new = os.path.getsize(new_path)
    with open(base_path, "rb") as fb, open(new_path, "rb") as fn:
        off = 0
        while off < size_new:
            a, b = fb.read(block), fn.read(block)
            if a != b:
                if runs and runs[-1][0] + runs[-1][1] == off:
                    runs[-1][1] += len(b)
                else:
                    runs.append([off, len(b)])
            off += len(b)
    return [tuple(r) for r in runs]


def tensors_touched(header, ranges):
    """Tensors whose data overlaps any changed range.

    Returns {name: [(begin, end)]}: the changed bytes relative to the start of
    that tensor's data, merged. A range that hits only the header (the JSON
    at the front) marks nothing here; changes to the header are checked by
    plan_update comparing the two headers.
    """
    items = sorted(((v["start"], v["end"], k) for k, v in header.items()))
    starts = [s for s, _, _ in items]
    out = {}
    for off, length in sorted(ranges):
        lo, hi = off, off + length
        i = max(bisect.bisect_right(starts, lo) - 1, 0)
        while i < len(items) and items[i][0] < hi:
            s, e, name = items[i]
            a, b = max(s, lo), min(e, hi)
            if a < b:
                out.setdefault(name, []).append((a - s, b - s))
            i += 1
    for name, spans in out.items():
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[name] = [tuple(m) for m in merged]
    return out


def rows_of(info, spans):
    """Byte spans inside one tensor -> [row_start, row_end) along dim 0."""
    shape = info["shape"]
    if not shape:
        return [(0, 1)]
    row = DTYPE_BYTES[info["dtype"]]
    for d in shape[1:]:
        row *= d
    rows = []
    for a, b in spans:
        r0, r1 = a // row, -(-b // row)
        if rows and r0 <= rows[-1][1]:
            rows[-1] = (rows[-1][0], max(rows[-1][1], r1))
        else:
            rows.append((r0, r1))
    return rows


class UpdatePlan:
    """What to upload to turn the resident base weights into the new ones."""

    def __init__(self, tensors, changed_bytes, total_bytes, header_changes):
        self.tensors = tensors                  # {name: dict(file, rows, bytes)}
        self.changed_bytes = changed_bytes      # bytes the diff marked
        self.total_bytes = total_bytes          # all tensor bytes in the new files
        self.header_changes = header_changes    # names whose dtype/shape/placement moved

    @property
    def upload_bytes(self):
        return sum(t["bytes"] for t in self.tensors.values())

    def summary(self):
        return dict(tensors=len(self.tensors), upload_bytes=self.upload_bytes,
                    changed_bytes=self.changed_bytes, total_bytes=self.total_bytes,
                    header_changes=sorted(self.header_changes),
                    names=sorted(self.tensors))


def plan_update(base_dir, new_dir, ranges_by_file=None, block=1 << 20):
    """Plan an update from the checkpoint in base_dir to the one in new_dir.

    ranges_by_file: {relative file path: [(offset, length)]} for the files of
    new_dir, as load_onggi_ranges gives it. None compares the files locally.
    A file missing from the mapping is taken as unchanged; a tensor whose
    dtype or shape differs between the two headers is planned whole.
    """
    files = sorted(f for f in os.listdir(new_dir) if f.endswith(".safetensors"))
    tensors, header_changes = {}, set()
    changed = total = 0
    base_headers = {}
    for f in files:
        bp = os.path.join(base_dir, f)
        if os.path.exists(bp):
            for name, info in read_header(bp).items():
                base_headers[name] = info
    for f in files:
        np_ = os.path.join(new_dir, f)
        hdr = read_header(np_)
        total += sum(v["end"] - v["start"] for v in hdr.values())
        if ranges_by_file is None:
            bp = os.path.join(base_dir, f)
            rng = diff_ranges_local(bp, np_, block) if os.path.exists(bp) else \
                [(0, os.path.getsize(np_))]
        else:
            rng = ranges_by_file.get(f, [])
        changed += sum(n for _, n in rng)
        for name, spans in tensors_touched(hdr, rng).items():
            info = hdr[name]
            tensors[name] = dict(file=f, rows=rows_of(info, spans),
                                 bytes=info["end"] - info["start"])
        for name, info in hdr.items():
            old = base_headers.get(name)
            if old is None or old["dtype"] != info["dtype"] or old["shape"] != info["shape"]:
                header_changes.add(name)
                tensors[name] = dict(file=f, rows=[(0, info["shape"][0] if info["shape"] else 1)],
                                     bytes=info["end"] - info["start"])
    return UpdatePlan(tensors, changed, total, header_changes)
