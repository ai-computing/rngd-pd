import json
import os
import struct
import tempfile
import unittest
import shutil

from rngd_pd.swap import delta


def write_st(path, tensors):
    """Minimal safetensors writer: {name: (dtype, shape, bytes)} in the given order."""
    header, off, blobs = {}, 0, []
    for name, (dt, shape, data) in tensors.items():
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class DeltaTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.base, self.new = os.path.join(self.d, "base"), os.path.join(self.d, "new")
        os.makedirs(self.base)
        os.makedirs(self.new)
        self.t = {"a.weight": ("BF16", [4, 8], bytes(64)),
                  "b.weight": ("BF16", [16, 8], bytes(256)),
                  "c.weight": ("F32", [8], bytes(32))}
        write_st(os.path.join(self.base, "m.safetensors"), self.t)

    def test_header_offsets_are_absolute(self):
        p = os.path.join(self.base, "m.safetensors")
        h = delta.read_header(p)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        self.assertEqual(h["a.weight"]["start"], 8 + n)
        self.assertEqual(h["b.weight"]["end"] - h["b.weight"]["start"], 256)

    def test_ranges_to_tensors_and_rows(self):
        h = delta.read_header(os.path.join(self.base, "m.safetensors"))
        b0 = h["b.weight"]["start"]
        got = delta.tensors_touched(h, [(b0 + 20, 4), (b0 + 22, 30), (b0 + 200, 1)])
        self.assertEqual(list(got), ["b.weight"])
        self.assertEqual(got["b.weight"], [(20, 52), (200, 201)])
        # rows of 8 bf16 = 16 bytes
        self.assertEqual(delta.rows_of(h["b.weight"], got["b.weight"]), [(1, 4), (12, 13)])
        # a range across the boundary of two tensors hits both
        both = delta.tensors_touched(h, [(h["a.weight"]["end"] - 2, 4)])
        self.assertEqual(sorted(both), ["a.weight", "b.weight"])
        # the header alone touches nothing
        self.assertEqual(delta.tensors_touched(h, [(0, 8)]), {})

    def test_plan_local_and_onggi_agree(self):
        t = dict(self.t)
        b = bytearray(t["b.weight"][2])
        b[100] = 7
        t["b.weight"] = ("BF16", [16, 8], bytes(b))
        write_st(os.path.join(self.new, "m.safetensors"), t)
        local = delta.plan_update(self.base, self.new, None, block=16)
        self.assertEqual(sorted(local.tensors), ["b.weight"])
        # block-granular (16 B blocks from the file start): covers row 6, maybe a neighbour
        (r0, r1), = local.tensors["b.weight"]["rows"]
        self.assertTrue(r0 <= 6 < r1 and r1 - r0 <= 2)
        h = delta.read_header(os.path.join(self.new, "m.safetensors"))
        off = h["b.weight"]["start"] + 100
        doc = {"files": [{"path": "m.safetensors", "base": "m.safetensors", "size": 1,
                          "changed": [[off, 1]]}]}
        og = delta.plan_update(self.base, self.new, delta.load_onggi_ranges(json.dumps(doc)))
        self.assertEqual(og.summary()["names"], ["b.weight"])
        self.assertEqual(og.upload_bytes, 256)

    def test_refine_drops_neighbours_flagged_by_a_coarse_range(self):
        t = dict(self.t)
        b = bytearray(t["b.weight"][2])
        b[100] = 7
        t["b.weight"] = ("BF16", [16, 8], bytes(b))
        write_st(os.path.join(self.new, "m.safetensors"), t)
        h = delta.read_header(os.path.join(self.new, "m.safetensors"))
        # one coarse range over a, all of b and c
        start = h["a.weight"]["start"]
        coarse = {"m.safetensors": [(start, h["c.weight"]["end"] - start)]}
        raw = delta.plan_update(self.base, self.new, coarse, refine=False)
        self.assertEqual(sorted(raw.tensors), ["a.weight", "b.weight", "c.weight"])
        fine = delta.plan_update(self.base, self.new, coarse)
        self.assertEqual(sorted(fine.tensors), ["b.weight"])
        self.assertEqual(fine.tensors["b.weight"]["rows"], [(6, 7)])

    def test_unpaired_file_is_all_new_and_shape_change_is_whole(self):
        doc = {"files": [{"path": "x.safetensors", "size": 10}]}
        self.assertEqual(delta.load_onggi_ranges(doc), {"x.safetensors": [(0, 10)]})
        t = dict(self.t)
        t["c.weight"] = ("F32", [4, 2], bytes(32))
        write_st(os.path.join(self.new, "m.safetensors"), t)
        p = delta.plan_update(self.base, self.new, {})
        # Enlarging the JSON header moves every absolute data offset. An
        # incomplete range mapping must not let moved tensors remain stale.
        self.assertEqual(p.header_changes, set(t))
        self.assertIn("c.weight", p.tensors)

    def test_resharding_and_removed_names(self):
        write_st(os.path.join(self.new, "renamed.safetensors"), {"a.weight": self.t["a.weight"]})
        p = delta.plan_update(self.base, self.new, {})
        self.assertEqual(set(p.tensors), {"a.weight"})
        self.assertEqual(p.removed, {"b.weight", "c.weight"})

    def test_reject_bad_ranges_and_truncated_files(self):
        for ranges in [[(-1, 4)], [(0, -1)], [(0.5, 2)]]:
            with self.assertRaises(ValueError):
                delta.tensors_touched({}, ranges)
        with self.assertRaises(ValueError):
            delta.load_onggi_ranges({"files": [{"path": "../x", "size": 1}]})
        with self.assertRaises(ValueError):
            delta.diff_ranges_local("unused", "unused", block=0)
        path = os.path.join(self.new, "truncated.safetensors")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", 999999999))
        with self.assertRaises(ValueError):
            delta.read_header(path)

    def test_overlapping_ranges_count_once(self):
        doc = {"files": [{"path": "m.safetensors", "base": "m.safetensors",
                          "changed": [[10, 10], [15, 10], [25, 0]]}]}
        self.assertEqual(delta.load_onggi_ranges(doc), {"m.safetensors": [(10, 15)]})

    def test_diff_without_ranges_cannot_be_used_for_delta(self):
        doc = {"files": [{"path": "m.safetensors", "base": "m.safetensors",
                          "size": 16, "same": 8, "new": 8, "changed_runs": 1}]}
        with self.assertRaisesRegex(ValueError, "--ranges"):
            delta.load_onggi_ranges(doc)


if __name__ == "__main__":
    unittest.main()
