import unittest

from rngd_pd.swap.variants import BatchSelector, Variant


class VariantTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(Variant.parse("B8,S512"), Variant(8, 512, 8))
        self.assertEqual(Variant.parse("B4-S256-g2"), Variant(4, 256, 2))
        self.assertEqual(Variant.parse("b16 s512", group=4), Variant(16, 512, 4))
        self.assertEqual(Variant(4, 512).name, "B4-S512-g8")
        self.assertEqual(Variant.parse("B4-S512", group=0).name, "B4-S512")
        with self.assertRaises(ValueError):
            Variant.parse("B8")


class SelectorTest(unittest.TestCase):
    def test_up_needs_consecutive_observations(self):
        s = BatchSelector([4, 8, 16], start=4, up_after=2)
        self.assertEqual(s.observe(6), 4)
        self.assertEqual(s.observe(3), 4)      # streak broken
        self.assertEqual(s.observe(6), 4)
        self.assertEqual(s.observe(12), 16)    # fits the depth, skips 8

    def test_down_hysteresis_and_dwell(self):
        t = [0.0]
        s = BatchSelector([4, 8, 16], start=16, down_after=3, down_ratio=0.5,
                          min_dwell_s=10, clock=lambda: t[0])
        for _ in range(3):
            self.assertEqual(s.observe(9), 16)  # 9 > 0.5*16: no pressure to shrink
        for _ in range(5):
            self.assertEqual(s.observe(3), 16)  # dwell not reached
        t[0] = 11
        self.assertEqual(s.observe(3), 4)

    def test_oscillation_does_not_swap(self):
        s = BatchSelector([4, 8], start=4, up_after=2, down_after=4)
        seen = {s.observe(d) for d in [5, 4, 5, 4, 5, 4, 5, 4]}
        self.assertEqual(seen, {4})

    def test_depth_above_largest(self):
        s = BatchSelector([4, 8], start=8)
        self.assertEqual(s.observe(100), 8)


if __name__ == "__main__":
    unittest.main()
