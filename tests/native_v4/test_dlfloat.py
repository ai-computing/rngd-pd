import unittest
import numpy as np
from rngd_pd.native_v4.dlfloat import bf16_expanded_to_dlfloat


class DLFloatTests(unittest.TestCase):
    def test_known_patterns(self):
        values=np.array([0.,-0.,1.,2.,4.,10.,-2.],np.float32)
        np.testing.assert_array_equal(bf16_expanded_to_dlfloat(values),
                                      [0,0x8000,0x3e00,0x4000,0x4200,0x4480,0xc000])

    def test_bf16_range_roundtrip(self):
        raw=np.arange(65536,dtype=np.uint32)
        exponent=(raw>>7)&255
        selected=raw[(exponent>=97)&(exponent<=159)]
        original=(selected<<16).view(np.float32)
        dl=bf16_expanded_to_dlfloat(original).astype(np.uint32)
        rebuilt=((dl&0x8000)<<16)|((((dl>>9)&63)+96)<<23)|((dl&511)<<14)
        np.testing.assert_array_equal(rebuilt,original.view(np.uint32))

    def test_reject_non_bf16_and_nonfinite(self):
        for values in [[np.nan],[np.inf],[1.00001]]:
            with self.assertRaises(ValueError):bf16_expanded_to_dlfloat(np.array(values,np.float32))


if __name__=='__main__':unittest.main()
