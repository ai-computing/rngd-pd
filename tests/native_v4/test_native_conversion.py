import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rngd_pd.native_v4 import native_conversion
from rngd_pd.native_v4.dlfloat import bf16_bits_to_dlfloat

@unittest.skipUnless(shutil.which('cc'), 'C compiler required')
class NativeConversionTests(unittest.TestCase):
    def test_exhaustive_format_and_strided_input(self):
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder)/'converter.so'
            source = Path(native_conversion.__file__).with_name('dlfloat_native.c')
            subprocess.run(['cc','-O3','-fPIC','-shared',str(source),'-o',str(binary)],check=True)
            convert = native_conversion.NativeConverter(binary)
            bits = np.arange(65536,dtype=np.uint16)
            finite = bits[(bits & 0x7f80) != 0x7f80]
            np.testing.assert_array_equal(convert(finite),bf16_bits_to_dlfloat(finite))
            matrix = finite.reshape(255,256).T
            np.testing.assert_array_equal(convert(matrix),bf16_bits_to_dlfloat(matrix))
            for value in bits[(bits & 0x7f80) == 0x7f80]:
                with self.assertRaises(ValueError):
                    convert(np.array([0,value,0],dtype=np.uint16))
