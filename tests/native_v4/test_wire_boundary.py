import unittest
from rngd_pd.native_v4.wire import control,pull,parse
class WireBoundaryTest(unittest.TestCase):
    def test_eight_byte_inline_string(self):
        # ArchivedString's entire 8-byte inline capacity is valid; an outlined
        # representation for length 8 is rejected by native rkyv validation.
        for n in (1,7,8,9,16,128):
            rid='r'*n
            b=control(3,rid)
            self.assertEqual(parse(b)['request_id'],rid)
            if n<=8:
                self.assertEqual(len(b),36)
                self.assertEqual(b[8:8+n],rid.encode())
            self.assertEqual(parse(pull(rid,[0,1]))['request_id'],rid)
