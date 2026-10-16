import unittest
import numpy as np
from rngd_pd.native_v4.packed_kv import PackedKVView
from rngd_pd.native_v4.atom_resident import AtomResidentDecoder

class PackedKVTest(unittest.TestCase):
    def test_layout_and_admission_tail_and_bank_ownership(self):
        storage = np.zeros(64*2*8*512*128, np.uint16)
        keys, values = (PackedKVView(storage,2,k) for k in (0,1))
        for layer in range(32):
            keys[layer,1,:,:,0:3] = layer+1
            values[layer,1,:,0:3,:] = layer+101
        calls=[]
        class Runtime:
            def set_kv_cache(self,lane,start,end,source):
                calls.append(source)
                packed=source.reshape(64,2,8,512,128)
                for slot,index in enumerate(sorted(range(64),key=lambda i:f'past_key_values_{i}')):
                    expected=index//2+(1 if index%2==0 else 101)
                    np.testing.assert_array_equal(packed[slot,1,:,:3],expected)
                self.assert_tail=not packed[:,lane,:,end:].any()
        decoder=AtomResidentDecoder.__new__(AtomResidentDecoder)
        decoder.direct_admit=True; decoder.batch=2; decoder.runtime=Runtime()
        # Dirty tail must never reach the next admitted request.
        keys.packed[:,1,:,3:]=65535
        decoder.admit([{'tokens':[1,2,3,4]},{'tokens':[1,2,3,4]}],(keys,values))
        self.assertTrue(all(x is storage for x in calls))
        self.assertFalse(keys.packed[:,1,:,3:].any())
        self.assertTrue(np.shares_memory(keys[31,1],storage))
        keys.clear_lane(1)
        self.assertFalse(keys[31,1].any())
        self.assertTrue(values[31,1,:,:3].any())
        with self.assertRaises(IndexError):
            keys[:,1]

if __name__=='__main__': unittest.main()
