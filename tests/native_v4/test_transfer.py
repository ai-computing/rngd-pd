import unittest
from pathlib import Path
import numpy as np
from rngd_pd.native_v4.kv import KVReceiver, host_layout
from rngd_pd.native_v4.wire import parse, control, pull, handshake, ProtocolError, ENTRY_DTYPE

class TransferTests(unittest.TestCase):
    def buffers(self):
        return np.zeros((32,1,8,128,512),np.float32),np.zeros((32,1,8,512,128),np.float32)
    def message(self, entries, final=False):
        return dict(kind='kv',request_id='test-long-request',entries=entries,is_final=final)
    def entry(self,t,b=0,n=1):
        return dict(tier=0,tensor=t,block=b,bytes=n*2048)
    def test_codec(self):
        for rid in ['a','abcdefg','abcdefgh','request-1234567890']:
            self.assertEqual(parse(control(1,rid))['request_id'],rid)
            self.assertEqual(parse(pull(rid,[0,1,2]))['blocks'],[0,1,2])
        self.assertEqual(parse(handshake('127.0.0.1:18422',host_layout()))['layout'],host_layout())
    def test_split_and_layout(self):
        k,v=self.buffers();r=KVReceiver('test-long-request',2,k,v)
        bits=np.arange(64*2*8*128,dtype=np.uint16).reshape(64,2,8,128)
        raw=bits.astype('<u2').tobytes()
        entries=[self.entry(t,n=2) for t in range(64)]
        self.assertTrue(r.accept(self.message(entries,True),[raw[:3],raw[3:2047],raw[2047:8193],raw[8193:]]))
        expected=(bits.astype(np.uint32)<<16).view(np.float32)
        # Compare bit patterns, including generated NaN bit patterns.
        np.testing.assert_array_equal(k[:,0,:,:,:2].view(np.uint32),expected[::2].transpose(0,2,3,1).view(np.uint32))
        np.testing.assert_array_equal(v[:,0,:,:2,:].view(np.uint32),expected[1::2].transpose(0,2,1,3).view(np.uint32))
        self.assertEqual(r.payload_bytes,len(raw))
    def test_reject_incomplete_duplicate_bounds(self):
        k,v=self.buffers();r=KVReceiver('test-long-request',2,k,v)
        with self.assertRaises(ProtocolError):r.accept(self.message([self.entry(0)],True),[bytes(2048)])
        self.assertFalse(r.seen.any())
        r.accept(self.message([self.entry(0)]),[bytes(2048)])
        for entry in [self.entry(0),self.entry(64),self.entry(1,b=2),self.entry(1,n=3)]:
            with self.assertRaises(ProtocolError):r.accept(self.message([entry]),[bytes(entry['bytes'])])
        with self.assertRaises(ProtocolError):r.accept(self.message([self.entry(1)]),[bytes(2047)])
    def test_random_fragmentation_and_slot_reuse(self):
        k,v=self.buffers()
        rng=np.random.default_rng(422)
        for n in [3,1,4]:
            k.fill(0);v.fill(0)
            r=KVReceiver('test-long-request',n,k,v)
            bits=rng.integers(0,65536,size=(64,n,8,128),dtype=np.uint16)
            raw=bits.astype('<u2').tobytes()
            cuts=sorted(set([0,len(raw)]+rng.integers(0,len(raw),size=37).tolist()))
            frames=[raw[a:b] for a,b in zip(cuts,cuts[1:])]
            entries=[self.entry(t,b=b) for t in range(64) for b in range(n)]
            r.accept(self.message(entries,True),frames)
            expected=(bits.astype(np.uint32)<<16).view(np.float32)
            np.testing.assert_array_equal(k[:,0,:,:,:n].view(np.uint32),expected[::2].transpose(0,2,3,1).view(np.uint32))
            np.testing.assert_array_equal(v[:,0,:,:n,:].view(np.uint32),expected[1::2].transpose(0,2,1,3).view(np.uint32))
            self.assertFalse(k[:,0,:,:,n:].any())
    def test_reject_malformed_headers_and_cancel_roundtrip(self):
        self.assertEqual(parse(control(4,'cancel-request'))['kind'],'cancel')
        for header in [b'',bytes(35),bytes(37),b'\xff'*36]:
            with self.assertRaises((ProtocolError,UnicodeError)):parse(header)
        with self.assertRaises(ProtocolError):pull('x',[512])
    def test_unit_frame_fast_path_matches_fragmented_path(self):
        rng=np.random.default_rng(7);n=7
        indices=rng.permutation(64*n)
        entries=[self.entry(int(i//n),b=int(i%n)) for i in indices]
        raw=rng.integers(0,65536,size=64*n*1024,dtype=np.uint16).tobytes()
        outputs=[]
        for frames in [[raw],[raw[:17],raw[17:]]]:
            k,v=self.buffers();r=KVReceiver('test-long-request',n,k,v)
            self.assertTrue(r.accept(self.message(entries,True),frames))
            outputs.append((k,v))
        for a,b in zip(*outputs):
            np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    def test_real_native_wire_fixtures(self):
        root=Path(__file__).parent/'fixtures'
        for file,tag in [('ready.bin',1),('ack.bin',3)]:
            raw=(root/file).read_bytes();m=parse(raw)
            self.assertEqual(m['request_id'],'native-probe-1')
            self.assertEqual(control(tag,m['request_id']),raw)
        raw=(root/'handshake.bin').read_bytes();m=parse(raw)
        self.assertEqual(m['layout']['element_type'],'Bfloat16')
        self.assertEqual(handshake(m['endpoint'],m['layout']),raw)
        m=parse((root/'kv.bin').read_bytes())
        self.assertEqual([e['tensor'] for e in m['entries']],[0,1,2,3])
        self.assertEqual(sum(e['bytes'] for e in m['entries']),8192)
        self.assertFalse(m['is_final'])
        fast=parse((root/'kv.bin').read_bytes(),array_entries=True)
        np.testing.assert_array_equal(fast['entries']['tensor'],[0,1,2,3])

    def test_array_metadata_and_direct_dlfloat(self):
        from rngd_pd.native_v4.dlfloat import bf16_expanded_to_dlfloat
        n=3
        entries=np.array([(0,t,b,2048) for t in range(64) for b in range(n)],dtype=ENTRY_DTYPE)
        original=np.ones((64,n,8,128),np.float32)*2
        original[:,1]=4;original[:,2]=-1
        raw=(original.view(np.uint32)>>16).astype('<u2').tobytes()
        for frames in [[raw],[raw[:31],raw[31:]]]:
            k=np.zeros((32,1,8,128,512),np.uint16)
            v=np.zeros((32,1,8,512,128),np.uint16)
            r=KVReceiver('test-long-request',n,k,v)
            self.assertTrue(r.accept(self.message(entries,True),frames))
            expected=bf16_expanded_to_dlfloat(original)
            np.testing.assert_array_equal(k[:,0,:,:,:n],expected[::2].transpose(0,2,3,1))
            np.testing.assert_array_equal(v[:,0,:,:n,:],expected[1::2].transpose(0,2,1,3))

if __name__=='__main__':unittest.main()
