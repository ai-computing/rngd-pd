import unittest
from rngd_pd.native_v4.coordinator import Coordinator,AdmissionError

class CoordinatorTest(unittest.TestCase):
    def test_stage_ownership_mismatch_duplicate_and_reuse(self):
        c=Coordinator(2,2,batch_wait=0)
        a=c.submit('a',[1,2],'a');b=c.submit('b',[3,4],'b')
        with self.assertRaises(AdmissionError):c.submit('c',[1,2],'c')
        batch=c.prefill_batch();self.assertEqual(len(c.free),0)
        c.cancel('a');self.assertEqual(len(c.free),0) # writer still owns lane
        c.prefilled(batch);self.assertEqual(len(c.free),1)
        with self.assertRaises(AdmissionError):c.authorize_decode('b','wrong',2,False,[])
        c.authorize_decode('b','b',2,False,[])
        with self.assertRaises(AdmissionError):c.authorize_decode('b','b',2,False,[])
        job=c.take_ready()[0];c.cancel('b');self.assertEqual(len(c.free),1)
        c.finish(job,'cancelled');self.assertEqual(c.free,{0,1})
        self.assertEqual(c.metrics()['active'],0)

    def test_expired_prefill_reclaimed(self):
        c=Coordinator(1,batch_wait=0)
        a=c.submit('a',[1,2],'a');c.prefilled(c.prefill_batch())
        a.deadline=0;c.reap();self.assertEqual(c.free,{0})
        with self.assertRaises(AdmissionError):c.authorize_decode('a','a',2,False,[])

class CancellationIdentityTest(unittest.TestCase):
    def test_old_response_cannot_cancel_reused_id(self):
        c=Coordinator(1,batch_wait=0)
        old=c.submit('same',[1,2],'fp');c.cancel('same',old)
        new=c.submit('same',[1,2],'fp');c.cancel('same',old)
        self.assertFalse(new.cancelled.is_set())
        self.assertIs(c.states['same'],new)
