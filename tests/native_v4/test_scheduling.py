import unittest
from rngd_pd.native_v4.scheduling import plan_batches, padding_stats
class SchedulingTest(unittest.TestCase):
    def test_bounded_reorder_reduces_padding_without_loss(self):
        jobs=[dict(id=i,tokens=[1,2],max_new_tokens=n) for i,n in enumerate([1,64,2,32,1,64,2,32,7])]
        fifo=list(plan_batches(jobs,2))
        packed=list(plan_batches(jobs,2,'length',2))
        self.assertEqual(sorted(j['id'] for b in packed for j in b),list(range(9)))
        self.assertEqual({j['id'] for b in packed[:2] for j in b},set(range(4)))
        self.assertLess(padding_stats(packed,2)['compiled_lane_steps'],padding_stats(fifo,2)['compiled_lane_steps'])
        self.assertEqual([j['id'] for b in fifo for j in b],list(range(9)))
        with self.assertRaises(ValueError): list(plan_batches(jobs,2,'length',0))
