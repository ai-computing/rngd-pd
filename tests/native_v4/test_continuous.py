import unittest
import numpy as np
from rngd_pd.native_v4.continuous import decode_loop

class ContinuousTest(unittest.TestCase):
    def test_refill_preserves_other_lane_and_position(self):
        emitted={};ended=[];steps=[]
        class Runtime:
            def run(self,emb,cos,sin,mask,pos,blocks):
                steps.append(pos[:,0].tolist())
                logits=np.zeros((2,20));logits[np.arange(2),pos[:,0]+1]=1
                return logits
        class Decoder:
            direct_admit=True;batch=2;runtime=Runtime()
            embed=np.zeros((20,2));cos=sin=binary_mask=np.zeros((512,2));blocks=np.arange(2)
            keys=values=None
            def admit(self,jobs,buffers): pass
        jobs=[dict(id='a',tokens=[1,2],max_new_tokens=1,_lane=0),dict(id='b',tokens=[1,2,3],max_new_tokens=3,_lane=1),dict(id='c',tokens=[1,2,3,4],max_new_tokens=2,_lane=0)]
        for j in jobs:j['cancelled']=lambda:False
        count=[0]
        def ready():
            count[0]+=1
            return jobs[:2] if count[0]==1 else ([jobs[2]] if count[0]==2 else [])
        decode_loop(Decoder(),ready,lambda j,t:emitted.setdefault(j['id'],[]).append(t),lambda j,r:ended.append(j['id']),lambda:len(ended)==3)
        self.assertEqual(emitted,dict(a=[2],b=[3,4,5],c=[4,5]))
        self.assertEqual(steps,[[1,2],[3,3],[4,4]])
