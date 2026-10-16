"""Single-owner ATOM decode loop with safe slot refill between forward calls."""
import time
import numpy as np


def decode_loop(decoder, take_ready, token, finish, stopped, record=lambda x:None):
    if not decoder.direct_admit:
        raise ValueError('continuous decode requires direct-admit buffers')
    active = {}
    current = np.zeros(decoder.batch, np.int64)
    positions = np.zeros(decoder.batch, np.int32)
    while not stopped():
        for job in take_ready():
            lane = job['_lane']
            if lane in active:
                raise RuntimeError('slot reused before prior decode finished')
            start = time.perf_counter()
            decoder.admit([job], (decoder.keys,decoder.values))
            active[lane] = [job,0]
            current[lane] = job['tokens'][-1]
            positions[lane] = len(job['tokens'])-1
            record(dict(kind='admit',id=job['id'],lane=lane,seconds=time.perf_counter()-start))
        for lane,(job,count) in list(active.items()):
            if job['cancelled']():
                del active[lane]; current[lane]=0;positions[lane]=0
                finish(job, 'cancelled')
        if not active:
            time.sleep(.002)
            continue
        start = time.perf_counter()
        logits = decoder.runtime.run(decoder.embed[current],decoder.cos[positions],decoder.sin[positions],
                                     decoder.binary_mask[positions],positions.reshape(decoder.batch,1),decoder.blocks)
        if not np.isfinite(logits[list(active)]).all():
            raise FloatingPointError('non-finite active logits')
        nxt = np.argmax(logits,axis=-1)
        record(dict(kind='decode_step',seconds=time.perf_counter()-start,useful_lanes=len(active)))
        for lane,(job,count) in list(active.items()):
            value=int(nxt[lane]);count+=1
            token(job,value)
            if count >= job['max_new_tokens'] or (not job.get('ignore_eos',False) and value in job.get('eos_ids',())):
                del active[lane];current[lane]=0;positions[lane]=0
                finish(job,'length' if count>=job['max_new_tokens'] else 'stop')
            else:
                active[lane][1]=count;current[lane]=value;positions[lane]+=1
