"""Bounded two-stage admission and KV slot ownership for a resident PD server."""
from dataclasses import dataclass, field
from threading import Condition, Event
from queue import Queue
import time

class AdmissionError(Exception):
    def __init__(self, status, message):
        self.status=status
        super().__init__(message)

@dataclass
class RequestState:
    id: str
    tokens: list
    fingerprint: str
    hint: int
    deadline: float
    created: float = field(default_factory=time.monotonic)
    phase: str = 'queued'
    lane: int | None = None
    ready: Event = field(default_factory=Event)
    cancelled: Event = field(default_factory=Event)
    output: Queue = field(default_factory=lambda: Queue(maxsize=514))
    job: dict | None = None
    error: str | None = None

class Coordinator:
    def __init__(self, batch, capacity=64, policy='length', batch_wait=.01, ttl=120):
        if batch<1 or capacity<batch or policy not in ('length','fifo') or batch_wait<0 or ttl<=0:
            raise ValueError('invalid admission configuration')
        self.batch=batch;self.capacity=capacity;self.policy=policy;self.batch_wait=batch_wait;self.ttl=ttl
        self.cv=Condition();self.states={};self.free=set(range(batch));self.stopped=False;self.failure=None
        self.completed=0;self.rejected=0

    def submit(self,rid,tokens,fingerprint,hint=1):
        with self.cv:
            if self.stopped: raise AdmissionError(503,self.failure or 'server stopping')
            if rid in self.states: raise AdmissionError(409,'request ID already active')
            if len(self.states)>=self.capacity:
                self.rejected+=1;raise AdmissionError(429,'admission queue full')
            state=RequestState(rid,list(tokens),fingerprint,hint,time.monotonic()+self.ttl)
            self.states[rid]=state;self.cv.notify_all();return state

    def _release(self,s,reason):
        if s.lane is not None:
            self.free.add(s.lane)
        self.states.pop(s.id,None);s.phase='finished';s.ready.set()
        s.output.put_nowait(('finish',reason));self.cv.notify_all()

    def reap(self):
        with self.cv:
            for s in list(self.states.values()):
                if time.monotonic()>=s.deadline: s.cancelled.set()
                # In-flight prefill/decode own their slots until they acknowledge.
                if s.cancelled.is_set() and s.phase in ('queued','prefilled','decode_ready'):
                    self._release(s,'cancelled')

    def prefill_batch(self):
        with self.cv:
            while not self.stopped:
                self.reap()
                queued=[s for s in self.states.values() if s.phase=='queued']
                if queued and self.free:
                    wait=self.batch_wait-(time.monotonic()-queued[0].created)
                    if len(queued)<len(self.free) and wait>0:
                        self.cv.wait(wait);continue
                    # A finite arrival window limits starvation. Old requests
                    # override length ordering after a bounded queue wait.
                    window=queued[:self.batch*4]
                    if self.policy=='length' and time.monotonic()-window[0].created < .1:
                        window.sort(key=lambda s:(-s.hint,-len(s.tokens)))
                    chosen=window[:len(self.free)]
                    for s in chosen:
                        s.lane=min(self.free);self.free.remove(s.lane);s.phase='prefilling'
                    return chosen
                self.cv.wait(.02)
            return []

    def prefilled(self,states):
        with self.cv:
            for s in states:
                if s.cancelled.is_set(): self._release(s,'cancelled')
                else: s.phase='prefilled';s.ready.set()
            self.cv.notify_all()

    def authorize_decode(self,rid,fingerprint,max_tokens,ignore_eos,eos_ids):
        with self.cv:
            self.reap()
            s=self.states.get(rid)
            if s is None: raise AdmissionError(404,'no live prefill for request ID')
            if s.fingerprint!=fingerprint: raise AdmissionError(409,'prefill/decode prompt mismatch')
            if s.phase!='prefilled': raise AdmissionError(409,'prefill not ready or decode already claimed')
            if type(max_tokens) is not int or not 1<=max_tokens<=512-len(s.tokens):
                raise AdmissionError(400,'invalid output length')
            s.job=dict(id=s.id,tokens=s.tokens,max_new_tokens=max_tokens,_lane=s.lane,
                       ignore_eos=ignore_eos,eos_ids=eos_ids,cancelled=s.cancelled.is_set)
            s.phase='decode_ready';self.cv.notify_all();return s

    def take_ready(self):
        with self.cv:
            self.reap()
            chosen=[s for s in self.states.values() if s.phase=='decode_ready']
            for s in chosen: s.phase='decoding'
            return [s.job for s in chosen]

    def token(self,job,value):
        with self.cv: self.states[job['id']].output.put_nowait(('token',value))

    def finish(self,job,reason):
        with self.cv:
            s=self.states[job['id']];self.completed+=1;self._release(s,reason)

    def cancel(self,rid,expected_state=None):
        with self.cv:
            state=self.states.get(rid)
            if state is not None and (expected_state is None or state is expected_state): state.cancelled.set()
            self.reap();self.cv.notify_all()

    def close(self,error=None):
        with self.cv:
            if self.stopped:return
            self.stopped=True;self.failure=error
            for s in self.states.values():
                s.error=error or 'server stopped';s.cancelled.set();s.ready.set()
                s.output.put_nowait(('error',s.error))
            self.cv.notify_all()

    def metrics(self):
        with self.cv:
            return dict(active=len(self.states),free_slots=len(self.free),completed=self.completed,rejected=self.rejected,
                        **{p:sum(s.phase==p for s in self.states.values()) for p in ('queued','prefilling','prefilled','decode_ready','decoding')})
