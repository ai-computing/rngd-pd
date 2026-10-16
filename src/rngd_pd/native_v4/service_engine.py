"""Persistent RNGD prefill and ATOM continuous decode with bounded host staging."""
import json
import hashlib
import shutil
import importlib.metadata
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
import zmq
from .atom_resident import AtomResidentDecoder
from .coordinator import Coordinator
from .continuous import decode_loop
from .kv import KVReceiver,host_layout
from .wire import parse,handshake,pull,control,ProtocolError

class ServiceEngine:
    def __init__(self,args):
        self.args=args;self.out=Path(args.out);self.out.mkdir(parents=True,exist_ok=False)
        self.lock=threading.Lock();self.ready=threading.Event();self.decoder=None;self.child=None;self.threads=[]
        self.coordinator=Coordinator(args.batch,args.capacity,args.policy,args.batch_wait_ms/1000,args.request_ttl)
        self.log=(self.out/'native.log').open('w')
        self.record(dict(kind='config',settings=vars(args)))
        package=Path(__file__).resolve().parent
        shutil.copytree(package,self.out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        self.record(dict(kind='provenance',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')},
            rebel_version=importlib.metadata.version('rebel-compiler'),
            repo_sha=subprocess.check_output(['git','-C',str(package),'rev-parse','HEAD'],text=True).strip(),
            repo_dirty=bool(subprocess.check_output(['git','-C',str(package),'status','--porcelain'],text=True).strip())))

    def record(self,event):
        with self.lock,(self.out/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=time.time(),monotonic=time.monotonic(),**event),default=str)+'\n')

    def start(self):
        t=threading.Thread(target=self._guard,args=(self._prefill,),daemon=True,name='rngd-prefill')
        self.threads.append(t);t.start()

    def _guard(self,fn):
        try:fn()
        except BaseException as exc:
            self.record(dict(kind='fatal',error=repr(exc),traceback=traceback.format_exc()))
            self.coordinator.close(repr(exc));self.ready.clear()

    def _prefill(self):
        a=self.args
        self.decoder=AtomResidentDecoder(a.artifact,a.tables,a.atom_device,a.batch,
            bulk_admit=True,token_major_host=True,skip_clear=True,direct_admit=True)
        ctx=zmq.Context();rx=ctx.socket(zmq.PULL);tx=ctx.socket(zmq.PUSH)
        rx.setsockopt(zmq.MAXMSGSIZE,64*512*2048);rx.setsockopt(zmq.RCVHWM,32)
        tx.setsockopt(zmq.SNDTIMEO,5000)
        rx.bind(f'tcp://127.0.0.1:{a.receive_port}');tx.connect(f'tcp://127.0.0.1:{a.prefill_port}')
        try:
            env=os.environ.copy()
            self.child=subprocess.Popen([a.furiosa_python,'-u','-m','rngd_pd.native_v4.native_worker',
                '--model',a.model,'--fxb',a.fxb,'--device',a.rngd_device,'--port',str(a.prefill_port),
                '--peer',f'127.0.0.1:{a.receive_port}','--events',str(self.out/'native-events.jsonl'),
                '--concurrency',str(a.prefill_concurrency)],stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,env=env)
            deadline=time.monotonic()+120;events=self.out/'native-events.jsonl'
            while not events.exists() or 'initialized' not in events.read_text():
                if self.child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('native initialization failed')
                if self.coordinator.stopped:return
                time.sleep(.1)
            t=threading.Thread(target=self._guard,args=(self._decode,),daemon=True,name='atom-decode')
            self.threads.append(t);t.start();self.ready.set();self.record(dict(kind='ready'))
            while not self.coordinator.stopped:
                batch=self.coordinator.prefill_batch()
                if not batch:break
                jobs=[dict(id=s.id,tokens=s.tokens,max_new_tokens=1) for s in batch]
                receivers={s.id:KVReceiver(s.id,len(s.tokens)-1,self.decoder.keys,self.decoder.values,s.lane) for s in batch}
                pending=set(receivers);announced=set()
                self.child.stdin.write(json.dumps(jobs)+'\n');self.child.stdin.flush()
                self.record(dict(kind='prefill_batch',ids=[s.id for s in batch],lanes=[s.lane for s in batch]))
                deadline=time.monotonic()+120
                while pending:
                    if self.coordinator.stopped:return
                    if self.child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('native transfer timed out or worker failed')
                    if not rx.poll(100):continue
                    frames=rx.recv_multipart(copy=False);m=parse(bytes(frames[0]),array_entries=True);rid=m.get('request_id')
                    if rid not in pending:raise ProtocolError('unexpected request ID')
                    if m['kind']=='ready' and rid not in announced:
                        announced.add(rid);tx.send(pull(rid,list(range(receivers[rid].n))))
                        self.record(dict(kind="kv_ready_notice",id=rid))
                    elif m['kind']=='kv' and rid in announced:
                        if receivers[rid].accept(m,[f.buffer for f in frames[1:]]):
                            tx.send(control(3,rid));pending.remove(rid)
                            self.record(dict(kind='kv_received',id=rid,bytes=receivers[rid].payload_bytes))
                    else:raise ProtocolError('unexpected connector event')
                self.coordinator.prefilled(batch)
        finally:
            self.ready.clear()
            if self.child and self.child.poll() is None:
                self.child.terminate()
                try:self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
            rx.close(linger=0);tx.close(linger=0);ctx.term()

    def _decode(self):
        c=self.coordinator
        def token(job,value):
            c.token(job,value);self.record(dict(kind='token',id=job['id'],token=value,lane=job['_lane']))
        def finish(job,reason):
            self.record(dict(kind='finished',id=job['id'],reason=reason));c.finish(job,reason)
        decode_loop(self.decoder,c.take_ready,token,finish,lambda:c.stopped,self.record)

    def close(self):
        self.coordinator.close();self.ready.clear()
        for t in self.threads:t.join(timeout=15)
        if any(t.is_alive() for t in self.threads):
            self.record(dict(kind='shutdown_timeout'))
            return # Do not free buffers underneath an in-flight SDK call.
        if self.decoder:self.decoder.close()
        self.log.close()
