"""Two OpenAI HTTP stages for llm-d shared-storage PD, bound to loopback."""
import argparse
import asyncio
import hashlib
import json
import os
import signal
import threading
import time
import uuid
from pathlib import Path
from queue import Empty
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse,StreamingResponse,PlainTextResponse
from transformers import AutoTokenizer
import uvicorn
from .coordinator import AdmissionError
from .service_engine import ServiceEngine


def parse_request(body,tokenizer,model):
    if not isinstance(body,dict):raise AdmissionError(400,'JSON object required')
    supported={'model','prompt','messages','max_tokens','max_completion_tokens','temperature','top_p','n','stream','stream_options','ignore_eos','cache_hit_threshold'}
    extra=set(body)-supported
    if extra:raise AdmissionError(400,'unsupported options: '+','.join(sorted(extra)))
    if body.get('model')!=model:raise AdmissionError(404,'unknown model')
    if body.get('temperature',0)!=0 or body.get('top_p',1)!=1 or body.get('n',1)!=1:
        raise AdmissionError(400,'only greedy sampling, top_p=1, n=1 supported')
    if type(body.get('stream',False)) is not bool or type(body.get('ignore_eos',False)) is not bool:
        raise AdmissionError(400,'stream and ignore_eos must be boolean')
    options=body.get('stream_options',{})
    if not isinstance(options,dict) or set(options)-{'include_usage'} or type(options.get('include_usage',False)) is not bool:
        raise AdmissionError(400,'only stream_options.include_usage boolean supported')
    if options and not body.get('stream',False):raise AdmissionError(400,'stream_options requires stream=true')
    if body.get('cache_hit_threshold',0)!=0:raise AdmissionError(400,'cache reuse thresholds unsupported')
    if 'messages' in body:
        if 'prompt' in body:raise AdmissionError(400,'choose prompt or messages')
        try:tokens=tokenizer.apply_chat_template(body['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
        except Exception as exc:raise AdmissionError(400,'invalid chat messages') from exc
    else:
        prompt=body.get('prompt')
        if isinstance(prompt,str):tokens=tokenizer.encode(prompt,add_special_tokens=True)
        elif isinstance(prompt,list):tokens=prompt
        else:raise AdmissionError(400,'prompt must be text or token ID list')
    if len(tokens)<2 or any(type(t) is not int or not 0<=t<128256 for t in tokens):raise AdmissionError(400,'invalid prompt tokens')
    count=body.get('max_completion_tokens',body.get('max_tokens',16))
    if type(count) is not int or not 1<=count or len(tokens)+count>512:raise AdmissionError(400,'context limit is 512 tokens')
    fp=hashlib.sha256(json.dumps([model,tokens],separators=(',',':')).encode()).hexdigest()
    return tokens,count,fp


def create_app(engine,tokenizer,role,model,eos_ids=None):
    eos_ids=list(eos_ids) if eos_ids is not None else ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
    app=FastAPI();c=engine.coordinator
    @app.exception_handler(AdmissionError)
    async def admission_error(request,exc):return JSONResponse({'error':{'message':str(exc),'type':'invalid_request_error'}},status_code=exc.status)
    @app.get('/health')
    async def health():return JSONResponse({'status':'ok' if not c.stopped else 'failed'},status_code=200 if not c.stopped else 503)
    @app.get('/ready')
    async def ready():return JSONResponse({'ready':engine.ready.is_set()},status_code=200 if engine.ready.is_set() else 503)
    @app.get('/metrics')
    async def metrics():return PlainTextResponse(''.join(f'koni_pd_{k} {v}\n' for k,v in c.metrics().items()))
    @app.get('/v1/models')
    async def models():return {'object':'list','data':[{'id':model,'object':'model','owned_by':'local'}]}
    @app.post('/v1/completions')
    @app.post('/v1/chat/completions')
    async def complete(request:Request):
        if not engine.ready.is_set():raise AdmissionError(503,c.failure or 'model loading')
        try:body=await request.json()
        except Exception as exc:raise AdmissionError(400,'invalid JSON') from exc
        tokens,count,fp=parse_request(body,tokenizer,model)
        rid=request.headers.get('x-request-id','')
        if role=='combined' and not rid:rid='cmpl-'+uuid.uuid4().hex
        if not rid or len(rid)>128 or not rid.isascii() or any(ord(x)<33 for x in rid):
            raise AdmissionError(400,'x-request-id must be 1..128 visible ASCII characters')
        chat=request.url.path.endswith('/chat/completions');created=int(time.time())
        def response(ids,reason):
            text=tokenizer.decode(ids,skip_special_tokens=True)
            choice={'index':0,'finish_reason':reason,'logprobs':None}
            choice.update({'message':{'role':'assistant','content':text}} if chat else {'text':text})
            return {'id':rid,'object':'chat.completion' if chat else 'text_completion','created':created,'model':model,
                    'choices':[choice],'usage':{'prompt_tokens':len(tokens),'completion_tokens':len(ids),'total_tokens':len(tokens)+len(ids)},'token_ids':ids}
        if role in ('prefill','combined'):
            hint=request.headers.get('x-pd-output-tokens',str(count) if role=='combined' else '1')
            try:hint=int(hint)
            except ValueError:raise AdmissionError(400,'invalid output length hint')
            if not 1<=hint<=512:raise AdmissionError(400,'invalid output length hint')
            state=c.submit(rid,tokens,fp,hint);engine.record(dict(kind='http_prefill',id=rid,fingerprint=fp))
            try:
                while not state.ready.is_set():
                    if await request.is_disconnected():raise AdmissionError(499,'client disconnected')
                    await asyncio.sleep(.005)
                if state.error:raise AdmissionError(503,state.error)
                if state.cancelled.is_set():raise AdmissionError(408,'request expired')
            except BaseException:
                c.cancel(rid,state)
                raise
            # This is the internal prefill handshake: generation occurs on ATOM.
            if role=='prefill':return response([], 'stop')
        state=c.authorize_decode(rid,fp,count,body.get('ignore_eos',False),eos_ids)
        engine.record(dict(kind='http_decode',id=rid,fingerprint=fp,output_tokens=count))
        async def events():
            done=False
            try:
                while True:
                    if await request.is_disconnected():break
                    try:kind,value=state.output.get_nowait()
                    except Empty:await asyncio.sleep(.002);continue
                    if kind=='error':raise AdmissionError(503,value)
                    if kind=='finish' and value=='cancelled':raise AdmissionError(408,'request expired or cancelled')
                    yield kind,value
                    if kind=='finish':done=True;break
            finally:
                if not done:c.cancel(rid,state)
        if body.get('stream',False):
            async def stream():
                ids=[];sent=''
                try:
                    async for kind,value in events():
                        if kind=='token':ids.append(value)
                        text=tokenizer.decode(ids,skip_special_tokens=True)
                        # Delay incomplete UTF-8 tokens; do not emit replacement characters.
                        if kind=='token':text=text.rstrip('\ufffd')
                        if not text.startswith(sent):raise RuntimeError('non-prefix tokenizer decoding')
                        delta=text[len(sent):];sent=text
                        choice={'index':0,'finish_reason':value if kind=='finish' else None}
                        choice.update({'delta':{'content':delta}} if chat else {'text':delta,'logprobs':None})
                        item={'id':rid,'object':'chat.completion.chunk' if chat else 'text_completion','created':created,'model':model,'choices':[choice]}
                        if kind=='token':item['token_ids']=[value]
                        yield 'data: '+json.dumps(item)+'\n\n'
                    if body.get('stream_options',{}).get('include_usage',False):
                        item=response(ids,'stop');item['choices']=[];item.pop('token_ids',None)
                        item['object']='chat.completion.chunk' if chat else 'text_completion'
                        yield 'data: '+json.dumps(item)+'\n\n'
                    yield 'data: [DONE]\n\n'
                except Exception as exc:
                    yield 'data: '+json.dumps({'error':{'message':str(exc)}})+'\n\n'
                finally:
                    c.cancel(rid,state)
            return StreamingResponse(stream(),media_type='text/event-stream')
        ids=[];reason='length'
        try:
            async for kind,value in events():
                if kind=='token':ids.append(value)
                else:reason=value
        finally:
            c.cancel(rid,state)
        return response(ids,reason)
    return app


def main():
    p=argparse.ArgumentParser()
    for key in ('model','fxb','artifact','tables','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--served-model',default='koni')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--atom-device',type=int,default=0);p.add_argument('--rngd-device',default='npu:0')
    p.add_argument('--capacity',type=int,default=64);p.add_argument('--policy',choices=['fifo','length'],default='fifo')
    p.add_argument('--batch-wait-ms',type=float,default=10);p.add_argument('--request-ttl',type=float,default=120)
    p.add_argument('--combined-http-port',type=int,default=18603)
    p.add_argument('--prefill-http-port',type=int,default=18601);p.add_argument('--decode-http-port',type=int,default=18602)
    p.add_argument('--prefill-port',type=int,default=18621);p.add_argument('--receive-port',type=int,default=18622)
    p.add_argument('--prefill-concurrency',type=int,default=4)
    p.add_argument('--furiosa-python',default=os.path.expanduser('~/venvs/furiosa-2026.4/bin/python'))
    a=p.parse_args();engine=ServiceEngine(a);tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    generation_path=Path(a.model)/'generation_config.json'
    configured=json.loads(generation_path.read_text()).get('eos_token_id',[]) if generation_path.exists() else []
    configured=configured if isinstance(configured,list) else [configured]
    eos_ids=sorted({x for x in [*configured,tokenizer.eos_token_id] if type(x) is int})
    engine.record(dict(kind='eos_policy',stop_token_ids=eos_ids,policy='model EOS plus tokenizer end-of-turn; ignore_eos disables both'))
    engine.start();servers=[];threads=[];stop=threading.Event()
    for role,port in [('prefill',a.prefill_http_port),('decode',a.decode_http_port),('combined',a.combined_http_port)]:
        server=uvicorn.Server(uvicorn.Config(create_app(engine,tokenizer,role,a.served_model,eos_ids),host='127.0.0.1',port=port,log_level='warning'))
        t=threading.Thread(target=server.run,daemon=True);t.start();servers.append(server);threads.append(t)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    try:
        while not stop.wait(.2):
            if engine.coordinator.stopped or any(not t.is_alive() for t in threads):break
    finally:
        for server in servers:server.should_exit=True
        engine.close()
        for t in threads:t.join(timeout=5)

if __name__=='__main__':main()
