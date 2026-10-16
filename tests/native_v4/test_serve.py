import asyncio
import threading
import unittest
import httpx
from rngd_pd.native_v4.coordinator import Coordinator
from rngd_pd.native_v4.serve import create_app

class Tokenizer:
    eos_token_id=9
    def encode(self,text,add_special_tokens=True):return [1,2]
    def decode(self,ids,skip_special_tokens=True):return ''.join(chr(97+i) for i in ids)
    def apply_chat_template(self,*args,**kwargs):return [1,2,3]
class Engine:
    def __init__(self):
        self.coordinator=Coordinator(2,batch_wait=0);self.ready=threading.Event();self.ready.set()
    def record(self,event):
        c=self.coordinator
        if event['kind']=='http_prefill':c.prefilled(c.prefill_batch())
        if event['kind']=='http_decode':
            for job in c.take_ready():
                for i in range(job['max_new_tokens']):c.token(job,i)
                c.finish(job,'length')
class ServeTest(unittest.TestCase):
    def test_two_stage_contract_and_stream(self):
        async def run():
            e=Engine();t=Tokenizer()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(e,t,'prefill','koni')),base_url='http://p') as p, httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(e,t,'decode','koni')),base_url='http://d') as d:
                body=dict(model='koni',prompt=[1,2],max_tokens=3)
                self.assertEqual((await p.post('/v1/completions',json=body)).status_code,400)
                h={'x-request-id':'test'}
                self.assertEqual((await p.post('/v1/completions',json=body,headers=h)).status_code,200)
                self.assertEqual((await p.post('/v1/completions',json=body,headers=h)).status_code,409)
                self.assertEqual((await d.post('/v1/completions',json={**body,'prompt':[3,4]},headers=h)).status_code,409)
                r=await d.post('/v1/completions',json=body,headers=h)
                self.assertEqual(r.json()['token_ids'],[0,1,2]);self.assertEqual(e.coordinator.metrics()['active'],0)
                await p.post('/v1/completions',json=body,headers=h)
                r=await d.post('/v1/completions',json={**body,'stream':True},headers=h)
                self.assertIn('data: [DONE]',r.text);self.assertIn('"token_ids": [2]',r.text)
                self.assertEqual((await p.post('/v1/completions',json={**body,'temperature':1},headers=h)).status_code,400)
        asyncio.run(run())

class CombinedServeTest(unittest.TestCase):
    def test_single_request_performs_both_stages(self):
        async def run():
            e=Engine()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(e,Tokenizer(),'combined','koni')),base_url='http://local') as c:
                r=await c.post('/v1/completions',json=dict(model='koni',prompt=[1,2],max_tokens=3))
                self.assertEqual(r.status_code,200)
                self.assertEqual(r.json()['token_ids'],[0,1,2])
                self.assertTrue(r.json()['id'].startswith('cmpl-'))
                self.assertEqual(e.coordinator.metrics()['active'],0)
        asyncio.run(run())

class ChatUsageTest(unittest.TestCase):
    def test_chat_tokenizer_contract_and_usage_without_duplicate_tokens(self):
        class ModernTokenizer(Tokenizer):
            def apply_chat_template(self,*args,**kw):
                return [1,2,3] if kw.get('return_dict') is False else {'input_ids':[1,2,3]}
        async def run():
            e=Engine()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(e,ModernTokenizer(),'combined','koni')),base_url='http://local') as c:
                r=await c.post('/v1/chat/completions',json=dict(model='koni',messages=[{'role':'user','content':'hi'}],max_tokens=3,stream=True,stream_options={'include_usage':True}))
                self.assertEqual(r.status_code,200)
                import json
                items=[json.loads(x[6:]) for x in r.text.splitlines() if x.startswith('data: ') and x!='data: [DONE]']
                self.assertEqual([t for x in items for t in x.get('token_ids',[])],[0,1,2])
                self.assertEqual(items[-1]['usage']['completion_tokens'],3)
                self.assertEqual(items[-1]['choices'],[])
        asyncio.run(run())
