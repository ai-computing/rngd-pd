"""Isolated Furiosa 2026.4 FXB worker; JSONL jobs on stdin.

Run with the Furiosa venv, never with the ATOM Python environment.
"""
import argparse
import asyncio
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--fxb', required=True)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--port', type=int, default=18421)
    p.add_argument('--peer', default='127.0.0.1:18422')
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--logprobs', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--cpu-affinity')
    p.add_argument('--overlap', action='store_true')
    a = p.parse_args()
    if a.cpu_affinity:
        from .affinity import parse_cpu_list
        os.sched_setaffinity(0, parse_cpu_list(a.cpu_affinity))
    version = importlib.metadata.version('furiosa-llm')
    if version != '2026.4.0':
        raise RuntimeError(f'Experimental wire codec requires 2026.4.0; found {version}')
    from furiosa_llm import LLM, SamplingParams
    from furiosa_llm.metadata.config_types import SchedulerConfig, PrefixCacheConfig, SchedulingLoop
    from furiosa_llm.llm_engine import AsyncLLMEngine

    def emit(event):
        with a.events.open('a') as f:
            f.write(json.dumps(dict(epoch=time.time(), **event)) + '\n')

    llm = LLM(a.model, fxb=a.fxb, devices=a.device, max_model_len=512,
              scheduler_config=SchedulerConfig(profile_guided_decode_batching=False,
                  scheduling_loop_type=SchedulingLoop.OVERLAP if a.overlap else SchedulingLoop.EAGER,
                  max_concurrency=a.concurrency, prefix_cache_config=PrefixCacheConfig(enabled=False)),
              kv_transfer_config=None if a.baseline else dict(kv_connector='TcpConnector',
                  host='127.0.0.1', port=a.port, transfer_timeout_ms=30000))
    emit(dict(kind='initialized', sdk=version, device=a.device, baseline=a.baseline,
              cpu_affinity=sorted(os.sched_getaffinity(0))))
    engine = AsyncLLMEngine.from_llm(llm)
    async def generate(job):
            rid = job['id']
            ids, count = job['tokens'], job['max_new_tokens']
            if not 2 <= len(ids) or not 1 <= count or len(ids)+count > 512:
                raise ValueError('Invalid prompt/output lengths')
            emit(dict(kind='start', id=rid, prompt_tokens=len(ids), output_tokens=count))
            start = time.perf_counter()
            params = SamplingParams(temperature=0, max_tokens=count if a.baseline else 1,
                ignore_eos=True, logprobs=a.logprobs or None,
                extra_args=None if a.baseline else {'kv_transfer_params': {
                    'remote_decoder': a.peer, 'id': rid}})
            try:
                result = None
                async for item in engine.generate({'prompt_token_ids': ids}, params, rid):
                    result = item
                if result is None:
                    raise RuntimeError('native worker returned no result')
                emit(dict(kind='result', id=rid, token_ids=result.outputs[0].token_ids,
                          text=result.outputs[0].text, seconds=time.perf_counter()-start,
                          logprobs=[{str(k): dict(logprob=v.logprob, rank=v.rank)
                                     for k, v in step.items()} for step in result.outputs[0].logprobs]
                          if result.outputs[0].logprobs else None))
            except Exception as exc:
                emit(dict(kind='error', id=rid, error=repr(exc)))
                raise
    async def group(jobs):
        await asyncio.gather(*(generate(job) for job in jobs))
    try:
        for line in sys.stdin:
            jobs = json.loads(line)
            asyncio.run(group(jobs if isinstance(jobs, list) else [jobs]))
    finally:
        llm.engine.shutdown()


if __name__ == '__main__':
    main()
