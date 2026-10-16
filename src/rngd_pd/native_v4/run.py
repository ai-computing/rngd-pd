"""Reproducible, bounded native RNGD prefill -> ATOM decode runner.

Each output directory is new. Old artifacts/results are never overwritten.
This local experimental transport is deliberately bound to loopback only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import zmq
import numpy as np
from .atom import AtomDecoder
from .kv import KVReceiver, host_layout
from .wire import parse, handshake, pull, control, ProtocolError
from .power import PowerSampler
from threading import Lock
from .affinity import pci_cpus


def main():
    p = argparse.ArgumentParser()
    for name in ['model', 'fxb', 'artifact', 'tables', 'cases', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--furiosa-python', default='/home/gspark/venvs/furiosa-2026.4/bin/python')
    p.add_argument('--rngd-device', default='npu:0')
    p.add_argument('--atom-device', type=int, default=0)
    p.add_argument('--atom-devices', help='comma-separated independent decode cards; requires --pipeline')
    p.add_argument('--batch', type=int, choices=[4, 8, 16, 24], default=4)
    p.add_argument('--active-batch', type=int, default=1)
    p.add_argument('--batch-policy', choices=['fifo', 'length'], default='fifo')
    p.add_argument('--lookahead-batches', type=int, default=4)
    p.add_argument('--prefill-port', type=int, default=18421)
    p.add_argument('--receive-port', type=int, default=18422)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--native-batch', action='store_true')
    p.add_argument('--prefill-concurrency', type=int, default=4)
    p.add_argument('--pipeline', action='store_true')
    p.add_argument('--power', action='store_true')
    p.add_argument('--numa-local', action='store_true')
    p.add_argument('--native-overlap', action='store_true')
    p.add_argument('--huge-host', action='store_true')
    p.add_argument('--resident-kv', action='store_true')
    p.add_argument('--array-metadata', action='store_true')
    p.add_argument('--bulk-admit', action='store_true')
    p.add_argument('--direct-admit', action='store_true')
    p.add_argument('--token-major-host', action='store_true')
    p.add_argument('--skip-resident-clear', action='store_true')
    p.add_argument('--kv-converter', type=Path, help='optional compiled CPU DLFloat16 converter')
    a = p.parse_args()
    a.atom_devices = [int(x) for x in a.atom_devices.split(',')] if a.atom_devices else [a.atom_device]
    if not a.atom_devices or len(set(a.atom_devices)) != len(a.atom_devices) or min(a.atom_devices) < 0:
        p.error('decode card IDs must be unique nonnegative integers')
    if len(a.atom_devices) > 1 and not a.pipeline:
        p.error('multiple decode cards require --pipeline')
    a.atom_device = a.atom_devices[0]
    if a.direct_admit and not (a.resident_kv and a.bulk_admit):
        p.error("--direct-admit requires --resident-kv --bulk-admit")
    if a.bulk_admit and not a.resident_kv:
        p.error('--bulk-admit requires --resident-kv')
    if a.token_major_host and not a.resident_kv:
        p.error('--token-major-host requires --resident-kv')
    if a.skip_resident_clear and not a.resident_kv:
        p.error('--skip-resident-clear requires --resident-kv')
    if a.kv_converter and not a.resident_kv:
        p.error('--kv-converter requires --resident-kv')
    converter = None
    if a.kv_converter:
        from .native_conversion import NativeConverter
        converter = NativeConverter(a.kv_converter)
    affinity = dict(parent_original=sorted(os.sched_getaffinity(0)))
    native_affinity_args = []
    if a.numa_local:
        atom_info = json.loads(subprocess.check_output(['rbln-smi', '--json']))
        atom_infos = {d['npu']: d for d in atom_info['devices']}
        atom_info = atom_infos[a.atom_device]
        rngd_info = json.loads(subprocess.check_output(['furiosa-smi', 'info', '--format', 'json']))
        rngd_info = next(d for d in rngd_info if d['dev_name'] == 'npu'+a.rngd_device.split(':')[1])
        atom_node, atom_cpus = pci_cpus(atom_info['pci']['bus_id'], affinity['parent_original'])
        if any(pci_cpus(atom_infos[d]['pci']['bus_id'], affinity['parent_original'])[0] != atom_node
               for d in a.atom_devices):
            raise ValueError('multi-card NUMA placement currently requires decode cards on the same NUMA node')
        rngd_node, rngd_cpus = pci_cpus(rngd_info['pci_bdf'], affinity['parent_original'])
        os.sched_setaffinity(0, atom_cpus)
        native_affinity_args = ['--cpu-affinity', ','.join(map(str, rngd_cpus))]
        affinity.update(atom_node=atom_node, atom_cpus=atom_cpus, rngd_node=rngd_node, rngd_cpus=rngd_cpus,
                        memory_policy='default first-touch; no mbind or global NUMA policy change')
    jobs = json.loads(a.cases.read_text())
    if not jobs or not 1 <= a.active_batch <= a.batch:
        raise ValueError('empty jobs or invalid active batch')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('request IDs must be unique')
    for j in jobs:
        if not isinstance(j['id'], str) or not j['id'].isascii() or not 1 <= len(j['id']) <= 128:
            raise ValueError('request IDs must be 1..128 ASCII characters')
        if (not 2 <= len(j['tokens']) or not 1 <= j['max_new_tokens'] or
                len(j['tokens'])+j['max_new_tokens'] > 512 or
                any(type(t) is not int or not 0 <= t < 128256 for t in j['tokens'])):
            raise ValueError('invalid token IDs or context lengths')
    from .scheduling import plan_batches, padding_stats
    planned_batches = list(plan_batches(jobs, a.active_batch, a.batch_policy, a.lookahead_batches))
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'batch-plan.json').write_text(json.dumps(dict(policy=a.batch_policy,
        eligibility='offline replay: all input jobs available at start',
        batches=[[j['id'] for j in b] for b in planned_batches],
        **padding_stats(planned_batches, a.batch)), indent=2))
    (a.out/'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(a).items()}, indent=2))
    (a.out/'cases.json').write_text(json.dumps(jobs, indent=2))
    package = Path(__file__).resolve().parent
    shutil.copytree(package, a.out/'source/native_v4', ignore=shutil.ignore_patterns('__pycache__'))
    def git(args):
        return subprocess.check_output(['git', '-C', str(package), *args], text=True).strip()
    build_manifest_path = a.artifact.parent/'manifest.json'
    build_manifest = json.loads(build_manifest_path.read_text()) if build_manifest_path.exists() else {}
    atom_weight_dtype = build_manifest.get('weight_dtype', 'unverified (build manifest absent)')
    provenance = dict(python=sys.version, argv=sys.argv, affinity=affinity, repo_sha=git(['rev-parse', 'HEAD']),
        repo_dirty=bool(git(['status', '--porcelain'])),
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pyzmq', 'rebel-compiler']},
        source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in package.iterdir() if f.suffix in ('.py', '.c')},
        converter_sha256=hashlib.sha256(a.kv_converter.read_bytes()).hexdigest() if a.kv_converter else None,
        artifacts={name: dict(path=str(path.resolve()), size=path.stat().st_size,
                             mtime_ns=path.stat().st_mtime_ns)
                   for name, path in [('fxb', a.fxb), ('atom_decode', a.artifact), ('tables', a.tables)]},
        timing_scope='model loading excluded; batch filling and host conversion included; padding lanes excluded from useful tokens',
        precision=(('RNGD BF16 KV -> ATOM SDK DLFloat16 resident cache; compiled weights '
                    if a.resident_kv else 'RNGD BF16 KV -> ATOM float32 host inputs; compiled weights ')
                   + atom_weight_dtype),
        atom_weight_dtype=atom_weight_dtype,
        transport='host_staged_dma', device_resident_atom_kv=a.resident_kv,
        allocated_atom_devices=a.atom_devices)
    (a.out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    ctx = zmq.Context()
    rx, tx = ctx.socket(zmq.PULL), ctx.socket(zmq.PUSH)
    rx.setsockopt(zmq.MAXMSGSIZE, 64*512*2048)
    rx.setsockopt(zmq.RCVHWM, 32)
    tx.setsockopt(zmq.SNDTIMEO, 5000)
    rx.bind(f'tcp://127.0.0.1:{a.receive_port}')
    tx.connect(f'tcp://127.0.0.1:{a.prefill_port}')
    child = None
    pending = set()
    pool = ThreadPoolExecutor(max_workers=len(a.atom_devices)) if a.pipeline else None
    flight = None
    sampler = None
    decoder = None
    decode_states = []
    log = (a.out/'native.log').open('w')
    record_lock = Lock()
    def record(name, obj):
        with record_lock, (a.out/name).open('a') as f:
            f.write(json.dumps(obj)+'\n')
    try:
        import rebel
        record('compiled-artifact.jsonl', rebel.RBLNCompiledModel.inspect(str(a.artifact)))
        for filename in ('manifest.json', 'completed.json'):
            build_record = a.artifact.parent/filename
            if build_record.exists():
                shutil.copyfile(build_record, a.out/('atom-build-'+filename))
        decoder_class = AtomDecoder
        if a.resident_kv:
            from .atom_resident import AtomResidentDecoder
            decoder_class = AtomResidentDecoder
        for device in a.atom_devices:
            decoder = decoder_class(a.artifact, a.tables, device, a.batch, a.huge_host,
                                    **({'bulk_admit': a.bulk_admit, 'token_major_host': a.token_major_host,
                                        'skip_clear': a.skip_resident_clear, 'direct_admit': a.direct_admit}
                                       if a.resident_kv else {}))
            state = dict(device=device, decoder=decoder, flight=None, banks=[], bank_futures=[])
            decode_states.append(state)
            banks = [(decoder.keys, decoder.values)]
            if a.pipeline:
                banks.append(decoder.new_buffers())
            state.update(banks=banks, bank_futures=[None]*len(banks))
            record('host-buffers.jsonl', dict(device=device,
                allocator='rebel_aligned_packed' if a.direct_admit else ('rebel_huge_host' if a.huge_host else 'numpy'),
                bank_count=len(banks), bytes_per_bank=sum(v.nbytes for v in banks[0]),
                alignment_remainders=[[int(v.ctypes.data % (2*1024*1024)) for v in bank] for bank in banks]))
        env = os.environ.copy()
        env['RUST_LOG'] = 'furiosa_generator::connector=debug'
        child = subprocess.Popen([a.furiosa_python, '-u', '-m',
            'rngd_pd.native_v4.native_worker', '--model', str(a.model), '--fxb', str(a.fxb),
            '--device', a.rngd_device, '--port', str(a.prefill_port),
            '--concurrency', str(a.prefill_concurrency),
            '--peer', f'127.0.0.1:{a.receive_port}', '--events', str(a.out/'native-events.jsonl'),
            *(['--overlap'] if a.native_overlap else []),
            *native_affinity_args],
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        # Exclude model loading from request latency; it remains in native.log.
        deadline = time.monotonic()+a.timeout
        events = a.out/'native-events.jsonl'
        while not events.exists() or 'initialized' not in events.read_text():
            if child.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('native worker failed initialization; see native.log')
            time.sleep(.1)
        def decode_and_record(batch, receivers, metrics, buffers, active_decoder, device):
            decode_start = time.perf_counter()
            results, steps = active_decoder.decode(batch, a.diagnostics, buffers)
            decode_end = time.perf_counter()
            record('decode-steps.jsonl', dict(device=device, ids=[j['id'] for j in batch], steps=steps,
                                             start=decode_start, end=decode_end))
            expected = {j['id']: j.get('expected_tokens') for j in batch}
            mismatches = []
            for result in results:
                rid = result['id']; metric = metrics[rid]
                result.update(status='ok', atom_device=device, transport='host_staged_dma', compiled_batch=a.batch,
                              useful_batch=len(batch), stage_metrics=metric,
                              decode_start=decode_start,
                              ttft_seconds=decode_start-metric['submitted']+result['token_ready_seconds'][0],
                              e2e_seconds=decode_start-metric['submitted']+result['token_ready_seconds'][-1],
                              prefix_tokens_transferred=receivers[rid].n)
                if expected[rid] is not None:
                    result['expected_tokens_match'] = result['token_ids'] == expected[rid]
                    if not result['expected_tokens_match']:
                        result['status'] = 'wrong_output'
                        mismatches.append(rid)
                record('results.jsonl', result)
                print(json.dumps(result), flush=True)
            if mismatches:
                raise RuntimeError(f'Greedy reference mismatch: {mismatches}')
        if a.power:
            rngd_name = 'npu'+a.rngd_device.split(':')[1]
            sampler = PowerSampler(a.out, rngd_name,
                                   a.atom_devices if len(a.atom_devices) > 1 else a.atom_device)
            sampler.start()
        window_start = time.perf_counter()
        for batch_index, batch in enumerate(planned_batches):
            state = decode_states[batch_index % len(decode_states)]
            decoder = state['decoder']
            bank_index = (batch_index // len(decode_states)) % len(state['banks'])
            previous = state['bank_futures'][bank_index]
            if previous is not None:
                previous.result()  # Never clear a host bank still used by a decode task.
            buffers = state['banks'][bank_index]
            receivers, metrics = {}, {}
            for lane, j in enumerate(batch):
                metrics[j['id']] = dict(submitted=time.perf_counter(), convert_seconds=0., batches=0)
                decoder.clear(lane, buffers)
                receivers[j['id']] = KVReceiver(j['id'], len(j['tokens'])-1,
                                               *buffers, lane, converter=converter)
                pending.add(j['id'])
                if not a.native_batch:
                    child.stdin.write(json.dumps(j)+'\n')
            if a.native_batch:
                child.stdin.write(json.dumps(batch)+'\n')
            child.stdin.flush()
            ready = set()
            deadline = time.monotonic()+a.timeout
            while pending:
                if child.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('native KV transfer failed or timed out; see native.log')
                if not rx.poll(100):
                    continue
                frames = rx.recv_multipart(copy=False)
                m = parse(bytes(frames[0]), array_entries=a.array_metadata)
                rid = m.get('request_id')
                if rid not in pending:
                    raise ProtocolError(f'unexpected request {rid}: {m["kind"]}')
                if m['kind'] == 'ready':
                    if rid in ready:
                        raise ProtocolError('duplicate ready')
                    ready.add(rid)
                    metrics[rid]['prefill_ready'] = time.perf_counter()
                    tx.send(handshake(f'127.0.0.1:{a.receive_port}', host_layout()))
                    tx.send(pull(rid, list(range(receivers[rid].n))))
                elif m['kind'] == 'kv' and rid in ready:
                    start = time.perf_counter()
                    done = receivers[rid].accept(m, [f.buffer for f in frames[1:]])
                    metrics[rid]['convert_seconds'] += time.perf_counter()-start
                    metrics[rid]['batches'] += 1
                    if done:
                        metrics[rid]['kv_ready'] = time.perf_counter()
                        metrics[rid]['payload_bytes'] = receivers[rid].payload_bytes
                        metrics[rid]['payload_frames'] = receivers[rid].frame_count
                        metrics[rid]['kv_entries'] = receivers[rid].entry_count
                        tx.send(control(3, rid))
                        pending.remove(rid)
                else:
                    raise ProtocolError(f'unexpected connector event {m["kind"]}')
            if pool is not None:
                if state['flight'] is not None:
                    state['flight'].result()  # A card's resident cache has one owner.
                flight = pool.submit(decode_and_record, batch, receivers, metrics, buffers,
                                     decoder, state['device'])
                state['flight'] = state['bank_futures'][bank_index] = flight
            else:
                decode_and_record(batch, receivers, metrics, buffers, decoder, state['device'])
        for state in decode_states:
            if state['flight'] is not None:
                state['flight'].result()
        window_end = time.perf_counter()
        record('windows.jsonl', dict(start=window_start, end=window_end, requests=len(jobs),
                                     ids=[j['id'] for j in jobs], pipeline=a.pipeline,
                                     native_batch=a.native_batch))
        if sampler is not None:
            sampler.finish(window_end)
        child.stdin.close()
        if child.wait(timeout=30):
            raise RuntimeError('native worker exited with error')
        (a.out/'status.json').write_text(json.dumps(dict(status='completed', requests=len(jobs))))
    except BaseException as exc:
        for rid in pending:
            try:
                tx.send(control(4, rid), flags=zmq.DONTWAIT)
            except zmq.ZMQError:
                pass
        record('errors.jsonl', dict(error=repr(exc), traceback=traceback.format_exc(), pending=sorted(pending)))
        (a.out/'status.json').write_text(json.dumps(dict(status='failed', error=repr(exc))))
        raise
    finally:
        if sampler is not None:
            sampler.finish()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for state in decode_states:
            state['decoder'].close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        log.close()
        rx.close(linger=0); tx.close(linger=0); ctx.term()


if __name__ == '__main__':
    main()
