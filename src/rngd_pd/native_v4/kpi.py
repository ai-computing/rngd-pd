"""Useful model FLOPs and allocated-board energy over identical PD windows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def useful_flops(prompt, output):
    h, m, nh, nkv, hd, layers, vocab = 4096, 14336, 32, 8, 128, 32, 128256
    dense = layers*(2*h*(nh*hd+2*nkv*hd)+2*h*nh*hd+6*h*m)
    lm = 2*h*vocab
    # Only the exported P-1 prefix is useful to the decoder; padding, any
    # redundant native work, and compilation are excluded from the numerator.
    pre = (prompt-1)*dense + layers*4*nh*hd*(prompt-1)*prompt/2
    dec = output*(dense+lm) + layers*4*nh*hd*(output*prompt+output*(output-1)/2)
    return pre, dec


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def fp16_atom_peak(root):
    """Only use the published CA22 FP16 peak for explicitly recorded FP16 builds."""
    manifest = root/'atom-build-manifest.json'
    provenance = root/'provenance.json'
    compiled = read(root/'compiled-artifact.jsonl')
    if not manifest.exists() or not provenance.exists() or len(compiled) != 1:
        return None
    build = json.loads(manifest.read_text())
    run = json.loads(provenance.read_text())
    if (build.get('weight_dtype') == 'FP16' and run.get('atom_weight_dtype') == 'FP16'
            and compiled[0].get('npu') == 'RBLN-CA22' and compiled[0].get('num_devices') == 1):
        return 32.
    return None


def integrate(rows, start, end):
    points = sorted(((r['start']+r['end'])/2, r['watts']) for r in rows if 'watts' in r)
    if len(points) < 2 or points[0][0] > start or points[-1][0] < end:
        return None
    energy = 0.
    for (t, w), (u, v) in zip(points, points[1:]):
        lo, hi = max(start, t), min(end, u)
        if hi > lo:
            if u-t > 3:
                return None  # Reject poorly sampled windows, do not fill sensor gaps.
            energy += (hi-lo)*(2*w+(v-w)*(lo+hi-2*t)/(u-t))/2
    return energy


def evaluate(root):
    status_path = root/'status.json'
    if not status_path.exists() or json.loads(status_path.read_text()).get('status') != 'completed':
        return dict(status='invalid', reason='run did not complete successfully')
    jobs = {j['id']: j for j in json.loads((root/'cases.json').read_text())}
    results = read(root/'results.jsonl')
    windows = read(root/'windows.jsonl')
    if len(windows) != 1:
        return dict(status='invalid', reason='missing or ambiguous completed window')
    window = windows[0]
    if (len(results) != len(jobs) or {r['id'] for r in results} != set(jobs) or
            any(r['status'] != 'ok' or len(r['token_ids']) != jobs[r['id']]['max_new_tokens'] for r in results)):
        return dict(status='invalid', reason='incomplete, duplicated, or failed requests')
    if any('expected_tokens' in jobs[r['id']] and r['token_ids'] != jobs[r['id']]['expected_tokens']
           for r in results):
        return dict(status='invalid', reason='reference output mismatch')
    seconds = window['end']-window['start']
    if seconds <= 0:
        return dict(status='invalid', reason='nonpositive measurement duration')
    er = integrate(read(root/'power-rngd.jsonl'), window['start'], window['end'])
    config = json.loads((root/'config.json').read_text())
    devices = config.get('atom_devices') or [config.get('atom_device', 0)]
    atom_energy = {device: integrate(read(root/(f'power-atom-{device}.jsonl' if len(devices)>1
                                              else 'power-atom.jsonl')), window['start'], window['end'])
                   for device in devices}
    ea = sum(atom_energy.values()) if all(e is not None for e in atom_energy.values()) else None
    atom_flops = dict.fromkeys(devices, 0.)
    fp = fd = 0.
    tpot = []
    intervals = []
    for result in results:
        job = jobs[result['id']]
        pre, dec = useful_flops(len(job['tokens']), job['max_new_tokens'])
        fp += pre;fd += dec
        device = result.get('atom_device', devices[0])
        if device not in atom_flops:
            return dict(status='invalid', reason='result assigned to an unallocated decode card')
        atom_flops[device] += dec
        times = result['token_ready_seconds']
        if len(times) > 1:
            tpot.append((times[-1]-times[0])/(len(times)-1))
            intervals.extend(np.diff(times).tolist())
    p99 = float(np.quantile(tpot, .99)) if tpot else None
    atom_peak = fp16_atom_peak(root)
    system_mfu = 100*(fp+fd)/seconds/((256+len(devices)*atom_peak)*1e12) if atom_peak else None
    return dict(status='ok' if er is not None and ea is not None else 'power_unverified',
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requests=len(results), seconds=seconds, requests_s=len(results)/seconds,
        useful_prefill_flops=fp, useful_decode_flops=fd, system_TFLOPS=(fp+fd)/seconds/1e12,
        rngd_joules=er, atom_joules=ea,
        atom_boards=[dict(device=d, joules=atom_energy[d], useful_flops=atom_flops[d],
                         mean_W=atom_energy[d]/seconds if atom_energy[d] is not None else None,
                         GFLOPS_W=atom_flops[d]/atom_energy[d]/1e9 if atom_energy[d] else None)
                     for d in devices],
        rngd_mean_W=er/seconds if er is not None else None,
        atom_mean_W=ea/seconds if ea is not None else None,
        rngd_GFLOPS_W=fp/er/1e9 if er else None,
        atom_GFLOPS_W=fd/ea/1e9 if ea else None,
        system_GFLOPS_W=(fp+fd)/(er+ea)/1e9 if er and ea else None,
        rngd_reserved_MFU_pct=100*fp/seconds/256e12,
        atom_MFU_pct=100*fd/seconds/(len(devices)*atom_peak*1e12) if atom_peak else None,
        system_MFU_pct=system_mfu,
        system_MFU_precision='RNGD BF16 + ATOM FP16' if atom_peak else 'BF16 peak unverified',
        system_MFU_method='model-FLOPs estimate / allocated precision-specific published peak',
        atom_peak_TFLOPS_per_card=atom_peak,
        atom_peak_source='https://rebellions.ai/rebellions-product/rbln-ca22/' if atom_peak else None,
        atom_per_card_BF16_peak_limit_for_20pct_TFLOPS=((fp+fd)/seconds/.20/1e12-256)/len(devices),
        peak_limit_definition='Conditional bound only: system MFU >=20% iff identical ATOM cards each have BF16 peak at or below this value; not a verified hardware peak',
        system_MFU_upper_bound_pct=100*(fp+fd)/seconds/256e12,
        system_MFU_reason=('FP16 build manifest and CA22 compiled target verified; mixed-precision estimate, not all-BF16 MFU'
                           if atom_peak else 'ATOM CA22 BF16 theoretical peak not independently established; FP16 peak is not substituted'),
        tpot_p99_ms=p99*1000 if p99 is not None else None,
        itl_p99_ms=float(np.quantile(intervals, .99))*1000 if intervals else None,
        itl_max_ms=max(intervals)*1000 if intervals else None,
        slo_case_pass=p99<=.150 if p99 is not None else None,
        tpot_definition='p99 across per-request mean (last token time-first token time)/(O-1); queue/TTFT separate',
        ttft_p99_ms=float(np.quantile([r['ttft_seconds'] for r in results], .99))*1000,
        power_scope='allocated RNGD and ATOM boards including idle time; host CPU/DRAM excluded',
        measurement='model FLOPs estimated from architecture; latency and board sensor power measured',
        workload='synthetic fixed-shape when prompts repeated; not a public-trace SLO achievement',
        window=window)


def main():
    p = argparse.ArgumentParser();p.add_argument('root', type=Path);a = p.parse_args()
    result = evaluate(a.root)
    (a.root/'kpi.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'window'}, indent=2))


if __name__ == '__main__':
    main()
