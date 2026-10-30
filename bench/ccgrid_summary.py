#!/usr/bin/env python3
"""Derive CCGrid medians/IQR from a normalized, raw-linked sample collection.

python bench/ccgrid_summary.py collection.json --out new-summary.json

Collection: {"samples": [{"trial_id": str, "experiment": "E6", "arm": str,
"cache_state": str, "status": "completed", "validity": "valid",
"tokens_equal_reference": true, "metrics": {"metric_s": number},
"raw_path": "relative/sample.json", "raw_sha256": "..."}, ...]}.
Additional grouping fields: mode, graph_state, base_variant, target_variant,
fixture. A collection must contain one frozen source/input/environment plan;
use a separate collection for a changed plan or corrected measurement attempt.
Input references are byte-verified, original samples retained in the output.
Only completed/ok, valid, reference-equal samples enter statistics. A failed
sample makes its group ineligible for a headline even if five others pass.
Missing metrics are reported, never imputed. Quartiles use linear inclusive
interpolation (statistics.quantiles(method='inclusive')). No units are changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


GROUP_FIELDS = ('experiment','arm','mode','cache_state','graph_state',
                'base_variant','target_variant','fixture')


def summarize(samples):
    grouped, seen = {}, set()
    for sample in samples:
        trial = sample['trial_id']
        if trial in seen:
            raise ValueError('duplicate trial_id: ' + trial)
        seen.add(trial)
        for value in sample['metrics'].values():
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError('metric must be finite nonnegative number')
        key = tuple(sample.get(k) for k in GROUP_FIELDS)
        grouped.setdefault(key, []).append(sample)
    result = []
    for key, rows in grouped.items():
        valid = [r for r in rows if r['status'] in ('ok','completed')
                 and r['validity']=='valid' and r['tokens_equal_reference'] is True]
        names = sorted({name for r in rows for name in r['metrics']})
        metrics = {}
        for name in names:
            values = [r['metrics'][name] for r in valid if name in r['metrics']]
            if len(values)>=2:
                q1, _, q3 = statistics.quantiles(values,n=4,method='inclusive')
            else:
                q1 = q3 = values[0] if values else None
            metrics[name] = dict(n=len(values),values=values,
                median=statistics.median(values) if values else None,
                q1=q1,q3=q3,iqr=q3-q1 if values else None,
                min=min(values) if values else None,max=max(values) if values else None)
        excluded = [dict(trial_id=r['trial_id'],status=r['status'],validity=r['validity'],
                        tokens_equal_reference=r['tokens_equal_reference'])
                    for r in rows if r not in valid]
        result.append(dict(group=dict(zip(GROUP_FIELDS,key)),n_total=len(rows),
            n_valid=len(valid),excluded=excluded,metrics=metrics,
            headline_eligible=(len(valid)>=5 and not excluded and bool(metrics)
                               and all(m['n']==len(valid) for m in metrics.values()))))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    raw=args.collection.read_bytes()
    document=json.loads(raw)
    for sample in document['samples']:
        path=(args.collection.parent/sample['raw_path']).resolve()
        if not path.is_relative_to(args.collection.parent.resolve()):
            raise ValueError('raw_path must stay inside collection directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sample['raw_sha256']:
            raise ValueError('raw file SHA256 differs: '+sample['raw_path'])
    output=dict(schema='ccgrid-npu-summary-v1',collection_sha256=hashlib.sha256(raw).hexdigest(),
        quantile_method='inclusive linear interpolation',minimum_n=5,
        groups=summarize(document['samples']),collection=document)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
