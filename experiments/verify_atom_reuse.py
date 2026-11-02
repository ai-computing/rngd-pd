#!/usr/bin/env python3
"""Offline ATOM reuse evidence checks. No SDK or accelerator is used.

Checks exact exported bytes, pinned source blobs, one-card Pod identity,
separate smoke/main matrices, real-call counters and independently recomputed
clocks/statistics. Does not reproduce native execution or measure DMA bytes.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import tarfile
import io

SOURCE='fec2d370cc35f66dcbbf6897bc1d41fed1b48850'
GRAPH='02b4eebd765c1acd3412517ea3804fe0ba38ddf2aa020a88db853245f2908ebe'
TARGET='sha256:4784f48f923a9dbceb48db7b231f120bdffc784cafb85a9583b945059d0b44c3'
TOKENS=[[220,16,13,16,13,16,13,16]]
VERSIONS=dict(torch='2.11.0+cpu',rebel='0.11.0',compiler='0.11.0')
METRICS=('transition_s','trigger_to_first_token_s','admission_s','parse_s','create_runtime_s','prompt_ttft_s')


def need(ok,message):
    if not ok:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(path.read_bytes())


def stat(values):
    q1,_,q3=statistics.quantiles(values,method='inclusive') if len(values)>1 else [values[0]]*3
    return dict(n=len(values),median=statistics.median(values),q1=q1,q3=q3,iqr=q3-q1,values=values)


def native_generation(row):
    need(row['tokens']==TOKENS,'not same-target fresh tokens')
    need(row['run_attempts']==row['run_completed']==len(row['run_seconds'])==15,'native run count')
    need(all(type(t) in (float,int) and math.isfinite(t) and t>0 for t in row['run_seconds']),'native run clocks')
    need(row['native_runtime_type']=='rebel.sync_runtime.SyncRuntime'
         and row['compiled_type']=='rebel.compiled_model.RBLNCompiledModel','native public types')
    need(type(row['generate_call_ns']) is int and row['generate_return_ns']>row['generate_call_ns'],'generation clock')
    duration=(row['generate_return_ns']-row['generate_call_ns'])/1e9
    need(sum(row['run_seconds'][:8])<=row['prompt_ttft_s']+1e-6
         and row['prompt_ttft_s']<=duration+1e-6 and sum(row['run_seconds'])<=duration+1e-6,'native calls exceed generation/TTFT clock')


def verify_record(r,phase):
    pairs=1 if phase=='smoke' else 5
    need(r['status']=='completed' and r['validity']=='valid' and r['native_evidence'] is True
         and r['owners_closed'] is True and r['npu_requested']==1,'native run incomplete')
    need(r['config']['phase']==phase and r['config']['pairs']==pairs,'phase/n mismatch')
    need(r['config']['source_commit']==SOURCE and r['config']['graph_sha256']==GRAPH,'source/graph changed')
    need(r['versions']==VERSIONS and r['compile_attempts']==[] and r['vendor_internal_apis_used']==[],'SDK/compiler gate')
    a=r['assignment'];need(len(a['allocated'])==len(a['exposed'])==1 and a['allocated']==a['exposed'],'broad device exposure')
    need(a['allocated']==['/dev/rbln'+str(a['physical_id'])] and a['sdk_id']==0,'device mapping')
    native_generation(r['fresh_reference']);need(r['fresh_reference']['closed_before_matrix'] is True,'fresh still live')
    order=[(i,a) for i in range(pairs) for a in (('full','reuse') if i%2==0 else ('reuse','full'))]
    need([(t['pair'],t['arm']) for t in r['trials']]==order,'incomplete/reordered matrix')
    previous=r['fresh_reference']['generate_return_ns']
    for t in r['trials']:
        reused=t['arm']=='reuse'
        need(t['status']=='completed' and t['manifest']==TARGET and t['transition_mode']=='reload','target/mode')
        native_generation(t['base']);native_generation(t)
        need(previous<t['base']['generate_call_ns']<t['base']['generate_return_ns']<=t['trigger_ns']
             <t['transition_return_ns']<=t['generate_call_ns']<t['generate_return_ns'],'overlapping/invalid clocks')
        previous=t['generate_return_ns']
        need(t['parse_calls']==int(not reused) and t['create_calls']==1 and t['compile_attempts']==0,'parse/create gate')
        need(t['compiled_reused'] is reused and t['parsed_before'] is reused
             and t['host_object_retained'] is reused and t['old_runtime_closed'] is True,'object lifetime')
        need(all(type(t[k]) in (float,int) and math.isfinite(t[k]) and t[k]>=0 for k in METRICS),'invalid metric')
        need(math.isclose(t['transition_s'],(t['transition_return_ns']-t['trigger_ns'])/1e9,abs_tol=1e-9),'transition clock formula')
        endpoint=(t['generate_call_ns']-t['trigger_ns'])/1e9+t['prompt_ttft_s']
        need(math.isclose(t['trigger_to_first_token_s'],endpoint,abs_tol=1e-9)
             and 0<endpoint<=(t['generate_return_ns']-t['trigger_ns'])/1e9,'first-token clock formula')
        need(t['admission_s']+t['parse_s']+t['create_runtime_s']<=t['transition_s']+1e-6,'phase breakdown exceeds total')
    need(r['native_counters']==dict(parse_calls=3*pairs+1,create_calls=4*pairs+1)
         and r['native_run_completed']==15*(4*pairs+1),'whole-run native counters')
    summary={arm:{k:stat([t[k] for t in r['trials'] if t['arm']==arm]) for k in METRICS} for arm in ('full','reuse')}
    need(summary==r['summary'],'reported statistics differ')
    return summary


def exact_export(root,name):
    manifest=read(root/'runs'/name/'exact-manifest.json')
    raw=(root/'runs'/name/'exact.json.gz').read_bytes()
    need(len(raw)==manifest['bytes'] and digest(raw)==manifest['sha256'],'export archive mismatch')
    files=json.loads(gzip.decompress(raw))['files'];need(len(files)==manifest['files'],'export count')
    seen=set()
    for f in files:
        path=PurePosixPath(f['path']);need(not path.is_absolute() and '..' not in path.parts and '\\' not in f['path'],'unsafe export path')
        need(f['path'] not in seen,'duplicate export path');seen.add(f['path'])
        data=base64.b64decode(f['base64'],validate=True)
        need(len(data)==f['bytes'] and digest(data)==f['sha256'],'export file hash')
        need((root/str(path)).read_bytes()==data,'export file bytes differ')
    if name=='inputs':
        expected={'inputs/'+f for f in ('expected-inputs.json','input-receipt.json','source-manifest.json','switchplan.json')}
    else:
        pairs=1 if name=='smoke-01' else 5
        expected={'runs/'+name+'/'+f for f in ('config.json','result.json')}
        expected.update('runs/'+name+'/pair-%02d-%s.json'%(i,a) for i in range(pairs) for a in ('full','reuse'))
    need(seen==expected,'exact export inventory differs')


def verify_pod(root,trial,record):
    d=root/'runs'/trial;created=read(d/'create.json');evidence=read(d/'pod-evidence.json');p=evidence['pod']
    need(p['metadata']['uid']==record['pod_uid'] and p['status']['phase']=='Succeeded','Pod identity/terminal state')
    need(any(o['uid']==created['job_uid'] for o in p['metadata']['ownerReferences']),'Pod/Job ownership')
    need(p['metadata']['name'].startswith('cdx-atom-reuse-') and p['metadata']['namespace']=='codex-atom-reuse-20261102','Pod scope')
    spec=p['spec'];need(spec.get('automountServiceAccountToken') is False,'unexpected service token')
    need([c['name'] for c in spec['containers']]==['run']
         and [c['name'] for c in spec.get('initContainers',[])]==['assigned-atom'],'unexpected container layout/card aggregate')
    for c in spec.get('initContainers',[])+spec['containers']:
        need(not c.get('securityContext',{}).get('privileged',False),'privileged container')
        need(c['resources']['requests']['rebellions.ai/ATOM']==c['resources']['limits']['rebellions.ai/ATOM']=='1','not exactly one card')
    need(all(s['state'].get('terminated',{}).get('exitCode')==0 for s in p['status'].get('initContainerStatuses',[])+p['status']['containerStatuses']),'nonzero native exit')
    need(all(v['returncode']==0 for v in evidence['logs'].values()),'log export failure')
    job=created['job']['spec'];need(job['backoffLimit']==0 and job['ttlSecondsAfterFinished']==600
         and job['activeDeadlineSeconds'] in (900,1800),'unbounded Job')
    cleanup=read(d/'cleanup.json');need(cleanup['job_absent'] is True and cleanup['job_uid']==created['job_uid'],'card not released')


def verify(root):
    seal=read(root/'package-manifest.json')
    paths={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='package-manifest.json'}
    need(paths==set(seal['files']),'package inventory differs')
    for name,entry in seal['files'].items():
        p=PurePosixPath(name);need(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'unsafe package path')
        raw=(root/name).read_bytes();need(len(raw)==entry['bytes'] and digest(raw)==entry['sha256'],'package file hash: '+name)
    exact_export(root,'inputs')
    receipt=read(root/'inputs/input-receipt.json');source=read(root/'inputs/source-manifest.json')
    need(receipt['verified'] is True and receipt['status']=='completed' and receipt['npu_requested']==0,'unverified input receipt')
    need(receipt['graph_sha256']==GRAPH and source['source_commit']==SOURCE,'input/source mismatch')
    need(receipt['base']['session.rbln']['sha256']==receipt['target']['session.rbln']['sha256']==GRAPH,'graph identity')
    post=read(root/'post/input-receipt.json')
    need({k:v for k,v in receipt.items() if k not in ('started_utc','finished_utc')}==
         {k:v for k,v in post.items() if k not in ('started_utc','finished_utc')},'post-run inputs changed')
    for name in ('cluster-before.json','cluster-after.json'):
        inventory=read(root/name)
        need(inventory['active_accelerator_or_cdx_pods']==[] and inventory['cdx_jobs']==[],'accelerator allocation remains')
    archive=(root/'source.tar.gz').read_bytes();need(digest(archive)==source['archive_sha256'],'source archive hash')
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        members=tar.getmembers();need(len(members)==len(source['files']),'source archive count')
        need({m.name for m in members}==set(source['files']),'source archive inventory')
        for m in members:
            need(m.isfile() and digest(tar.extractfile(m).read())==source['files'][m.name],'source archive bytes')
    repo=Path(__file__).resolve().parents[1]
    for name,expected in source['files'].items():
        # Full-history CI checks the precise source that actually ran, not the evolving HEAD.
        p=PurePosixPath(name);need(not p.is_absolute() and '..' not in p.parts,'unsafe source path')
        raw=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=repo)
        need(digest(raw)==expected,'measured Git source hash: '+name)
    out={}
    for trial,phase in [('smoke-01','smoke'),('main-01','main')]:
        exact_export(root,trial);r=read(root/'runs'/trial/'result.json')
        need(r['input_receipt_sha256']==digest((root/'inputs/input-receipt.json').read_bytes()),'receipt hash')
        need(r['source_manifest_sha256']==digest((root/'inputs/source-manifest.json').read_bytes()),'source manifest hash')
        need(r['source_files']==source['files'] and r['harness_sha256']==source['files']['bench/ccgrid_atom_reuse.py'],'executed source identity')
        need(r['config_sha256']==digest((root/'runs'/trial/'config.json').read_bytes()),'config hash')
        need(r['switchplan_sha256']==receipt['switchplan_sha256']==digest((root/'inputs/switchplan.json').read_bytes()),'plan hash')
        out[phase]=verify_record(r,phase);verify_pod(root,trial,r)
        need(receipt['finished_utc']<r['started_utc']<r['finished_utc']<post['started_utc'],'pre/post hash boundary')
        for t in r['trials']:
            need(read(root/'runs'/trial/('pair-%02d-%s.json'%(t['pair'],t['arm'])))==t,'individual trial differs')
    a=out['main']['full']['trigger_to_first_token_s']['median'];b=out['main']['reuse']['trigger_to_first_token_s']['median']
    return dict(schema='ccgrid-atom-reuse-summary.v1',verified=True,source_commit=SOURCE,
                graph_sha256=GRAPH,versions=VERSIONS,package_files=len(paths),smoke=out['smoke'],main=out['main'],
                ratio_of_median_ttft=a/b,ttft_reduction_percent=100*(1-b/a),
                scope='Same graph/metadata revision, parsed host object reuse; B1 reload; not device-resident weights or changed-shape reuse.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).parent/'data/atom-reuse-20261102')
    args=parser.parse_args()
    try:print(json.dumps(verify(args.root),indent=2,allow_nan=False))
    except (ValueError,KeyError,FileNotFoundError) as e:
        print(str(e),file=sys.stderr);raise SystemExit(1)
