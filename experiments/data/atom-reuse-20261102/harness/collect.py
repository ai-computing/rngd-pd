import argparse,json
from control import ROOT,rpc
from cluster import scoped
p=argparse.ArgumentParser();p.add_argument('trial',choices=['env-01','smoke-01','main-01']);a=p.parse_args()
directory=ROOT/'runs'/a.trial
created=json.loads((directory/'create.json').read_bytes());uid=created['job_uid'];name='cdx-atom-reuse-'+a.trial
code=scoped()+f"""
name={name!r};uid={uid!r}
pods=json.loads(subprocess.check_output(k+['-n',ns,'get','pods','-l','job-name='+name,'-o','json'],text=True))['items']
assert len(pods)==1
assert any(o['uid']==uid for o in pods[0]['metadata'].get('ownerReferences',[]))
p=pods[0];logs={{}}
for c in p['spec'].get('initContainers',[])+p['spec']['containers']:
 r=subprocess.run(k+['-n',ns,'logs',p['metadata']['name'],'-c',c['name']],capture_output=True,text=True)
 logs[c['name']]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(dict(pod=p,logs=logs)))
"""
evidence=rpc('dell_0',code,'export-pod-'+a.trial)
(directory/'pod-evidence.json').write_text(json.dumps(evidence,indent=2))
code=f"""import json
from pathlib import Path
r=Path('/home/gspark/workspace/codex_atom_reuse_20261102/runs/{a.trial}')
print(json.dumps({{p.name:json.loads(p.read_bytes()) for p in r.glob('*.json')}}))
"""
results=rpc('rebellion',code,'export-result-'+a.trial)
(directory/'exported-results.json').write_text(json.dumps(results,indent=2))
r=results.get('result.json',{});print(json.dumps(dict(trial=a.trial,pod_phase=evidence['pod']['status']['phase'],status=r.get('status'),phase=r.get('phase'),validity=r.get('validity'),error=r.get('error'),trials=[{k:t.get(k) for k in ('pair','arm','status','transition_s','trigger_to_first_token_s','parse_calls','create_calls')} for t in r.get('trials',[])],summary=r.get('summary')),indent=2))
