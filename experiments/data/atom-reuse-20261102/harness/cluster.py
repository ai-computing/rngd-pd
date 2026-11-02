"""Own-namespace K8s operations via hub transport; no SSH/direct node access."""
import argparse
import base64
import json
from pathlib import Path
from control import rpc,ROOT

NS='codex-atom-reuse-20261102'
K="k=['/usr/bin/kubectl','--kubeconfig=/home/gspark/.kube/config','--context=kubernetes-admin@kubernetes']\n"
PRE='import json,subprocess,time\n'+K


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def scoped():
    uid=json.loads((ROOT/'namespace.json').read_bytes())['metadata']['uid']
    return PRE+f"ns={NS!r}\nnamespace=json.loads(subprocess.check_output(k+['get','namespace',ns,'-o','json'],text=True))\nassert namespace['metadata']['uid']=={uid!r}\n"


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['namespace','submit','status','cleanup'])
    parser.add_argument('trial',nargs='?')
    args=parser.parse_args()
    if args.operation=='namespace':
        assert not (ROOT/'namespace.json').exists()
        doc=dict(apiVersion='v1',kind='Namespace',metadata=dict(name=NS,labels=dict(owner='codex',experiment='atom-reuse-20261102')))
        code=PRE+f"p=subprocess.run(k+['get','namespace',{NS!r},'--ignore-not-found','-o','name'],capture_output=True,text=True,check=True)\nassert not p.stdout.strip()\nprint(subprocess.check_output(k+['create','-f','-','-o','json'],input={json.dumps(doc)!r},text=True))"
        result=rpc('dell_0',code,'create-namespace');save(ROOT/'namespace.json',result)
    else:
        assert args.trial in ('env-01','smoke-01','main-01')
        trial=ROOT/'runs'/args.trial;trial.mkdir(parents=True,exist_ok=True)
        name='cdx-atom-reuse-'+args.trial
        if args.operation=='submit':
            assert not (trial/'create.json').exists()
            job=json.loads((trial/'job.json').read_bytes())
            assert job['metadata']['name']==name and job['metadata']['namespace']==NS
            assert job['spec']['backoffLimit']==0 and job['spec']['ttlSecondsAfterFinished']==600
            resource=job['spec']['template']['spec']['containers'][0]['resources']
            count=0 if args.trial=='env-01' else 1
            assert int(resource['requests'].get('rebellions.ai/ATOM','0'))==count
            code=scoped()+f"""allpods=json.loads(subprocess.check_output(k+['get','pods','-A','-o','json'],text=True))['items']
for p in allpods:
 if p.get('status',{{}}).get('phase') in ('Succeeded','Failed'):continue
 assert not (p['metadata']['namespace']==ns and p['metadata']['name'].startswith('cdx-atom-reuse-')),'own prior Job still active'
 if {count}:
  assert not any(int(c.get('resources',{{}}).get('requests',{{}}).get('rebellions.ai/ATOM','0')) for c in p['spec']['containers']),'ATOM benchmark already active'
data={json.dumps(job)!r}
validated=subprocess.check_output(k+['-n',ns,'create','--dry-run=server','-f','-','-o','json'],input=data,text=True)
created=json.loads(subprocess.check_output(k+['-n',ns,'create','-f','-','-o','json'],input=data,text=True))
print(json.dumps(dict(job_uid=created['metadata']['uid'],job=created,dry_run=json.loads(validated))))
"""
            result=rpc('dell_0',code,'submit-'+args.trial);save(trial/'create.json',result)
        else:
            created=json.loads((trial/'create.json').read_bytes());uid=created['job_uid']
            code=scoped()+f"""
name={name!r};uid={uid!r}
def get(kind,name):
 p=subprocess.run(k+['-n',ns,'get',kind,name,'--ignore-not-found','-o','json'],capture_output=True,text=True,check=True)
 return json.loads(p.stdout) if p.stdout.strip() else None
j=get('job',name)
pods=json.loads(subprocess.check_output(k+['-n',ns,'get','pods','-l','job-name='+name,'-o','json'],text=True))['items']
assert j is None or j['metadata']['uid']==uid
assert all(any(o['uid']==uid for o in p['metadata'].get('ownerReferences',[])) for p in pods)
"""
            if args.operation=='status':
                code+="print(json.dumps(dict(job=j,pods=pods)))\n"
                result=rpc('dell_0',code,'status-'+args.trial);save(trial/'latest-status.json',result)
            else:
                code+="""assert all(p['status']['phase'] in ('Succeeded','Failed') for p in pods),'Job is not terminal'
observed=[dict(name=p['metadata']['name'],uid=p['metadata']['uid'],phase=p['status']['phase'],
 containers=p['status'].get('containerStatuses',[])) for p in pods]
if j is not None:
 assert len(pods)==1
 options=dict(apiVersion='v1',kind='DeleteOptions',propagationPolicy='Foreground',preconditions=dict(uid=uid))
 subprocess.run(k+['delete','--raw','/apis/batch/v1/namespaces/'+ns+'/jobs/'+name,'-f','-'],input=json.dumps(options),text=True,capture_output=True,check=True)
until=time.monotonic()+40
while get('job',name) is not None or any(get('pod',p['name']) is not None for p in observed):
 assert time.monotonic()<until;time.sleep(.5)
print(json.dumps(dict(job_uid=uid,observed_terminal=observed,job_absent=True,pods_absent=True)))
"""
                result=rpc('dell_0',code,'cleanup-'+args.trial);save(trial/'cleanup.json',result)
    print(json.dumps(dict(operation=args.operation,trial=args.trial,
        job_uid=result.get('job_uid'),uid=result.get('metadata',{}).get('uid'),
        pods=[dict(name=p['metadata']['name'],phase=p.get('status',{}).get('phase')) for p in result.get('pods',[])],
        job_absent=result.get('job_absent')),indent=2))
