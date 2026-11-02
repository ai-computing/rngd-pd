"""Create a uniquely named CPU-only Job; never replace/adopt an existing name."""
import argparse,json
from pathlib import Path
from control import rpc,REMOTE,ROOT
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('script');p.add_argument('script_args',nargs='*');p.add_argument('--deadline',type=int,default=900);a=p.parse_args()
assert a.name.startswith('cdx-resident-')
job=json.loads(Path('C:/Users/USER/orca/scratch/ccgrid-b2-20261102/cpu-probe-job.json').read_bytes())
job['metadata'].update(name=a.name,namespace='codex-resident-delivery-20261102',labels=dict(owner='codex',experiment='resident-delivery-cpu'))
job['spec'].update(activeDeadlineSeconds=a.deadline,ttlSecondsAfterFinished=900)
spec=job['spec']['template']['spec'];spec.pop('initContainers',None)
c=spec['containers'][0];c['command']=['/usr/bin/python3.10',REMOTE+'/stage/'+a.script]+a.script_args;c['securityContext']=dict(privileged=True)
c['resources']=dict(requests=dict(cpu='4',memory='8Gi'),limits=dict(cpu='12',memory='32Gi'))
for mount in c['volumeMounts']:mount['readOnly']=True
for name,path,writable in [('owned',REMOTE,True),('old-npu','/home/gspark/workspace/codex_ccgrid_npu_20261030',False)]:
 if any(mount['mountPath']==path for mount in c['volumeMounts']):continue
 spec['volumes'].append(dict(name=name,hostPath=dict(path=path,type='Directory')));c['volumeMounts'].append(dict(name=name,mountPath=path,readOnly=not writable))
c['env'].append(dict(name='PYTHONPATH',value=REMOTE+'/runtime-source/src:'+REMOTE+'/runtime-source/bench'))
(ROOT/(a.name+'-job.json')).write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
code=f'''import json,subprocess
k=['/usr/bin/kubectl','--kubeconfig=/home/gspark/.kube/config','--context=kubernetes-admin@kubernetes'];ns='codex-resident-delivery-20261102'
n=json.loads(subprocess.check_output(k+['get','namespace',ns,'-o','json'],text=True));assert n['metadata']['uid']=='610c63b2-f6d7-42f2-80e7-c725b2dbeb1b'
job={job!r}
p=subprocess.run(k+['-n',ns,'get','job',job['metadata']['name'],'--ignore-not-found','-o','json'],capture_output=True,text=True,check=True);assert not p.stdout.strip(),'Adopt existing Job'
j=json.loads(subprocess.check_output(k+['create','-f','-','-o','json'],input=json.dumps(job),text=True))
print(json.dumps(dict(name=j['metadata']['name'],job_uid=j['metadata']['uid'],namespace_uid=n['metadata']['uid'])))
'''
result=rpc('dell_0',code,'create-'+a.name)
(ROOT/(a.name+'-create.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
