import argparse,copy,json
from pathlib import Path
from control import rpc,ROOT,REMOTE
p=argparse.ArgumentParser();p.add_argument('cohort');a=p.parse_args();assert a.cohort.replace('-','').isalnum()
name='cdx-resident-native-'+a.cohort
job=json.loads(Path('C:/Users/USER/orca/scratch/ccgrid-b2-20261102/cdx-b2-main-01-job.json').read_bytes())
job['metadata'].update(name=name,namespace='codex-resident-delivery-20261102',labels=dict(owner='codex',experiment='context-capacity-native'))
job['spec'].update(activeDeadlineSeconds=1200,ttlSecondsAfterFinished=900)
spec=job['spec']['template']['spec'];c=spec['containers'][0]
c['command']=['/home/gspark/venvs/furiosa-2026.4/bin/python',REMOTE+'/stage/run_capacity.py']
for mount in c['volumeMounts']:
 if mount['name'] not in ('assigned','shm'):mount['readOnly']=True
# Preserve writable assignment/shm; give only this new task a writable persistent tree.
spec['volumes'].append(dict(name='resident-task',hostPath=dict(path=REMOTE,type='Directory')))
c['volumeMounts'].append(dict(name='resident-task',mountPath=REMOTE,readOnly=False))
env={row['name']:row for row in c['env']}
for k,v in dict(PYTHONPATH=REMOTE+'/runtime-source/src:'+REMOTE+'/runtime-source/bench',TMPDIR=REMOTE+'/tmp',XDG_CACHE_HOME=REMOTE+'/xdg',HF_HUB_OFFLINE='1',RNGD_MAX_SHAPES='8').items():env[k]=dict(name=k,value=v)
c['env']=list(env.values())
(ROOT/(name+'-job.json')).write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
code=f'''import json,subprocess
k=['/usr/bin/kubectl','--kubeconfig=/home/gspark/.kube/config','--context=kubernetes-admin@kubernetes'];ns='codex-resident-delivery-20261102'
n=json.loads(subprocess.check_output(k+['get','namespace',ns,'-o','json'],text=True));assert n['metadata']['uid']=='610c63b2-f6d7-42f2-80e7-c725b2dbeb1b'
pods=json.loads(subprocess.check_output(k+['get','pods','-A','-o','json'],text=True))['items']
active=[p for p in pods if p['metadata']['name'].startswith('cdx-') and p['status'].get('phase') not in ('Succeeded','Failed')];assert not active,'Adopt existing active cdx process first'
job={job!r}
p=subprocess.run(k+['-n',ns,'get','job',job['metadata']['name'],'--ignore-not-found','-o','json'],capture_output=True,text=True,check=True);assert not p.stdout.strip(),'Adopt existing Job'
j=json.loads(subprocess.check_output(k+['create','-f','-','-o','json'],input=json.dumps(job),text=True))
print(json.dumps(dict(name=j['metadata']['name'],job_uid=j['metadata']['uid'],namespace_uid=n['metadata']['uid'])))
'''
r=rpc('dell_0',code,'launch-'+name);(ROOT/(name+'-create.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
