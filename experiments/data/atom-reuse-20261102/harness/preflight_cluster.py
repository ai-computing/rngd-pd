import json
import subprocess
import time
k=['/usr/bin/kubectl','--kubeconfig=/home/gspark/.kube/config','--context=kubernetes-admin@kubernetes']
def get(kind):
    return json.loads(subprocess.check_output(k+['get',kind,'-A','-o','json'],text=True))['items']
keys=('rebellions.ai/ATOM','furiosa.ai/rngd')
pods=[]
for p in get('pods'):
    if p.get('status',{}).get('phase') in ('Succeeded','Failed'):continue
    requests={}
    for c in p['spec'].get('containers',[])+p['spec'].get('initContainers',[]):
        for key in keys:
            value=c.get('resources',{}).get('requests',{}).get(key)
            if value is not None:requests[key]=requests.get(key,0)+int(value)
    if requests or p['metadata']['name'].startswith('cdx-'):
        pods.append(dict(namespace=p['metadata']['namespace'],name=p['metadata']['name'],uid=p['metadata']['uid'],
            node=p['spec'].get('nodeName'),phase=p.get('status',{}).get('phase'),requests=requests))
nodes=[dict(name=n['metadata']['name'],allocatable={key:n['status'].get('allocatable',{}).get(key) for key in keys},
    ready=any(c['type']=='Ready' and c['status']=='True' for c in n['status'].get('conditions',[]))) for n in get('nodes')]
jobs=[dict(namespace=j['metadata']['namespace'],name=j['metadata']['name'],uid=j['metadata']['uid'],status=j.get('status',{}))
      for j in get('jobs') if j['metadata']['name'].startswith('cdx-')]
print(json.dumps(dict(checked_utc=time.time(),nodes=nodes,active_accelerator_or_cdx_pods=pods,cdx_jobs=jobs)))
