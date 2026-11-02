"""UID-scoped terminal Job cleanup; preserves observations before deletion."""
import argparse, json
from control import rpc, ROOT
parser=argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('uid')
args=parser.parse_args()
assert args.name.startswith('cdx-resident-')
code=f'''import json,subprocess,time
k=['/usr/bin/kubectl','--kubeconfig=/home/gspark/.kube/config','--context=kubernetes-admin@kubernetes']
ns='codex-resident-delivery-20261102';name={args.name!r};uid={args.uid!r}
namespace=json.loads(subprocess.check_output(k+['get','namespace',ns,'-o','json'],text=True))
assert namespace['metadata']['uid']=='610c63b2-f6d7-42f2-80e7-c725b2dbeb1b'
def get(kind,name):
    p=subprocess.run(k+['-n',ns,'get',kind,name,'--ignore-not-found','-o','json'],text=True,capture_output=True,check=True)
    return json.loads(p.stdout) if p.stdout.strip() else None
j=get('job',name)
pods=json.loads(subprocess.check_output(k+['-n',ns,'get','pods','-l','job-name='+name,'-o','json'],text=True))['items']
assert all(p['status']['phase'] in ('Succeeded','Failed') for p in pods),'Nonterminal owned Job must be inspected/adopted'
observations=[]
for p in pods:
    log=subprocess.run(k+['-n',ns,'logs',p['metadata']['name'],'--all-containers=true'],capture_output=True,text=True)
    observations.append(dict(name=p['metadata']['name'],uid=p['metadata']['uid'],phase=p['status']['phase'],containers=p['status'].get('containerStatuses',[]),log=log.stdout,log_stderr=log.stderr))
if j:
    assert j['metadata']['uid']==uid
    options=dict(apiVersion='v1',kind='DeleteOptions',propagationPolicy='Foreground',preconditions=dict(uid=uid))
    subprocess.run(k+['delete','--raw','/apis/batch/v1/namespaces/'+ns+'/jobs/'+name,'-f','-'],input=json.dumps(options),text=True,capture_output=True,check=True)
until=time.monotonic()+40
while get('job',name) or any(get('pod',p['metadata']['name']) for p in pods):
    assert time.monotonic()<until;time.sleep(.5)
print(json.dumps(dict(job_uid=uid,name=name,job_absent=True,pods_absent=True,observations=observations)))
'''
result=rpc('dell_0',code,'cleanup-'+args.name)
path=ROOT/(args.name+'-cleanup.json')
assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(name=args.name,job_absent=result['job_absent'],pods_absent=result['pods_absent'],receipt=str(path))))
