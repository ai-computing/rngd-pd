"""Bounded five-cohort matrix. No retry/restart; failures require authoritative adoption."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from control import rpc,ROOT
STATE=ROOT/'main-controller-state.json';assert not STATE.exists(),'Adopt recorded controller and actual Jobs before resuming'
state=dict(status='running',pid=os.getpid(),runtime_source='829573e41959a4b0f6daa82e4862f1584367c347',cohorts=[],started_unix_ns=time.time_ns())
def save():
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(STATE)
def log(s):print(time.strftime('%H:%M:%S',time.gmtime())+' '+s,flush=True)
def local(script,*args):
 command=[sys.executable,str(ROOT/script),*args];p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
 with (ROOT/'main-controller-commands.jsonl').open('a') as f:f.write(json.dumps(dict(argv=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,unix_ns=time.time_ns()))+'\n')
 if p.returncode:raise RuntimeError('Command failed: '+script+' '+repr(args)+'; exact command outputs retained')
 return p.stdout
def wait(name):
 while True:
  code=f'''import subprocess,json
k=['/usr/bin/kubectl','--kubeconfig=/home/gspark/.kube/config','--context=kubernetes-admin@kubernetes'];ns='codex-resident-delivery-20261102';name={name!r}
j=json.loads(subprocess.check_output(k+['-n',ns,'get','job',name,'-o','json'],text=True))
p=json.loads(subprocess.check_output(k+['-n',ns,'get','pods','-l','job-name='+name,'-o','json'],text=True))['items']
print(json.dumps(dict(uid=j['metadata']['uid'],status=j.get('status',{{}}),pods=[dict(uid=q['metadata']['uid'],phase=q['status']['phase'],containers=q['status'].get('containerStatuses',[])) for q in p])))
'''
  r=rpc('dell_0',code,'watch-'+name);state['current_job']=dict(name=name,observed=r);save()
  if r['pods'] and all(p['phase'] in ('Succeeded','Failed') for p in r['pods']):
   if not all(p['phase']=='Succeeded' and all(c.get('state',{}).get('terminated',{}).get('exitCode')==0 and c.get('restartCount')==0 for c in p['containers']) for p in r['pods']):
    local('inspect_job.py',name);raise RuntimeError('Terminal failed Job: '+name)
   local('inspect_job.py',name);return r['uid']
  if any(c.get('type')=='Failed' and c.get('status')=='True' for c in r['status'].get('conditions',[])):
   local('inspect_job.py',name);raise RuntimeError('Job controller reports terminal failure: '+name)
  time.sleep(15)
def clean(name,uid):local('cleanup_job.py',name,uid);state['current_job']=None;save()
save()
try:
 for i in range(1,6):
  cohort='r%02d'%i;row=dict(cohort=cohort,status='preparing');state['cohorts'].append(row);save();log(cohort+' CPU preparation')
  name='cdx-resident-donors-'+cohort;local('create_cpu_job.py',name,'prepare_main.py',cohort,'measure');uid=wait(name);clean(name,uid)
  row['status']='native';save();log(cohort+' native four-arm measurement')
  name='cdx-resident-native-'+cohort;local('create_native.py',cohort);uid=wait(name);clean(name,uid)
  row['status']='collecting';save();log(cohort+' card released; exact export and independent verification')
  local('collect_cohort.py',cohort);local('check_cohort.py',cohort)
  name='cdx-resident-clean-'+cohort;local('create_cpu_job.py',name,'cleanup_payload.py',cohort);uid=wait(name);clean(name,uid)
  row.update(status='verified',finished_unix_ns=time.time_ns());save();log(cohort+' verified and payload cleanup complete')
 state.update(status='completed',finished_unix_ns=time.time_ns());save();log('All five cohorts verified; statistical analysis/publication still pending')
except BaseException:
 state.update(status='interrupted',error=traceback.format_exc(),finished_unix_ns=time.time_ns());save();log('STOPPED; preserve and inspect the recorded current Job, never relaunch blindly');raise
