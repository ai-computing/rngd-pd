"""Native coordinator: fresh interpreter/owner per trial, one assigned card per Job."""
import hashlib,json,os,pathlib,subprocess,sys,time
from rngd_pd import devices
ROOT=pathlib.Path('/home/gspark/workspace/codex_resident_delivery_20261102')
cohort=sys.argv[1];plan=ROOT/'inputs/cohorts'/cohort/'cohort.json';p=json.loads(plan.read_text());assert p['status']=='completed'
chip=devices.resolve_chip(-1);cards=devices.allocated_cards();assert cards is not None and len(cards)==1
assert pathlib.Path('/run/swap/rngd').read_text().strip()=='npu'+str(cards[0])
out=ROOT/'runs'/(cohort+'-cohort.json');assert not out.exists()
result=dict(status='running',cohort=cohort,phase=p['phase'],plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),assignment=dict(physical=cards[0],chip=chip,pod_uid=os.environ['POD_UID'],pod_name=os.environ['POD_NAME']),trials=[],started_unix_ns=time.time_ns())
def save():
 temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');os.replace(temp,out)
save()
for entry in p['trials']:
 directory=ROOT/'runs'/entry['trial_id'];directory.mkdir(exist_ok=False)
 template=pathlib.Path(entry['config']);assert hashlib.sha256(template.read_bytes()).hexdigest()==entry['config_sha256']
 cfg=json.loads(template.read_text());cfg['chip']=chip;cfg['provenance'].update(template_sha256=entry['config_sha256'],pod_uid=os.environ['POD_UID'],pod_name=os.environ['POD_NAME'])
 actual=directory/'config.json';actual.write_text(json.dumps(cfg,indent=2)+'\n')
 command=[sys.executable,str(ROOT/'runtime-source/bench/ccgrid_resident_delivery.py'),'trial','--config',str(actual),'--out',str(directory/'result.json')]
 with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
  started=time.time_ns();process=subprocess.run(command,stdout=stdout,stderr=stderr)
 row=dict(trial_id=entry['trial_id'],arm=entry['arm'],argv=command,started_unix_ns=started,finished_unix_ns=time.time_ns(),returncode=process.returncode)
 record=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else {}
 row.update(status=record.get('status','missing'),validity=record.get('validity','incomplete'),metrics=record.get('metrics'),result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest() if record else None)
 result['trials'].append(row);save();print(json.dumps(row),flush=True)
 if process.returncode or row['status']!='completed' or row['validity']!='valid':
  result.update(status='failed',finished_unix_ns=time.time_ns());save();sys.exit(1)
result.update(status='completed',finished_unix_ns=time.time_ns());save()
