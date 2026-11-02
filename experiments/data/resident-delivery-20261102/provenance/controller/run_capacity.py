"""Two separate native processes, same allocated RNGD; bounded correctness only."""
import hashlib,json,os,pathlib,subprocess,sys,time
from rngd_pd import devices
ROOT=pathlib.Path('/home/gspark/workspace/codex_resident_delivery_20261102')
for c in ['r02','r03','r04','r05','r06']:
 assert json.loads((ROOT/'runs'/(c+'-cohort.json')).read_text())['status']=='completed'
r=ROOT/'capacity/c01';r.mkdir(parents=True,exist_ok=False)
chip=devices.resolve_chip(-1);cards=devices.allocated_cards();assert len(cards)==1
record=dict(status='running',pod_uid=os.environ['POD_UID'],chip=chip,cards=cards,started_unix_ns=time.time_ns(),roles=[])
def save():
 (r/'sequence.json').write_text(json.dumps(record,indent=2)+'\n')
save()
for role in ['fresh','resident']:
 command=[sys.executable,str(ROOT/'runtime-source/bench/ccgrid_context_capacity.py'),'--role',role,'--preparation',str(ROOT/'inputs/preparation.json'),'--out',str(r/(role+'.json'))]
 if role=='resident':command+=['--reference',str(r/'fresh.json')]
 with (r/(role+'.stdout.log')).open('x') as out,(r/(role+'.stderr.log')).open('x') as err:
  started=time.time_ns();code=subprocess.call(command,stdout=out,stderr=err)
 result=json.loads((r/(role+'.json')).read_text());row=dict(role=role,argv=command,returncode=code,started_unix_ns=started,finished_unix_ns=time.time_ns(),result_sha256=hashlib.sha256((r/(role+'.json')).read_bytes()).hexdigest(),status=result['status'],validity=result['validity'],counts=result.get('counts'))
 record['roles'].append(row);save();print(json.dumps(row),flush=True)
 if code or result['status']!='completed' or result['validity']!='valid':
  record.update(status='failed',finished_unix_ns=time.time_ns());save();sys.exit(1)
fresh=json.loads((r/'fresh.json').read_text());resident=json.loads((r/'resident.json').read_text())
assert fresh['tokens']==resident['tokens'] and resident['base_rejected_overflow'] and resident['resident_buffers_preserved'] and resident['base_programs_preserved']
record.update(status='completed',finished_unix_ns=time.time_ns());save()
