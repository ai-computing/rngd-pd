"""CPU-only authentic onggi manifests for the previously compiled public graph."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
r=Path('/home/gspark/workspace/codex_binding_conformance_20261030')
record=dict(status='running',utc=time.time(),scope='off-card graph hardlinks and actual onggi push/diff; no runtime creation',commands=[])
out=r/'evidence/atom-artifact-preparation.json'
def save():out.write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    graph=r/'atom-public-session/session.rbln'
    for role in ['base','target']:
        artifact=r/'atom-artifacts'/role;artifact.mkdir(parents=True,exist_ok=False)
        for source,name in [(graph,'session.rbln'),(Path(str(graph)+'.json'),'session.rbln.json')]:
            os.link(source,artifact/name)
        (artifact/'atom-binding.json').write_text(json.dumps(dict(graph='session.rbln'))+'\n')
        profile=dict(schema='onggi.profile.v1',device='RBLN-CA22',runtime='rebel-compiler-0.11.0',format='rbln',layout='public-embedded-weight-session',batch=1,revision=role)
        (artifact/'profile.json').write_text(json.dumps(profile,sort_keys=True)+'\n')
    binary='/home/gspark/workspace/codex_ccgrid_npu_20261030/tools/onggi-v1'
    record['binary_sha256']=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    store=str(r/'atom-store');env=dict(os.environ,GOMAXPROCS='4')
    for role in ['base','target']:
        argv=[binary,'push','--store',store,'--sub','0','--compress','none','--workers','2','--uploads','2','--quiet','--json','--ref',role,str(r/'atom-artifacts'/role)]
        t=time.monotonic();p=subprocess.run(argv,capture_output=True,env=env,timeout=600)
        row=dict(argv=argv,returncode=p.returncode,elapsed_s=time.monotonic()-t,stderr=p.stderr.decode(),stdout=json.loads(p.stdout) if p.returncode==0 else p.stdout.decode())
        record['commands'].append(row);save();assert p.returncode==0,row
    cases=[]
    for role,name in [('base','native-identical'),('target','native-metadata-revision')]:
        argv=[binary,'diff','--store',store,'--ranges','--json','base',role]
        p=subprocess.run(argv,capture_output=True,env=env,timeout=90);assert p.returncode==0,p.stderr
        plan=r/'inputs'/('atom-'+role+'.switchplan.json');plan.write_bytes(p.stdout)
        record['commands'].append(dict(argv=argv,returncode=0,output=str(plan),sha256=hashlib.sha256(p.stdout).hexdigest()))
        cases.append(dict(name=name,base_dir=str(r/'atom-artifacts/base'),target_dir=str(r/'atom-artifacts'/role),switchplan=plan.name,expect_mode='reload'))
    provider=r/'inputs/atom-fixtures.json';provider.write_text(json.dumps(dict(schema='onggi.binding-fixtures.v1',cases=cases),indent=2)+'\n')
    record.update(status='completed',finished_utc=time.time(),provider=str(provider),provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest());save()
except BaseException:
    record.update(status='failed',traceback=traceback.format_exc(),finished_utc=time.time());save();raise
