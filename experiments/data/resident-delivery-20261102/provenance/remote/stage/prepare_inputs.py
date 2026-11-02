"""CPU-only preparation executed in a bounded privileged Job, never an NPU owner."""
import hashlib,json,os,pathlib,shutil,subprocess,time,traceback

ROOT=pathlib.Path('/home/gspark/workspace/codex_resident_delivery_20261102')
OLD=pathlib.Path('/home/gspark/workspace/codex_ccgrid_npu_20261030')
FULL_BASE='sha256:04b45065cea8408e08043c1cbd13bdc4aa230c9683a9d85ed97f5fb1e5567526'
FULL_TARGET='sha256:66cedc7b2a1389394c7ca5275d171625993c23742d7c7834bcb7e93a2be52e52'
BIN=str(ROOT/'bin/onggi'); CLIENT=str(ROOT/'bin/csi-call')
started=time.perf_counter_ns()
out=ROOT/'inputs/preparation.json'
assert not out.exists(),'Existing preparation must be inspected/adopted'
record=dict(status='running',npu_requested=0,started_unix_ns=time.time_ns(),events=[],sources={},graph_pushes={})

def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(record,indent=2)+'\n');os.replace(p,out)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def inventory(root):
 rows={}
 for p in sorted(pathlib.Path(root).rglob('*')):
  if p.is_file():
   s=p.stat();rows[p.relative_to(root).as_posix()]=dict(size=s.st_size,inode=s.st_ino,device=s.st_dev,blocks=s.st_blocks,sha256=sha(p))
 return rows

def run(argv,label):
 row=dict(label=label,argv=list(map(str,argv)),started_unix_ns=time.time_ns(),started_perf_ns=time.perf_counter_ns())
 record['events'].append(row);save()
 p=subprocess.run(row['argv'],capture_output=True,text=True)
 row.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,finished_unix_ns=time.time_ns(),finished_perf_ns=time.perf_counter_ns());save()
 if p.returncode:raise RuntimeError('CPU command failed: '+label)
 return row

try:
 device_fds=[]
 for p in pathlib.Path('/proc/self/fd').iterdir():
  try: target=p.resolve(strict=True)
  except FileNotFoundError: continue
  if str(target).startswith(('/dev/rngd','/dev/rbln','/dev/furiosa')):device_fds.append(str(target))
 record['accelerator_fds_before']=device_fds
 assert not device_fds,'CPU preparation has an accelerator fd'
 oldbase=OLD/'artifacts-v2/base-B4-S256-g2';oldtarget=OLD/'artifacts-v2/target-B4-S512-g2'
 for name,p in [('full_base',oldbase),('full_target',oldtarget),('checkpoint',pathlib.Path('/home/gspark/workspace/codex_swap_benchmark_20261023/fixtures/resident-base'))]:
  record['sources'][name]=dict(root=str(p),inventory=inventory(p));save()
 for name,source in [('base',oldbase),('target',oldtarget)]:
  dest=ROOT/'artifacts'/('graph-'+name);dest.mkdir(parents=True)
  shutil.copytree(source/'cache',dest/'cache');shutil.copy2(source/'variant.json',dest/'variant.json')
  record['sources']['graph_'+name]=dict(root=str(dest),inventory=inventory(dest));save()
  record['graph_pushes'][name]=run([BIN,'push','--store',str(ROOT/'graph-store'),'--json','--ref','resident/graph-'+name,str(dest)],'push-graph-'+name)
 refs={name:json.loads(row['stdout'])['manifest'] for name,row in record['graph_pushes'].items()}
 record['graph_refs']=refs;save()
 run([BIN,'diff','--store',str(ROOT/'graph-store'),'--json','--ranges',refs['base'],refs['target']],'diff-graph')
 # Source payload hashes are fixed before allocating an accelerator. Store roots remain read-only in native Jobs.
 record['full_refs']=dict(base=FULL_BASE,target=FULL_TARGET)
 record['full_store']=str(OLD/'store/e6-posix')
 record['reference']=dict(path=str(OLD/'inputs/e6-base-to-target-reference.json'),sha256=sha(OLD/'inputs/e6-base-to-target-reference.json'))
 record['base_cache_dir']=str(OLD/'cache/graphs')
 record['binary_files']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [ROOT/'bin/onggi',ROOT/'bin/csi-call']}
 record.update(status='completed',finished_unix_ns=time.time_ns(),wall_s=(time.perf_counter_ns()-started)/1e9)
except Exception:
 record.update(status='failed',traceback=traceback.format_exc(),finished_unix_ns=time.time_ns());raise
finally:save()
