"""CPU-only follow-up preparation; never modifies original compiled files/stores."""
import json,pathlib,shutil,sys,time,traceback
ROOT=pathlib.Path('/home/gspark/workspace/codex_resident_delivery_20261102')
sys.path.insert(0,str(ROOT/'cpu-causal'))
import causal as h
r=ROOT/'cpu-model-a01';r.mkdir(exist_ok=False)
record=dict(status='running',npu_requested=0,question='same-cache materialization writes, repeated local baselines and matched chunking',started_unix_ns=time.time_ns(),inputs=[])
def save():h.save(r/'preparation.json',record)
save()
try:
 assert json.loads((ROOT/'cpu-causal-fixture02/fixture-result.json').read_text())['valid']
 fixture_root=r/'ptrace-fixture';fixture_root.mkdir();assert h.fixture(ROOT/'bin/onggi',fixture_root)['valid'],'CPU model environment must pass actual CLI/ptrace fixture'
 assert h.sha(ROOT/'cpu-causal/causal.py')=='1546801671fc19d3871487b2d2d4f0f8e445bae82cbfd25aabf9e87351554f5a'
 assert h.sha(ROOT/'bin/onggi')=='e6b8ee52a69c7cf9fe970cc1df08e93eab9c175c061baccb078725e549a15ae5'
 assert shutil.disk_usage(r).free>160_000_000_000
 for label,name,size in [('base','decode_S256_B2_lm.rbln',15238278850),('target','decode_S256_B8_lm.rbln',15228048882)]:
  source=pathlib.Path('/home/gspark/rbln-llama8b-pd')/name;assert source.is_file() and not source.is_symlink() and source.stat().st_size==size
  dest=r/label;dest.mkdir();copied=dest/name;start=time.perf_counter_ns();expected=h.sha(source);shutil.copy2(source,copied)
  assert h.sha(copied)==expected and (copied.stat().st_dev,copied.stat().st_ino)!=(source.stat().st_dev,source.stat().st_ino) and copied.stat().st_nlink==1
  record['inputs'].append(dict(original=str(source),copy=str(copied),bytes=size,sha256=expected,copy_and_hash_s=(time.perf_counter_ns()-start)/1e9));save()
 cfg=dict(status='not_run',source_revision='05c97605b07882f9ceaaeb90b94878f8923f81cf',go_version='go1.25.1 linux/amd64',binary_build_provenance='resident build02 pinned Go image; go-version.txt and binary SHA retained',binary_sha256=h.sha(ROOT/'bin/onggi'),onggi=str(ROOT/'bin/onggi'),base_dir=str(r/'base'),target_dir=str(r/'target'),store=None,base_manifest=None,target_manifest=None,cache_dir=None,workers=8,gomaxprocs=32,limits_description='K8s CPU-only etri-001, request4/limit12 CPU, request8/limit32Gi memory; actual cgroup/affinity/mounts in frozen.host; nice19; warm rehash; no cache drops',deployment_arms=['P','R','rsync','rsync_inplace','xdelta'],publication_arms=['C','F','T','TF','TA'],retain_outputs=False)
 assert h.sha('/usr/bin/xdelta3')=='fb861f85195a3e6afedd9d5d7ab42273855f9dacb5893b1b6069412d2c574c0f'
 # One TA publication is shared by every P/R timing and cached diagnostic arm.
 for ref,source in [('a',r/'base'),('b',r/'target')]:
  row=h.checked(h.command(cfg,h.publish_argv(cfg,'TA',r,ref,source),r,'prepare-push-'+ref))
  cfg['base_manifest' if ref=='a' else 'target_manifest']=json.loads((r/('prepare-push-'+ref+'.stdout')).read_text())['manifest']
 cfg['store']=str(r/'store');h.save(r/'timing-config.json',cfg);h.freeze(r/'timing-config.json',r/'timing-frozen.json')
 publish=dict(cfg,store=None,base_manifest=None,target_manifest=None);h.save(r/'publish-config.json',publish);h.freeze(r/'publish-config.json',r/'publish-frozen.json')
 seed=h.checked(h.command(cfg,[cfg['onggi'],'pull','--store',cfg['store'],'--cache',str(r/'target-cache'),'--workers','8','--json',cfg['target_manifest'],str(r/'seed-target')],r,'seed-target-cache'))
 assert h.inventory(r/'seed-target')==h.inventory(r/'target');shutil.rmtree(r/'seed-target')
 io=dict(cfg,deployment_arms=['P','R'],cache_dir=str(r/'target-cache'));h.save(r/'io-config.json',io);h.freeze(r/'io-config.json',r/'io-frozen.json')
 record.update(status='completed',finished_unix_ns=time.time_ns(),free_bytes=shutil.disk_usage(r).free,tools={name:h.sha('/usr/bin/'+name) for name in ['time','strace','rsync','xdelta3']});save()
except BaseException:
 record.update(status='failed',error=traceback.format_exc(),finished_unix_ns=time.time_ns());save();raise
print(json.dumps(record))
