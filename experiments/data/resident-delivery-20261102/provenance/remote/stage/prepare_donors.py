"""Off-card real CSI donor preparation; one bounded cohort at a time."""
import hashlib,json,os,pathlib,shutil,sys,time,traceback
import ccgrid_resident_delivery as h

ROOT=pathlib.Path('/home/gspark/workspace/codex_resident_delivery_20261102')
cohort,phase=sys.argv[1:3]
assert cohort.startswith(('s','r')) and phase in ('smoke','measure') and cohort.replace('-','').isalnum()
inputs=ROOT/'inputs';prep=json.loads((inputs/'preparation.json').read_text());assert prep['status']=='completed'
destination=inputs/'cohorts'/cohort;destination.mkdir(parents=True,exist_ok=False)
record=dict(status='running',cohort=cohort,phase=phase,npu_requested=0,started_unix_ns=time.time_ns(),trials=[])
def save():
 (destination/'cohort.json').write_text(json.dumps(record,indent=2)+'\n')
save()
arms=['graph-copy','graph-csi','full-csi-no-rebase','full-csi-rebase']
if phase=='smoke':arms+=['graph-csi-active']
elif int(cohort[1:])%2==0:arms.reverse()
try:
 for arm in arms:
  tid=cohort+'-'+arm;graph=arm.startswith('graph-');label={'graph-copy':'gc','graph-csi':'gd','full-csi-no-rebase':'fn','full-csi-rebase':'fd','graph-csi-active':'ge'}[arm]
  config=dict(trial_id=tid,phase=phase,arm=arm,chip=0,host_workers=2,intra_op=4,
   immutable_checkpoint=prep['sources']['checkpoint']['root'],base_cache_dir=prep['base_cache_dir'],
   base_bundle_source=prep['sources']['graph_base' if graph else 'full_base']['root'],
   target_bundle_source=prep['sources']['graph_target' if graph else 'full_target']['root'],
   base_ref=prep['graph_refs' if graph else 'full_refs']['base'],target_ref=prep['graph_refs' if graph else 'full_refs']['target'],
   store=str(ROOT/'graph-store') if graph else prep['full_store'],onggi_binary=str(ROOT/'bin/onggi'),csi_client=str(ROOT/'bin/csi-call'),
   reference_file=prep['reference']['path'],base_variant='B4-S256-g2',target_variant='B4-S512-g2',work_dir=str(ROOT/'runs'/tid/'work'),
   provenance=dict(runtime_baseline='db2dd2598daffb3d3f722007599b74d00ede61af',harness_source='a215ff8c1ba07c95a83a56715877b5cd1fd939af',onggi_source='05c9760',preparation_sha256=hashlib.sha256((inputs/'preparation.json').read_bytes()).hexdigest(),cohort=cohort,cache_state='empty_application_chunk_cache',peers=False))
  if arm not in ('graph-copy','graph-csi-active'):
   # Limit peak scratch footprint to a single cohort; never prepare five full cohorts in advance.
   free=shutil.disk_usage(ROOT).free;assert free>80000000000,'insufficient declared scratch headroom'
   control=ROOT/'prep'/cohort/label/'c';control.mkdir(parents=True,exist_ok=False);data=control.parent/'m';data.mkdir()
   result=dict(status='running',base_ref=config['base_ref'],materializer=str(data),npu_requested=0,started_unix_ns=time.time_ns(),disk_free_before=free)
   csi=h.CSI(config,control,data,result);mounted=False
   try:
    csi.start('prep');mounted=True
    csi.rpc('publish',control/'base','base',config['base_ref'],'base-publish')
    result['mount']=h.readonly_mount(control/'base')
    donor=data/'artifacts'/config['base_ref'].split(':')[1]
    result['inventory']=h.inventory(donor)
    expected=prep['sources']['graph_base' if graph else 'full_base']['inventory']
    assert set(result['inventory'])==set(expected)
    for name,row in result['inventory'].items():
     assert row['size']==expected[name]['size'] and row['sha256']==expected[name]['sha256']
     assert (row['inode'],row['device'])!=(expected[name]['inode'],expected[name]['device']),'donor shares source inode'
    csi.rpc('unpublish',control/'base','base',config['base_ref'],'base-idle');mounted=False
    csi.stop();h.verify_cleanup(data,[control/'base'])
    result['final_state']=h.snapshot(data,[donor])
    result.update(status='completed',finished_unix_ns=time.time_ns(),disk_free_after=shutil.disk_usage(ROOT).free)
   except Exception:
    result.update(status='failed',traceback=traceback.format_exc());raise
   finally:
    if mounted:
     try:csi.rpc('unpublish',control/'base','base',config['base_ref'],'failed-base-cleanup')
     except Exception:result['cleanup_error']=traceback.format_exc()
    csi.stop();(control/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
   config.update(prepared_materializer=str(data),preparation_receipt=str(control/'receipt.json'))
  path=destination/(tid+'.json');path.write_text(json.dumps(config,indent=2)+'\n')
  h.validate(config)
  record['trials'].append(dict(trial_id=tid,arm=arm,config=str(path),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));save()
 record.update(status='completed',finished_unix_ns=time.time_ns());save()
except Exception:
 record.update(status='failed',traceback=traceback.format_exc(),finished_unix_ns=time.time_ns());save();raise
