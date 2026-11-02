"""Independent local gates over exported records, before any statistical summary."""
import argparse,base64,gzip,hashlib,json,math,statistics,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
DATA=REPO/'experiments/data/resident-delivery-20261102'
PINS={'s01':'a215ff8c1ba07c95a83a56715877b5cd1fd939af'}
FINAL_PIN='829573e41959a4b0f6daa82e4862f1584367c347'
def load(p):return json.loads(p.read_bytes())
def verify_cohort(data_root,cohort,repository=REPO):
 root=Path(data_root)/cohort;manifest=load(root/'export-manifest.json')
 packed=(root/'small-evidence.json.gz').read_bytes()
 assert len(packed)==manifest['archive']['bytes'] and hashlib.sha256(packed).hexdigest()==manifest['archive']['sha256']
 archive=json.loads(gzip.decompress(packed));assert archive['cohort']==cohort
 assert [{k:v for k,v in q.items() if k!='base64'} for q in archive['files']]==manifest['files']
 assert len({q['path'] for q in archive['files']})==len(archive['files'])
 for q in archive['files']:
  p=(root/q['path']).resolve();assert p.is_relative_to(root.resolve())
  assert p.read_bytes()==base64.b64decode(q['base64'],validate=True)
 for entry in manifest['files']:
  b=(root/entry['path']).read_bytes();assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
 plan=load(root/'inputs/cohorts'/cohort/'cohort.json');result=load(root/'runs'/(cohort+'-cohort.json'));assert result['status']=='completed'
 preparation=load(root/'inputs/preparation.json')
 expected=['graph-copy','graph-csi','full-csi-no-rebase','full-csi-rebase']
 if cohort=='s02':expected=['full-csi-active']
 elif plan['phase']=='smoke':expected+=['graph-csi-active']
 elif int(cohort[1:])%2==0:expected.reverse()
 assert [t['arm'] for t in result['trials']]==expected
 for item in result['trials']:
  d=load(root/'runs'/item['trial_id']/'result.json');cfg=d['config'];arm=cfg['arm']
  if plan['phase']=='measure':
   assert cfg['base_cache_dir']==preparation['sources']['full_base']['root']+'/cache/graphs'
   assert cfg['provenance']['harness_source']==FINAL_PIN
   base_graphs={p for p in preparation['sources']['full_base']['inventory'] if p.startswith('cache/graphs/')}
   target_graphs={p for p in preparation['sources']['full_target']['inventory'] if p.startswith('cache/graphs/')}
   assert len(base_graphs)==2 and len(target_graphs-base_graphs)==1
   assert d['live_base_graph_cache']==cfg['base_cache_dir']
  assert d['status']=='completed' and d['validity']=='valid' and d['native_gates_passed'] and d['cleanup_passed'] and not d['cleanup_errors']
  assert d['target']['tokens_equal_reference'] and d['target']['tokens']==d['reference']['tokens']
  assert d['base_generation']['tokens_equal_reference'] and d['base_generation']['tokens']==d['reference']['base_tokens']
  assert d['resident_weights_preserved'] and d['base_graphs_preserved'] and d['weight_buffers_before']==d['target']['weight_buffers_after']
  assert d['counts']['compilation_attempts']==0 and d['counts']['shared_uploads']==1 and d['counts']['native_calls']==(765 if arm.endswith('csi-active') else 510)
  assert d['target']['target_native_calls']==255 and d['counts']['native_output_devices']==['furiosa:'+str(d['assignment']['torch_ordinal'])]
  assert d['packages']['furiosa-torch']=='2026.4.0' and d['packages']['torch']=='2.10.0+cpu'
  assert len(d['assignment']['allocation_uuid'])==1 and d['assignment']['pod_uid']==result['assignment']['pod_uid']
  for name,record in d['source_files'].items():
   commit=PINS.get(cohort,FINAL_PIN) if name=='bench/ccgrid_resident_delivery.py' else 'db2dd2598daffb3d3f722007599b74d00ede61af'
   source=subprocess.check_output(['git','-C',str(repository),'show',commit+':'+name]);assert hashlib.sha256(source).hexdigest()==record['sha256'],name
  points=[d['events'][k]['perf_ns'] for k in ['demand','delivery_finished','hash_finished']]+[d['target']['generation_start_perf_ns'],d['target']['first_token_perf_ns']]
  assert points==sorted(points)
  assert abs(d['metrics']['demand_to_first_token_s']-(points[-1]-points[0])/1e9)<1e-9
  assert abs(sum(d['metrics'][k] for k in ['delivery_s','hash_validation_s','hash_to_generation_s','prompt_to_first_token_s'])-d['metrics']['demand_to_first_token_s'])<1e-8
  before=d['expected_inventory']['target'];after=d['delivered_inventory'];assert set(before)==set(after)
  assert all((before[k]['size'],before[k]['sha256'])==(after[k]['size'],after[k]['sha256']) for k in before)
  if arm!='graph-copy':
   proof=d['delivery_log_proof'];st=proof['pull_stats'];assert st['bytes']==sum(q['size'] for q in after.values()) and st['cache_bytes']==st['peer_bytes']==0
   if arm=='full-csi-rebase':assert proof['claimed_base'] and st['base']==cfg['base_ref'] and st['reused']>0 and d['claim_witness']['donor_removed'] and d['claim_witness']['inode_continuity_files']
   elif arm=='full-csi-no-rebase':assert not st.get('base') and not d['claim_witness']['donor_removed']
   elif arm=='graph-csi':assert not st.get('base') and proof['idle_donor_refusal'] and st['base_best']==224097 and not d['claim_witness']['donor_removed']
   elif arm.endswith('csi-active'):
    assert proof['busy_donor_refusal'] and d['active_base_preserved'] and d['active_base_recheck']['tokens_equal_reference'] and not st.get('base') and not st.get('local_base')
    if arm=='full-csi-active':assert d['live_checkpoint']==cfg['work_dir']+'/base-mount/checkpoint' and d['live_base_graph_cache']==cfg['work_dir']+'/base-mount/cache/graphs'
  else:assert not d.get('rpcs') and not d.get('donor_prepared_inventory')
 return dict(cohort=cohort,phase=plan['phase'],verified=True,trials=len(result['trials']),native_calls=sum(load(root/'runs'/q['trial_id']/'result.json')['counts']['native_calls'] for q in result['trials']))


def distribution(values):
 values=list(values);assert values and all(math.isfinite(x) and x>=0 for x in values)
 q=statistics.quantiles(values,method='inclusive') if len(values)>1 else [values[0]]*3
 return dict(n=len(values),values=values,median=statistics.median(values),q1=q[0],q3=q[2],iqr=q[2]-q[0])


def summarize(data_root,cohorts,repository=REPO):
 assert list(cohorts)==['r02','r03','r04','r05','r06'],'Use every frozen main cohort in declared order'
 arms=['graph-copy','graph-csi','full-csi-no-rebase','full-csi-rebase']
 groups={a:[] for a in arms};verified=[]
 for cohort in cohorts:
  v=verify_cohort(data_root,cohort,repository);assert v['phase']=='measure' and v['trials']==4
  verified.append(v);root=Path(data_root)/cohort;run=load(root/'runs'/(cohort+'-cohort.json'))
  for t in run['trials']:
   d=load(root/'runs'/t['trial_id']/'result.json')
   groups[d['config']['arm']].append(dict(cohort=cohort,trial_id=t['trial_id'],metrics=d['metrics'],
    pull_stats=d.get('delivery_log_proof',{}).get('pull_stats'),native_calls=d['counts']['native_calls']))
 result=dict(schema='resident-delivery-summary.v1',verified=True,cohorts=verified,arms={},
  runtime_baseline='db2dd2598daffb3d3f722007599b74d00ede61af',harness_source=FINAL_PIN,
  interval='demand through full-file SHA validation and reconstructed native first token',
  first_token='generation entry + returned native prompt TTFT; excludes remaining continuation',
  conditions='one card; same resident weights; base-only initial graph cache; first target use; empty chunk cache; no peers; alternating order; shared OS page cache uncontrolled',
  scope='POSIX delivery; graph-copy is colocated copy, not public production FXB or cloud baseline; full-file hash included; setup excluded and retained separately; CLI bytes is target logical size, not writes; no service load',
  aggregation='per-arm median and inclusive IQR of all five observations; do not add marginal component medians')
 for arm,rows in groups.items():
  assert len(rows)==5
  keys=set(rows[0]['metrics']);assert all(set(r['metrics'])==keys for r in rows)
  result['arms'][arm]=dict(samples=rows,metrics={k:distribution([r['metrics'][k] for r in rows]) for k in sorted(keys)})
 return result


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=DATA);p.add_argument('--cohorts',nargs='+',required=True);p.add_argument('--summary',action='store_true');p.add_argument('--output',type=Path)
 a=p.parse_args();r=summarize(a.data_root,a.cohorts) if a.summary else [verify_cohort(a.data_root,c) for c in a.cohorts]
 payload=json.dumps(r,indent=2)+'\n'
 if a.output:
  with a.output.open('x',encoding='utf-8') as f:f.write(payload)
 else:print(payload,end='')
