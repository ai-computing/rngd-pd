"""Serial fresh-process E7 smoke then frozen n5 matrix; one allocated RNGD."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from rngd_pd import devices

root=Path(os.environ['CCGRID_ROOT']);source=Path(os.environ['CCGRID_SOURCE'])
run=Path(os.environ['CCGRID_RUN']);plan=json.loads((run/'matrix-plan.json').read_bytes())
work=run/'work';work.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record=dict(schema='ccgrid-e7-operational-v1',status='starting',utc=time.time(),pod_uid=os.environ['POD_UID'],
    adapter_sha256=sha(__file__),plan_sha256=sha(run/'matrix-plan.json'),samples=[])
def save():
    (run/'operational.json').write_text(json.dumps(record,indent=2)+'\n')
    if record['status'] in ['completed','failed']:
        (run/'result.json').write_text(json.dumps(dict(schema='ccgrid-e7-matrix-v1',experiment='E7',status=record['status'],plan=plan,operational=record),indent=2)+'\n')
save()
try:
    git=['git','-c','safe.directory='+str(source),'-C',str(source)]
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(git+['status','--porcelain'],text=True).strip()
    assert commit==plan['benchmark_source_commit'] and not dirty
    assert sha(__file__)==plan['adapter_sha256']
    record.update(source_commit=commit,source_dirty=dirty)
    cards=devices.allocated_cards();assert cards==[0]
    assert Path('/run/swap/rngd').read_text().strip()=='npu0'
    chip=devices.resolve_chip(-1);assert chip==0
    record['assignment']=dict(physical_host_index=0,torch_ordinal=chip,visible_cards=devices.visible_cards(),allocation_uuid=devices.allocated_uuids(os.environ['POD_UID'],'run'))
    for item in plan['trials']:
        record['status']='running';save()
        out=work/(item['trial_id']+'.json');ref=run/'inputs'/item['reference_file'];ranges=run/'inputs'/item['ranges_file']
        assert sha(ref)==item['reference_sha256'] and sha(ranges)==item['ranges_sha256']
        expected=json.loads(ref.read_bytes())
        argv=[os.sys.executable,str(source/'bench/ccgrid_e7.py'),'sample','--ckpt',plan['base_checkpoint'],
            '--target',plan['targets'][item['fixture']],'--ranges',str(ranges),'--cache',plan['cache'],
            '--chip',str(chip),'--reference',str(ref),'--variant','B4-S256-g2','--host-workers','2','--intra-op','4',
            '--mode',item['mode'],'--out',str(out)]
        print(json.dumps(dict(event='trial_started',trial_id=item['trial_id'],utc=time.time())),flush=True)
        with (work/(item['trial_id']+'.log')).open('x') as log:
            process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        row=json.loads(out.read_bytes()) if out.exists() else None
        evidence=dict(trial_id=item['trial_id'],exit=process.returncode,raw_sha256=sha(out) if out.exists() else None,status=row and row.get('status'))
        record['samples'].append(evidence);save()
        assert process.returncode==0 and row['status']=='ok',evidence
        assert row['metadata']['source']['commit']==commit and row['metadata']['source']['dirty']==''
        assert row['base_tokens']==expected['base_tokens'] and row['base_tokens_equal_reference'] is True
        assert row['tokens']==expected['tokens'] and row['tokens_equal_reference'] is True
        assert row['reference']['sha256']==item['reference_sha256'] and row['ranges']['sha256']==item['ranges_sha256']
        assert not row['plan']['header_changes'] and not row['plan']['removed']
        natives=[row] if item['mode']!='restart' else [row['workers'][p]['result'] for p in ['base','target']]
        for native in natives:
            assert native['sdk']=='2026.4.0' and native['metadata']['packages']['torch']=='2.10.0+cpu'
            assert native['expected_native_device']==dict(type='furiosa',index=chip)
            counts=native['native_counts'];assert counts['compilation_attempts']==0
            assert counts['native_calls']==counts['native_output_checks']==(510 if item['mode']!='restart' else 255)
            assert native['native_outputs_verified'] is True
            assert 'close_error' not in native
        if item['mode']=='restart':
            assert row['workers']['base']['parent_observed_exit_perf_ns']<=row['workers']['target']['parent_launch_perf_ns']
        assert row['logical_device_written_bytes']>0
        print(json.dumps(dict(event='trial_valid',trial_id=item['trial_id'],headline=item['headline'],utc=time.time())),flush=True)
    record.update(status='completed',finished_utc=time.time());save()
except BaseException:
    record.update(status='failed',traceback=traceback.format_exc(),finished_utc=time.time());save();raise
finally:
    owner=run.stat()
    for directory,dirs,files in os.walk(run,followlinks=False):
        for name in dirs+files:os.chown(Path(directory)/name,owner.st_uid,owner.st_gid,follow_symlinks=False)
        os.chown(directory,owner.st_uid,owner.st_gid,follow_symlinks=False)
