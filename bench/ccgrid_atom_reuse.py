"""Native ATOM public parsed-model reuse; run only in an allocated one-card Job."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback


def arm_order(pairs):
    for pair in range(pairs):
        for arm in (('full','reuse') if pair % 2 == 0 else ('reuse','full')):
            yield pair,arm


def stats(values):
    q1,_,q3=statistics.quantiles(values,method='inclusive') if len(values)>1 else [values[0]]*3
    return dict(n=len(values),median=statistics.median(values),q1=q1,q3=q3,iqr=q3-q1,values=values)


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_trial(row,reference,target):
    arm=row['arm']
    require(arm in ('full','reuse') and row['status']=='completed','invalid trial status/arm')
    reused=arm=='reuse'
    require(row['manifest']==target and row['tokens']==reference,'wrong target identity/output')
    require(row['parse_calls']==int(not reused) and row['create_calls']==1,'wrong native parse/create count')
    require(row['compile_attempts']==0 and row['run_completed']==15,'compiler/native execution gate')
    require(row['compiled_reused'] is reused and row['parsed_before'] is reused,'reuse flag differs')
    require(row['old_runtime_closed'] is True and (not reused or row['host_object_retained'] is True),'lifetime gate')
    require(row['native_runtime_type'].startswith('rebel.sync_runtime.')
            and row['compiled_type']=='rebel.compiled_model.RBLNCompiledModel','not a native public runtime')
    times=[row[k] for k in ('trigger_ns','transition_return_ns','generate_call_ns','generate_return_ns')]
    require(all(type(t) is int for t in times) and times==sorted(times),'clock ordering')
    for key in ('transition_s','trigger_to_first_token_s','admission_s','parse_s','create_runtime_s','prompt_ttft_s'):
        require(type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key]>=0,'invalid clock '+key)
    require(math.isclose(row['transition_s'],(times[1]-times[0])/1e9,abs_tol=1e-9),'transition clock formula')
    expected=(times[2]-times[0])/1e9+row['prompt_ttft_s']
    require(math.isclose(row['trigger_to_first_token_s'],expected,abs_tol=1e-9),'first-token clock formula')
    require(row['transition_s']>0 and expected <= (times[3]-times[0])/1e9+1e-6,'first token after return')


def summarize(rows,pairs,reference,target):
    require([(r['pair'],r['arm']) for r in rows]==list(arm_order(pairs)),'incomplete or reordered matrix')
    for row in rows:validate_trial(row,reference,target)
    keys=('transition_s','trigger_to_first_token_s','admission_s','parse_s','create_runtime_s','prompt_ttft_s')
    return {arm:{key:stats([r[key] for r in rows if r['arm']==arm]) for key in keys}
            for arm in ('full','reuse')}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def save(path,document):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(document,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def helper_module():
    path=Path(__file__).resolve().parents[1]/'experiments/data/atom-binding-20261030/harness/run_atom_b1.py'
    spec=importlib.util.spec_from_file_location('preserved_atom_public_helpers',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper,path


def generate(session,row,prompts,n_new,helper):
    session.rt=helper.CountedRuntime(session.rt,row)
    row['generate_call_ns']=time.perf_counter_ns()
    tokens,ttft=session.generate(prompts,n_new)
    row['generate_return_ns']=time.perf_counter_ns()
    row.update(tokens=tokens,prompt_ttft_s=ttft,
               compiled_type=type(session.compiled).__module__+'.'+type(session.compiled).__qualname__)
    require(row['run_completed']==len(prompts[0])+n_new-1,'native call count differs')
    return tokens


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_bytes())
    require(config['phase'] in ('smoke','main') and config['pairs']==(1 if config['phase']=='smoke' else 5),'invalid phase/pairs')
    run=Path(config['run_dir']);run.mkdir(exist_ok=True,parents=True)
    require(not (run/'result.json').exists(),'attempt directory already has results')
    record=dict(schema='ccgrid-atom-reuse-run.v1',status='running',validity='not_checked',config=config,
                config_sha256=sha(args.config),harness_sha256=sha(__file__),trials=[],milestones=[],
                pod_uid=os.environ.get('POD_UID'),started_utc=time.time(),npu_requested=1)
    owner=fresh=None
    def phase(name):
        record['phase']=name;record['milestones'].append(dict(phase=name,utc=time.time()));save(run/'result.json',record)
    try:
        phase('source_identity')
        source=json.loads(Path(config['source_manifest']).read_bytes())
        require(source['source_commit']==config['source_commit'],'source revision differs')
        root=Path(config['source_root']).resolve(strict=True)
        for name,expected in source['files'].items():
            path=(root/name).resolve(strict=True)
            require(path.is_relative_to(root) and sha(path)==expected,'staged source hash differs: '+name)
        record['source_manifest_sha256']=sha(config['source_manifest'])
        record['source_files']=source['files']
        phase('imports')
        import torch
        import rebel
        from importlib.metadata import version
        from rngd_pd.swap.atom import Tables
        from rngd_pd.swap.atom_binding import AtomBinding,load_atom_session
        import rngd_pd.swap.atom_binding as binding_module
        require(Path(binding_module.__file__).resolve().is_relative_to(root),'wrong imported owner source')
        helper,helper_path=helper_module()
        torch.set_num_threads(4);torch.set_grad_enabled(False)
        require(os.environ.get('RBLN_WEIGHT_FREE','0') in ('','0','off'),'experimental weight-free mode forbidden')
        assignment=json.loads(Path('/run/swap/atom.json').read_bytes())
        exposed=sorted(str(p) for p in Path('/dev').glob('rbln*'))
        physical=helper.allocated_device_id(assignment,exposed,allow_broad=False)
        command=['/host-bin/rbln-smi']
        diagnostic=subprocess.run(command,capture_output=True,text=True,timeout=15)
        require(diagnostic.returncode==0 and len(diagnostic.stdout)<16384,'public device mapping failed')
        device=helper.sdk_device_id_from_smi(diagnostic.stdout,physical,allow_broad=False)
        record.update(assignment=dict(allocated=assignment,exposed=exposed,physical_id=physical,sdk_id=device),
            public_smi=dict(argv=command,stdout=diagnostic.stdout,stderr=diagnostic.stderr),
            versions=dict(torch=torch.__version__,rebel=rebel.__version__,compiler=version('rebel-compiler')),
            helper_sha256=sha(helper_path),public_native_apis=['RBLNCompiledModel','create_runtime','SyncRuntime.run'],
            vendor_internal_apis_used=[],checkpoint=config['checkpoint'])
        require(record['versions']==config['expected_versions'],'SDK versions differ from frozen protocol')
        helper.install_compile_guards(rebel,torch,record)
        native_class=rebel.RBLNCompiledModel
        original_create=native_class.create_runtime
        counters=dict(parse_calls=0,create_calls=0)
        def observed_parse(*args,**kwargs):
            counters['parse_calls']+=1
            return native_class(*args,**kwargs)
        def observed_create(instance,*args,**kwargs):
            counters['create_calls']+=1
            return original_create(instance,*args,**kwargs)
        rebel.RBLNCompiledModel=observed_parse
        native_class.create_runtime=observed_create
        phase('input_receipt')
        receipt=json.loads(Path(config['input_receipt']).read_bytes())
        require(receipt['verified'] is True and receipt['graph_sha256']==config['graph_sha256'],'unverified inputs')
        for role in ('base','target'):
            directory=Path(config[role])
            require({p.name for p in directory.iterdir()}==set(receipt[role]),'artifact inventory differs')
            for name,entry in receipt[role].items():
                path=directory/name
                require(path.is_file() and not path.is_symlink() and path.stat().st_size==entry['bytes'],'artifact size/type differs')
                if name!='session.rbln':require(sha(path)==entry['sha256'],'small artifact hash differs')
        switch=json.loads(Path(config['switchplan']).read_bytes())
        require(sha(config['switchplan'])==receipt['switchplan_sha256'],'verified switch plan differs')
        record['input_receipt_sha256']=sha(config['input_receipt'])
        record['switchplan_sha256']=sha(config['switchplan'])
        phase('host_tables')
        start=time.perf_counter();tables=Tables(config['checkpoint'])
        record['host_tables_s']=time.perf_counter()-start
        prompts,n_new=[[1,42,256,1024,7,999,2048,13]],8
        record.update(prompts=prompts,n_new=n_new)
        phase('fresh_target')
        reference_row={};record['fresh_reference']=reference_row
        fresh=load_atom_session(config['target'],device=device,tables=tables)
        reference=generate(fresh,reference_row,prompts,n_new,helper)
        fresh.close();fresh=None;gc.collect()
        record['fresh_reference']=dict(reference_row,closed_before_matrix=True)
        class ObservedOwner(AtomBinding):
            def _admission(self,artifact,manifest):
                started=time.perf_counter()
                evidence=super()._admission(artifact,manifest)
                self.admission_s=time.perf_counter()-started
                return evidence
        phase('matrix')
        for pair,arm in arm_order(config['pairs']):
            # The explicit factory calls the same real public loader, without cache reuse.
            kwargs=dict(device=device,tables=tables)
            if arm=='full':kwargs['factory']=lambda path:load_atom_session(path,device=device,tables=tables)
            owner=ObservedOwner(**kwargs)
            trial=dict(pair=pair,arm=arm,status='running')
            record['trials'].append(trial);save(run/'result.json',record)
            owner.load(config['base'],switch['base_manifest'])
            with owner.request() as session:
                require(session.active.B==1 and session.active.S==64 and session.nl==32,'wrong native shape')
                base_row={};trial['base']=base_row
                base_tokens=generate(session,base_row,prompts,n_new,helper)
                require(base_tokens==reference,'base graph does not match same-graph fresh target')
            trial['base']=base_row
            old_session=owner.engine;old_compiled_id=id(old_session.compiled)
            before=dict(counters)
            trial['trigger_ns']=time.perf_counter_ns()
            result=owner.transition(switch,config['target'])
            trial['transition_return_ns']=time.perf_counter_ns()
            with owner.request() as session:
                generate(session,trial,prompts,n_new,helper)
                trial.update(parsed_before=session.timings['parsed_before'],parse_s=session.timings['parse_s'],
                    create_runtime_s=session.timings['create_runtime_s'],host_object_retained=arm=='reuse' and id(session.compiled)==old_compiled_id,
                    sdk_allocation_estimate=session.compiled.get_total_device_alloc())
            trial.update(status='completed',manifest=result['resident']['manifest'],transition_mode=result['mode'],
                compiled_reused=owner.graph_evidence()['compiled_reused'],admission_s=owner.admission_s,
                old_runtime_closed=old_session.rt is None and old_session.compiled is None,
                compile_attempts=len(record['compile_attempts']),
                **{key:counters[key]-before[key] for key in counters})
            require(owner.graph_evidence()['sha256']==config['graph_sha256'],'admitted graph SHA differs')
            trial['transition_s']=(trial['transition_return_ns']-trial['trigger_ns'])/1e9
            trial['trigger_to_first_token_s']=(trial['generate_call_ns']-trial['trigger_ns'])/1e9+trial['prompt_ttft_s']
            validate_trial(trial,reference,switch['manifest'])
            owner.close();owner=None;gc.collect()
            save(run/('pair-%02d-%s.json'%(pair,arm)),trial);save(run/'result.json',record)
        record['summary']=summarize(record['trials'],config['pairs'],reference,switch['manifest'])
        require(counters==dict(parse_calls=3*config['pairs']+1,create_calls=4*config['pairs']+1),'whole-run native counters differ')
        require(not record['compile_attempts'],'unexpected compilation')
        record.update(native_counters=counters,native_run_completed=15*(4*config['pairs']+1),
            status='completed',validity='valid',native_evidence=True)
        phase('completed')
        return 0
    except BaseException:
        record.update(status='failed',validity='invalid',error=traceback.format_exc())
        if 'counters' in locals():record['native_counters']=dict(counters)
        raise
    finally:
        try:
            if owner is not None:owner.close()
            if fresh is not None:fresh.close()
            record['owners_closed']=True
        except BaseException:
            record.update(status='failed',validity='invalid',cleanup_error=traceback.format_exc())
            raise
        finally:
            record['finished_utc']=time.time();save(run/'result.json',record)


if __name__=='__main__':
    raise SystemExit(main())
