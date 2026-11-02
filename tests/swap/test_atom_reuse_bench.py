"""CPU checks of measurement gates, not evidence of native execution."""
import importlib.util
from pathlib import Path
import copy

import pytest


@pytest.fixture
def bench():
    path=Path(__file__).parents[2]/'bench/ccgrid_atom_reuse.py'
    assert path.exists(), 'ATOM native reuse harness is not implemented'
    spec=importlib.util.spec_from_file_location('atom_reuse_bench',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(arm):
    return dict(arm=arm,pair=0,status='completed',manifest='target',
        parse_calls=int(arm=='full'),create_calls=1,compile_attempts=0,
        compiled_reused=arm=='reuse',parsed_before=arm=='reuse',old_runtime_closed=True,
        host_object_retained=arm=='reuse',native_runtime_type='rebel.sync_runtime.SyncRuntime',
        compiled_type='rebel.compiled_model.RBLNCompiledModel',tokens=[[1,2]],run_completed=15,
        trigger_ns=1000000000,transition_return_ns=2000000000,generate_call_ns=2100000000,
        generate_return_ns=2600000000,prompt_ttft_s=.2,transition_s=1.,
        trigger_to_first_token_s=1.3,admission_s=.5,parse_s=.1,create_runtime_s=.2)


def test_pair_order_and_quartiles_are_explicit(bench):
    assert list(bench.arm_order(3))==[(0,'full'),(0,'reuse'),(1,'reuse'),(1,'full'),(2,'full'),(2,'reuse')]
    assert bench.stats([1,2,3,4,9])==dict(n=5,median=3,q1=2,q3=4,iqr=2,values=[1,2,3,4,9])


@pytest.mark.parametrize('arm',['full','reuse'])
def test_valid_trial_uses_same_target_and_actual_native_call_gates(bench,arm):
    bench.validate_trial(row(arm),[[1,2]],'target')


@pytest.mark.parametrize('field,value',[('parse_calls',1),('create_calls',0),('compile_attempts',1),
    ('old_runtime_closed',False),('host_object_retained',False),('tokens',[[9,9]]),
    ('run_completed',0),('manifest','wrong'),('trigger_to_first_token_s',.2),
    ('native_runtime_type','fake.Runtime'),('status','failed')])
def test_bad_trial_is_never_accepted(bench,field,value):
    trial=row('reuse');trial[field]=value
    with pytest.raises(ValueError):bench.validate_trial(trial,[[1,2]],'target')


def test_partial_or_misordered_matrix_cannot_report_n5(bench):
    rows=[]
    for pair,arm in bench.arm_order(5):
        trial=row(arm);trial['pair']=pair;rows.append(trial)
    result=bench.summarize(rows,5,[[1,2]],'target')
    assert result['full']['transition_s']['n']==5
    with pytest.raises(ValueError):bench.summarize(rows[:-1],5,[[1,2]],'target')
    swapped=copy.deepcopy(rows);swapped[0],swapped[1]=swapped[1],swapped[0]
    with pytest.raises(ValueError):bench.summarize(swapped,5,[[1,2]],'target')


def test_failed_snapshot_write_preserves_previous_complete_record(bench,tmp_path,monkeypatch):
    path=tmp_path/'result.json'
    bench.save(path,{'status':'running','complete_trials':1})
    before=path.read_bytes()
    def broken(document,stream,**kwargs):
        stream.write('{"partial":')
        raise OSError('write interrupted')
    monkeypatch.setattr(bench.json,'dump',broken)
    with pytest.raises(OSError,match='interrupted'):
        bench.save(path,{'status':'running','complete_trials':2})
    assert path.read_bytes()==before
