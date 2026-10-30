import importlib.util
from pathlib import Path

import pytest


def module():
    p = Path(__file__).resolve().parents[2] / 'bench/ccgrid_summary.py'
    spec = importlib.util.spec_from_file_location('ccgrid_summary', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def samples(values, cache='empty'):
    return [dict(trial_id=f'{cache}-{i}', experiment='E6', arm='resident',
                 cache_state=cache, status='completed', validity='valid',
                 tokens_equal_reference=True, metrics={'ttft_s':v})
            for i,v in enumerate(values)]


def test_inclusive_iqr_and_raw_values():
    groups = module().summarize(samples([100, 4, 1, 3, 2]))
    g = groups[0]
    assert g['headline_eligible'] and g['n_valid'] == g['n_total'] == 5
    assert g['metrics']['ttft_s'] == dict(n=5, values=[100,4,1,3,2],
        median=3, q1=2, q3=4, iqr=2, min=1, max=100)


def test_preserves_failure_and_does_not_count_it_as_repeat():
    rows=samples([1,2,3,4,5])
    rows[4].update(status='failed', validity='invalid', error='native error')
    g=module().summarize(rows)[0]
    assert not g['headline_eligible'] and g['n_valid']==4 and g['n_total']==5
    assert g['excluded']==[dict(trial_id='empty-4',status='failed',validity='invalid',tokens_equal_reference=True)]
    assert len(g['metrics']['ttft_s']['values'])==4


def test_cache_states_do_not_mix_and_short_group_is_not_headline():
    groups=module().summarize(samples([1,2,3,4,5])+samples([100],cache='prepopulated'))
    assert len(groups)==2
    assert [g['headline_eligible'] for g in groups]==[True,False]


def test_missing_metric_prevents_headline_even_with_five_valid_trials():
    rows=samples([1,2,3,4,5]); rows[-1]['metrics']={}
    g=module().summarize(rows)[0]
    assert g['n_valid']==5 and not g['headline_eligible']
    assert g['metrics']['ttft_s']['n']==4


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True])
def test_invalid_durations_fail_instead_of_changing_summary(value):
    with pytest.raises(ValueError,match='finite nonnegative'):
        module().summarize(samples([1,2,3,4,value]))


def test_duplicate_trial_ids_rejected():
    rows=samples([1,2,3,4,5]); rows[-1]['trial_id']=rows[0]['trial_id']
    with pytest.raises(ValueError,match='duplicate'):
        module().summarize(rows)


def test_token_mismatch_excluded_even_when_execution_completed():
    rows=samples([1,2,3,4,5]); rows[-1]['tokens_equal_reference']=False
    g=module().summarize(rows)[0]
    assert g['n_valid']==4 and not g['headline_eligible']


def test_e8_first_use_and_prepared_remain_separate_conditions():
    rows=[]
    for state,value in [('first-use',100),('prepared',1)]:
        part=samples([value]*5,cache=state)
        for item in part:
            item.update(experiment='E8',cache_state=None,graph_state=state)
        rows.extend(part)
    groups=module().summarize(rows)
    assert len(groups)==2
    assert [g['metrics']['ttft_s']['median'] for g in groups]==[100,1]
    assert all(g['n_valid']==5 for g in groups)


def test_e7_modes_remain_separate_without_an_arm_normalization():
    rows=[]
    for mode in ['inplace','replace','restart']:
        part=samples([1]*5,cache=mode)
        for item in part:
            item.update(experiment='E7',cache_state=None,arm=None,mode=mode)
        rows.extend(part)
    assert len(module().summarize(rows))==3
