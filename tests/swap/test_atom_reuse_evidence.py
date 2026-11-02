"""Reject misleading native records; CPU tests do not execute the vendor SDK."""
import copy
import importlib.util
import json
import gzip
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def verifier():
    path=Path(__file__).parents[2]/'experiments/verify_atom_reuse.py'
    spec=importlib.util.spec_from_file_location('verify_atom_reuse',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke():
    return json.loads((Path(__file__).parents[2]/'experiments/data/atom-reuse-20261102/runs/smoke-01/result.json').read_bytes())


def test_smoke_is_separate_and_cannot_become_main(verifier,smoke):
    assert verifier.verify_record(smoke,'smoke')['full']['transition_s']['n']==1
    with pytest.raises(ValueError):verifier.verify_record(smoke,'main')


@pytest.mark.parametrize('mutation',[
    lambda r:r['trials'].pop(),
    lambda r:r['trials'][1].update(parse_calls=1),
    lambda r:r['trials'][1].update(tokens=[[0]*8]),
    lambda r:r['trials'][1].update(host_object_retained=False),
    lambda r:r['trials'][1].update(old_runtime_closed=False),
    lambda r:r['trials'][1].update(transition_mode='apply'),
    lambda r:r['trials'][1].update(trigger_to_first_token_s=0),
    lambda r:r['trials'][1]['base'].update(run_completed=0),
    lambda r:r.update(compile_attempts=['unexpected']),
    lambda r:r['config'].update(source_commit='0'*40),
    lambda r:r['assignment'].update(exposed=['/dev/rbln0','/dev/rbln1']),
    lambda r:r['summary']['reuse']['transition_s'].update(median=0),
    lambda r:r.update(status='failed'),
])
def test_rejects_incomplete_or_falsely_fast_native_record(verifier,smoke,mutation):
    changed=copy.deepcopy(smoke);mutation(changed)
    with pytest.raises(ValueError):verifier.verify_record(changed,'smoke')


def test_coherently_rewritten_ttft_cannot_hide_native_prompt_time(verifier,smoke):
    changed=copy.deepcopy(smoke);t=changed['trials'][1]
    t['prompt_ttft_s']=0
    t['trigger_to_first_token_s']=(t['generate_call_ns']-t['trigger_ns'])/1e9
    for key in ('prompt_ttft_s','trigger_to_first_token_s'):
        changed['summary']['reuse'][key]=verifier.stat([t[key]])
    with pytest.raises(ValueError):verifier.verify_record(changed,'smoke')


def test_empty_exact_export_cannot_attest_missing_evidence(verifier,tmp_path):
    directory=tmp_path/'runs/smoke-01';directory.mkdir(parents=True)
    raw=gzip.compress(json.dumps(dict(files=[])).encode())
    (directory/'exact.json.gz').write_bytes(raw)
    (directory/'exact-manifest.json').write_text(json.dumps(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),files=0)))
    with pytest.raises(ValueError):verifier.exact_export(tmp_path,'smoke-01')


def test_two_application_containers_cannot_attest_one_card(verifier,smoke,tmp_path):
    source=Path(__file__).parents[2]/'experiments/data/atom-reuse-20261102/runs/smoke-01'
    directory=tmp_path/'runs/smoke-01';directory.mkdir(parents=True)
    for name in ('create.json','pod-evidence.json','cleanup.json'):
        (directory/name).write_bytes((source/name).read_bytes())
    evidence=json.loads((directory/'pod-evidence.json').read_bytes())
    extra=copy.deepcopy(evidence['pod']['spec']['containers'][0]);extra['name']='second-app'
    evidence['pod']['spec']['containers'].append(extra)
    (directory/'pod-evidence.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError):verifier.verify_pod(tmp_path,'smoke-01',smoke)


def test_complete_published_package_reproduces_five_pair_summary(verifier):
    root=Path(__file__).parents[2]/'experiments/data/atom-reuse-20261102'
    summary=verifier.verify(root)
    assert summary['verified'] is True
    assert summary['main']['full']['trigger_to_first_token_s']['n']==5
    assert summary['main']['reuse']['trigger_to_first_token_s']['n']==5
    assert summary['smoke']['full']['trigger_to_first_token_s']['n']==1
