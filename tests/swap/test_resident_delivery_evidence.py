"""Mutation tests use copied real smoke evidence; no model payload/device needed."""
import importlib.util
import base64
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('resident_evidence', REPO / 'experiments/verify_resident_delivery.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
DATA = REPO / 'experiments/data/resident-delivery-20261102'


def test_real_smoke_and_active_checkpoint_are_verified():
    assert evidence.verify_cohort(DATA, 's01', REPO)['native_calls'] == 2805
    assert evidence.verify_cohort(DATA, 's02', REPO)['native_calls'] == 765


def test_exact_archive_and_files_detect_modified_bytes(tmp_path):
    shutil.copytree(DATA / 's02', tmp_path / 's02')
    manifest = json.loads((tmp_path / 's02/export-manifest.json').read_bytes())
    path = tmp_path / 's02' / manifest['files'][0]['path']
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises((AssertionError, ValueError)):
        evidence.verify_cohort(tmp_path, 's02', REPO)


def test_archive_corruption_is_not_accepted(tmp_path):
    shutil.copytree(DATA / 's02', tmp_path / 's02')
    archive = tmp_path / 's02/small-evidence.json.gz'
    archive.write_bytes(archive.read_bytes()[:-1])
    with pytest.raises((AssertionError, ValueError)):
        evidence.verify_cohort(tmp_path, 's02', REPO)


@pytest.mark.parametrize('failure', ['native_calls', 'busy_base', 'first_token'])
def test_semantic_failure_rejected_even_with_consistent_export_hashes(tmp_path, failure):
    shutil.copytree(DATA / 's02', tmp_path / 's02')
    root = tmp_path / 's02'
    archive_path = root / 'small-evidence.json.gz'
    archive = json.loads(gzip.decompress(archive_path.read_bytes()))
    entry = next(q for q in archive['files'] if q['path'] == 'runs/s02-full-csi-active/result.json')
    document = json.loads(base64.b64decode(entry['base64']))
    if failure == 'native_calls':
        document['counts']['native_calls'] = 0
    elif failure == 'busy_base':
        document['delivery_log_proof']['busy_donor_refusal'] = False
    else:
        document['metrics']['demand_to_first_token_s'] = 0.1
    payload = json.dumps(document).encode()
    (root / entry['path']).write_bytes(payload)
    entry.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), base64=base64.b64encode(payload).decode())
    packed = gzip.compress(json.dumps(archive).encode())
    archive_path.write_bytes(packed)
    manifest_path = root / 'export-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['archive'].update(bytes=len(packed), sha256=hashlib.sha256(packed).hexdigest())
    manifest['files'] = [{k: v for k, v in q.items() if k != 'base64'} for q in archive['files']]
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises((AssertionError, ValueError)):
        evidence.verify_cohort(tmp_path, 's02', REPO)


def test_smoke_cannot_supply_five_measurement_repetitions():
    with pytest.raises((AssertionError, ValueError)):
        evidence.summarize(DATA, ['s01', 's02'], REPO)


def test_iqr_is_inclusive_and_preserves_all_observations():
    result = evidence.distribution([9, 2, 3, 4, 100])
    assert result == dict(n=5, values=[9, 2, 3, 4, 100], median=4, q1=3, q3=9, iqr=6)


def test_nonfinite_measurement_has_no_statistical_summary():
    with pytest.raises((AssertionError, ValueError)):
        evidence.distribution([0.1, float('nan'), 0.2, 0.3, 0.4])
