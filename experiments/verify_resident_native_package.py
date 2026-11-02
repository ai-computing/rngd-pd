"""Verify context capacity, exact operational evidence and native UID cleanup."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_bytes())


def verify(root, repository=REPO):
    root = Path(root)
    manifest = load(root / 'manifest.json')
    assert len({q['path'] for q in manifest['files']}) == len(manifest['files'])
    for row in manifest['files']:
        path = (root / row['path']).resolve()
        assert path.is_relative_to(root.resolve())
        data = path.read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    packed = (root / 'remote-evidence.json.gz').read_bytes()
    assert len(packed) == manifest['remote_archive']['bytes']
    assert hashlib.sha256(packed).hexdigest() == manifest['remote_archive']['sha256']
    for row in json.loads(gzip.decompress(packed)):
        path = (root / 'remote' / row['path']).resolve()
        assert path.is_relative_to((root / 'remote').resolve())
        assert path.read_bytes() == base64.b64decode(row['base64'], validate=True)

    controller = root / 'controller'
    jobs = ['s01', 's02', 'r02', 'r03', 'r04', 'r05', 'r06', 'capacity01']
    cleaned = {}
    for cohort in jobs:
        name = 'cdx-resident-native-' + cohort
        job = load(controller / (name + '-job.json'))
        create = load(controller / (name + '-create.json'))
        cleanup = load(controller / (name + '-cleanup.json'))
        assert create['job_uid'] == cleanup['job_uid'] and cleanup['job_absent'] and cleanup['pods_absent']
        assert job['metadata']['name'] == name and job['spec']['activeDeadlineSeconds'] > 0 and job['spec']['ttlSecondsAfterFinished'] > 0
        pod = job['spec']['template']['spec']
        for container in pod['containers'] + pod.get('initContainers', []):
            assert int(container['resources']['limits']['furiosa.ai/rngd']) == 1
        assert len(cleanup['observations']) == 1
        observed = cleanup['observations'][0]
        assert observed['phase'] == 'Succeeded'
        assert all(c['restartCount'] == 0 and c['state']['terminated']['exitCode'] == 0 for c in observed['containers'])
        cleaned[cohort] = observed['uid']
    state = load(controller / 'main-controller-v2-state.json')
    assert state['status'] == 'completed' and state['current_job'] is None
    assert [c['cohort'] for c in state['cohorts']] == ['r02', 'r03', 'r04', 'r05', 'r06']
    assert all(c['status'] == 'verified' for c in state['cohorts'])

    capacity = root / 'remote/capacity/c01'
    sequence = load(capacity / 'sequence.json')
    assert sequence['status'] == 'completed' and sequence['pod_uid'] == cleaned['capacity01']
    assert [r['role'] for r in sequence['roles']] == ['fresh', 'resident']
    documents = {role: load(capacity / (role + '.json')) for role in ('fresh', 'resident')}
    for role, document in documents.items():
        assert document['status'] == 'completed' and document['validity'] == 'valid' and document['owner_closed']
        assert document['assignment']['pod_uid'] == sequence['pod_uid'] and len(document['assignment']['allocation_uuid']) == 1
        counts = document['counts']
        assert counts['compilation_attempts'] == 0 and counts['shared_uploads'] == 1
        assert counts['native_calls'] == (4488 if role == 'fresh' else 4743)
        assert counts['native_output_devices'] == ['furiosa:' + str(document['assignment']['torch_ordinal'])]
        for name, record in document['source_files'].items():
            pin = ('56b8cd4506f5fbbba9d22c0627b71080d4d58fe8' if name == 'bench/ccgrid_context_capacity.py'
                   else '829573e41959a4b0f6daa82e4862f1584367c347' if name == 'bench/ccgrid_resident_delivery.py'
                   else 'db2dd2598daffb3d3f722007599b74d00ede61af')
            data = subprocess.check_output(['git', '-C', str(repository), 'show', pin + ':' + name])
            assert hashlib.sha256(data).hexdigest() == record['sha256'], name
    fresh, resident = documents['fresh'], documents['resident']
    assert fresh['prompts'] == resident['prompts'] and len(fresh['prompts']) == 4
    assert {len(p) for p in fresh['prompts']} == {257} and {len(t) for t in fresh['tokens']} == {8}
    assert resident['base_rejected_overflow'] and resident['resident_buffers_preserved'] and resident['base_programs_preserved']
    assert fresh['tokens'] == resident['tokens'] and resident['same_target_tokens_equal']
    assert resident['fresh_reference_file']['sha256'] == hashlib.sha256((capacity / 'fresh.json').read_bytes()).hexdigest()
    return dict(verified=True, operational_files=len(manifest['files']), native_jobs_uid_cleaned=len(jobs),
                capacity_native_calls=9231, capacity_scope='n1 P257/O8 fresh-target parity, resident identity and S256 rejection; no latency distribution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'experiments/data/resident-delivery-20261102/provenance')
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
