"""Real file/output checks; the unavailable vendor SDK alone is substituted."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).parents[2] / 'bench/inspect_atom_artifacts.py'
    spec = importlib.util.spec_from_file_location('atom_inspect', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    (root / 'a.rbln').write_bytes(b'abc')
    (root / 'b.rbln').write_bytes(b'def')
    rows = [dict(path='a.rbln', bytes=3, sha256='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'),
            dict(path='b.rbln', bytes=3, sha256='cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34')]
    manifest = tmp_path / 'inputs.json'
    manifest.write_text(json.dumps(dict(schema='rngd-pd.atom-inspect-inputs.v1', files=rows)))
    return root, manifest, tmp_path / 'output'


def sdk(monkeypatch, inspect):
    class Compiled:
        def __init__(self, *args, **kwargs):
            raise AssertionError('compiled object must not be constructed')

    Compiled.inspect = staticmethod(inspect)
    monkeypatch.setitem(sys.modules, 'rebel', types.SimpleNamespace(RBLNCompiledModel=Compiled))


def test_complete_inputs_produce_exact_metadata_and_durable_result(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    def inspect(path):
        assert Path(path) in (root / 'a.rbln', root / 'b.rbln')
        return {'compiler_version': 'example', 'inputs': [{'shape': [1, 128, 4096]}], 'unknown_vendor_field': {'x': None}}
    sdk(monkeypatch, inspect)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'completed'
    assert json.loads((output / 'result.json').read_text()) == result
    assert [row['path'] for row in result['files']] == ['a.rbln', 'b.rbln']
    assert all(row['status'] == 'completed' for row in result['files'])
    assert result['files'][0]['metadata'] == {'compiler_version': 'example', 'inputs': [{'shape': [1, 128, 4096]}], 'unknown_vendor_field': {'x': None}}
    assert result['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert result['files'][0]['observed_sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert result['runtime_api_called'] is False and result['coverage'] == 'unknown'


def test_wrong_hash_prevents_inspection_but_later_input_is_preserved(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    (root / 'a.rbln').write_bytes(b'bad')
    def inspect(path):
        assert Path(path).name == 'b.rbln'
        return {'name': 'b'}
    sdk(monkeypatch, inspect)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed'
    assert result['files'][0]['status'] == 'failed'
    assert result['files'][0]['native_api_called'] is False
    assert result['files'][0]['error']['phase'] == 'input_hash'
    assert result['files'][1]['metadata'] == {'name': 'b'}


def test_native_failure_does_not_erase_earlier_or_later_attempts(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    def inspect(path):
        if Path(path).name == 'a.rbln':
            raise RuntimeError('native format refused')
        return {'name': 'b'}
    sdk(monkeypatch, inspect)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed'
    assert result['files'][0]['native_api_called'] is True
    assert result['files'][0]['error'] == dict(phase='inspect', type='RuntimeError', message='native format refused')
    assert result['files'][1]['status'] == 'completed'
    assert json.loads((output / 'result.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('bad', [None, [], {'bad': object()}, {'nan': float('nan')}, {1: 'not a string key'}, {'tuple': (1, 2)}])
def test_non_json_metadata_is_an_explicit_failure(probe, inputs, monkeypatch, bad):
    root, manifest, output = inputs
    sdk(monkeypatch, lambda path: bad)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed'
    assert all(row['error']['phase'] == 'metadata' for row in result['files'])
    assert all('metadata' not in row for row in result['files'])


@pytest.mark.parametrize('bad_path', ['../a.rbln', '/a.rbln', 'C:/a.rbln', 'x\\a.rbln', 'a.txt', './a.rbln', 'x//a.rbln'])
def test_invalid_manifest_path_is_rejected_before_sdk_import(probe, inputs, bad_path, monkeypatch):
    root, manifest, output = inputs
    document = json.loads(manifest.read_text())
    document['files'][0]['path'] = bad_path
    manifest.write_text(json.dumps(document))
    monkeypatch.setitem(sys.modules, 'rebel', None)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed' and result['files'] == []
    assert result['error']['phase'] == 'manifest'


def test_duplicate_inputs_are_not_counted_twice(probe, inputs):
    root, manifest, output = inputs
    document = json.loads(manifest.read_text())
    document['files'][1] = document['files'][0].copy()
    manifest.write_text(json.dumps(document))
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed' and result['error']['phase'] == 'manifest'


def test_symlink_cannot_escape_input_root(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    outside = root.parent / 'outside.rbln'
    outside.write_bytes(b'abc')
    (root / 'a.rbln').unlink()
    try:
        (root / 'a.rbln').symlink_to(outside)
    except OSError:
        pytest.skip('Windows symlink privilege unavailable; exercised on Linux CI')
    sdk(monkeypatch, lambda path: {'name': 'b'})
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed' and result['files'][0]['native_api_called'] is False


def test_sdk_absence_is_not_an_artifact_capability_refusal(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    monkeypatch.setitem(sys.modules, 'rebel', None)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed' and result['files'] == []
    assert result['error']['phase'] == 'sdk'
    assert result['coverage'] == 'unknown'


def test_input_modified_during_inspection_invalidates_metadata(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    def inspect(path):
        Path(path).write_bytes(b'changed')
        return {'would_be_valid': True}
    sdk(monkeypatch, inspect)
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed'
    assert all(row['error']['phase'] == 'input_stability' for row in result['files'])
    assert all(row['metadata'] == {'would_be_valid': True} for row in result['files'])


def test_existing_output_is_never_overwritten(probe, inputs):
    root, manifest, output = inputs
    output.mkdir()
    (output / 'result.json').write_bytes(b'old evidence')
    with pytest.raises(FileExistsError):
        probe.run_probe(manifest, root, output)
    assert (output / 'result.json').read_bytes() == b'old evidence'


def test_output_cannot_modify_readonly_input_tree(probe, inputs):
    root, manifest, output = inputs
    with pytest.raises(ValueError, match='input'):
        probe.run_probe(manifest, root, root / 'output')
    assert not (root / 'output').exists()


def test_cli_records_bad_manifest_and_returns_nonzero(probe, inputs):
    root, manifest, output = inputs
    manifest.write_bytes(b'{not json')
    completed = subprocess.run([sys.executable, probe.__file__, '--manifest', str(manifest), '--input-root', str(root), '--output', str(output)], capture_output=True)
    assert completed.returncode == 1
    result = json.loads((output / 'result.json').read_text())
    assert result['error']['phase'] == 'manifest'
    assert result['manifest_sha256'] == hashlib.sha256(b'{not json').hexdigest()
    assert (output / 'inputs.json').read_bytes() == b'{not json'


def test_interrupted_snapshot_preserves_previous_evidence(probe, tmp_path, monkeypatch):
    probe.save(tmp_path, {'status': 'running', 'files': [{'path': 'a.rbln'}]})
    before = (tmp_path / 'result.json').read_bytes()
    def fail_replace(source, destination):
        raise OSError('simulated storage interruption')
    monkeypatch.setattr(probe.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='storage interruption'):
        probe.save(tmp_path, {'status': 'completed', 'files': []})
    assert (tmp_path / 'result.json').read_bytes() == before
    assert not (tmp_path / '.result.tmp').exists()


@pytest.mark.parametrize('field,value', [('bytes', True), ('bytes', -1), ('sha256', 'f' * 63), ('sha256', 'F' * 64)])
def test_malformed_identity_is_rejected_before_native_call(probe, inputs, field, value):
    root, manifest, output = inputs
    document = json.loads(manifest.read_text())
    document['files'][0][field] = value
    manifest.write_text(json.dumps(document))
    result = probe.run_probe(manifest, root, output)
    assert result['status'] == 'failed' and result['files'] == []
    assert result['error']['phase'] == 'manifest'


def test_interrupted_native_call_preserves_previous_and_current_attempt(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    def inspect(path):
        if Path(path).name == 'a.rbln':
            return {'name': 'a'}
        snapshot = json.loads((output / 'result.json').read_text())
        assert len(snapshot['files']) == 2
        assert snapshot['files'][0]['status'] == 'completed'
        current = snapshot['files'][1]
        assert current['status'] == 'running' and current['phase'] == 'inspect_pending'
        assert current['observed_sha256'] == 'cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34'
        assert current['native_api_called'] is None
        raise KeyboardInterrupt('interrupted native call')
    sdk(monkeypatch, inspect)
    with pytest.raises(KeyboardInterrupt, match='native call'):
        probe.run_probe(manifest, root, output)
    snapshot = json.loads((output / 'result.json').read_text())
    assert snapshot['status'] == 'running' and len(snapshot['files']) == 2
    assert snapshot['files'][0]['metadata'] == {'name': 'a'}
    assert snapshot['files'][1]['status'] == 'running'


def test_interrupted_hash_preserves_current_input_attempt(probe, inputs, monkeypatch):
    root, manifest, output = inputs
    real_open = Path.open
    class InterruptedRead:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, size):
            snapshot = json.loads((output / 'result.json').read_text())
            assert len(snapshot['files']) == 1
            assert snapshot['files'][0]['path'] == 'a.rbln'
            assert snapshot['files'][0]['phase'] == 'input_hash'
            assert snapshot['files'][0]['native_api_called'] is False
            raise KeyboardInterrupt('interrupted hash')
    def open_with_interrupt(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        return InterruptedRead(stream) if path == root / 'a.rbln' and args == ('rb',) else stream
    monkeypatch.setattr(Path, 'open', open_with_interrupt)
    sdk(monkeypatch, lambda path: {'name': 'unused'})
    with pytest.raises(KeyboardInterrupt, match='hash'):
        probe.run_probe(manifest, root, output)
    assert json.loads((output / 'result.json').read_text())['files'][0]['status'] == 'running'
