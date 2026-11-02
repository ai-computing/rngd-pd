"""Off-card preparation: verify existing immutable inputs; never modify old roots."""
import hashlib, json, pathlib, time
root = pathlib.Path('/home/gspark/workspace/codex_ccgrid_b2_20261102')
root.mkdir(exist_ok=True)
inputs = root / 'inputs'
inputs.mkdir(exist_ok=True)
old = pathlib.Path('/home/gspark/workspace/codex_binding_conformance_20261030')
native = pathlib.Path('/home/gspark/workspace/codex_ccgrid_npu_20261030')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
mapping = {
    'fixtures.json': old / 'inputs/rngd-interior-fixtures.json',
    'references.json': old / 'runs/rngd-b2-native-02/work/refsactual.json',
    'interior.switchplan.json': old / 'inputs/interior.switchplan.json',
    'prior-kit-report.json': old / 'runs/rngd-b2-native-02/work/kit-report.json',
}
for name, source in mapping.items():
    content = source.read_bytes()
    target = inputs / name
    if target.exists():
        assert target.read_bytes() == content
    else:
        target.write_bytes(content)
provider = json.loads((inputs / 'fixtures.json').read_text())
case = next(c for c in provider['cases'] if c['name'] == 'native-interior-projection')
prior = next(c for c in json.loads((inputs / 'prior-kit-report.json').read_text())['cases'] if c['name'] == case['name'])
files = []
began = time.monotonic()
for role in ('base', 'target'):
    directory = pathlib.Path(case[role + '_dir'])
    for name, expected in sorted(prior[role + '_files'].items()):
        path = directory / name
        stat = path.stat()
        actual = digest(path)
        assert stat.st_size == expected['size'] and actual == expected['sha256'], (role, name, actual)
        files.append(dict(path=str(path), bytes=stat.st_size, sha256=actual, role=role, mtime_ns=stat.st_mtime_ns))
cache = native / 'cache/graphs'
for path in sorted(cache.rglob('*.runnable')) + sorted(mapping.values()) + sorted(inputs.iterdir()):
    if path.is_file() and path.name != 'input-manifest.json':
        files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), role='cache-or-input', mtime_ns=path.stat().st_mtime_ns))
assert any(p['path'].endswith('.runnable') for p in files)
plan = pathlib.Path(case['switchplan'])
canonical_plan = json.dumps(json.loads(plan.read_text()), sort_keys=True, separators=(',', ':')).encode('utf-8')
assert hashlib.sha256(canonical_plan).hexdigest() == prior['switchplan_sha256'], 'Prior kit hashes canonical JSON, not source bytes'
record = dict(schema='ccgrid-b2-inputs-v1', prepared_utc=time.time(), off_card=True,
              source_commit='4482faef87dabef8f586c6e7dff382b889cca115',
              case=case, cache=str(cache), files=files, full_hash_elapsed_s=time.monotonic()-began,
              scope='Full checkpoint hashes verified against prior native kit before allocation; old roots read-only')
out = inputs / 'input-manifest.json'
assert not out.exists(), 'Existing preparation must be adopted, not overwritten'
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(path=str(out), sha256=digest(out), files=len(files), full_hash_elapsed_s=record['full_hash_elapsed_s'])))
