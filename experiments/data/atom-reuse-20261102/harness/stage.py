"""Freeze Git bytes and stage own source/receipts through the hub transport."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from control import rpc,ROOT

REPO=Path('C:/Users/USER/orca/scratch/rngd-pd-atom-reuse-20261102')
REMOTE='/home/gspark/workspace/codex_atom_reuse_20261102'


def upload(name,data):
    require_path=REMOTE+'/'+name
    digest=hashlib.sha256(data).hexdigest()
    for offset in range(0,len(data),48000):
        chunk=data[offset:offset+48000]
        code=f'''import pathlib,base64,json
r=pathlib.Path({REMOTE!r});r.mkdir(exist_ok=True)
assert r.resolve()==r and not r.is_symlink()
p=r/{name!r};p.parent.mkdir(parents=True,exist_ok=True)
assert p.resolve().is_relative_to(r) and not p.is_symlink()
b=base64.b64decode({base64.b64encode(chunk).decode()!r})
if p.exists() and p.stat().st_size>{offset}:
 with p.open('rb') as f:f.seek({offset});assert f.read(len(b))==b
else:
 assert (p.stat().st_size if p.exists() else 0)=={offset}
 with p.open('ab') as f:f.write(b)
print(json.dumps(dict(bytes=p.stat().st_size)))
'''
        rpc('rebellion',code,'upload-'+Path(name).name+'-'+str(offset))
    meta=rpc('rebellion',f'''import pathlib,hashlib,json
p=pathlib.Path({require_path!r});b=p.read_bytes()
assert len(b)=={len(data)} and hashlib.sha256(b).hexdigest()=={digest!r}
print(json.dumps(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))
''','verify-upload-'+Path(name).name)
    return meta


if __name__=='__main__':
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',head,'--','src','pyproject.toml',
        'bench/ccgrid_atom_reuse.py','experiments/data/atom-binding-20261030/harness/run_atom_b1.py'],cwd=REPO,text=True).splitlines()
    packed=io.BytesIO();files={}
    with tarfile.open(fileobj=packed,mode='w') as archive:
        for name in names:
            data=subprocess.check_output(['git','show',head+':'+name],cwd=REPO)
            files[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
    tar=gzip.compress(packed.getvalue(),mtime=0)
    (ROOT/'source.tar.gz').write_bytes(tar)
    manifest=dict(source_commit=head,runtime_base='57e610eec6b91c5303060a3820241b0f4dd0c0c7',files=files,
        archive_sha256=hashlib.sha256(tar).hexdigest(),source_identity='exact pinned Git blobs')
    (ROOT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    upload('source.tar.gz',tar)
    upload('inputs/source-manifest.json',(ROOT/'source-manifest.json').read_bytes())
    rpc('rebellion',f'''import pathlib,tarfile,json,hashlib
r=pathlib.Path({REMOTE!r});out=r/'source';out.mkdir(exist_ok=True)
with tarfile.open(r/'source.tar.gz','r:gz') as archive:
 for entry in archive:
  assert entry.isfile() and not pathlib.PurePosixPath(entry.name).is_absolute() and '..' not in pathlib.PurePosixPath(entry.name).parts
  p=out/entry.name;assert p.resolve().is_relative_to(out)
  b=archive.extractfile(entry).read();p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
manifest=json.loads((r/'inputs/source-manifest.json').read_bytes())
for name,sha in manifest['files'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
print(json.dumps(dict(source_commit=manifest['source_commit'],files=len(manifest['files']),verified=True)))
''','extract-source')
    prior=REPO/'experiments/data/atom-binding-20261030'
    kit=json.loads((prior/'runs/atom-b1-native-04/kit-report.json').read_bytes())['cases'][1]
    result=json.loads((prior/'runs/atom-b1-native-04/result.json').read_bytes())
    expected=dict(graph_sha256=result['graph']['sha256'],base=kit['base_files'],target=kit['target_files'],
        checkpoint=result['host_checkpoint'],source_commit=head,
        switchplan_sha256=hashlib.sha256((prior/'preparation/inputs/atom-target.switchplan.json').read_bytes()).hexdigest())
    (ROOT/'expected-inputs.json').write_text(json.dumps(expected,indent=2)+'\n',encoding='utf-8')
    upload('inputs/expected-inputs.json',(ROOT/'expected-inputs.json').read_bytes())
    upload('scripts/prepare_inputs.py',(ROOT/'prepare_inputs.py').read_bytes())
    print(json.dumps(dict(source_commit=head,source_files=len(files),archive_bytes=len(tar),staged=True)))
