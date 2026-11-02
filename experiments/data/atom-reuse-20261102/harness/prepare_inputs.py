"""Off-card static/hash preparation. Imports no NPU SDK and changes no input."""
import hashlib
import json
from pathlib import Path
import os
import time
import traceback

ROOT=Path('/home/gspark/workspace/codex_atom_reuse_20261102')
PRIOR=Path('/home/gspark/workspace/codex_binding_conformance_20261030')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):digest.update(b)
    return digest.hexdigest()


if __name__=='__main__':
    expected=json.loads((ROOT/'inputs/expected-inputs.json').read_bytes())
    output=ROOT/'inputs/input-receipt.json'
    assert not output.exists()
    record=dict(status='running',verified=False,started_utc=time.time(),npu_requested=0)
    try:
        for role in ('base','target'):
            directory=PRIOR/'atom-artifacts'/role
            assert {p.name for p in directory.iterdir()}==set(expected[role])
            rows={}
            for name,reference in expected[role].items():
                path=directory/name;assert path.is_file() and not path.is_symlink()
                digest=sha(path);assert digest==reference['sha256'] and path.stat().st_size==reference['size']
                rows[name]=dict(bytes=path.stat().st_size,sha256=digest,inode=path.stat().st_ino)
            record[role]=rows
        checkpoint=Path(expected['checkpoint']['path'])
        reference_files=expected['checkpoint']['files']
        assert {p.name for p in checkpoint.iterdir() if p.is_file()}=={r['path'] for r in reference_files}
        files=[]
        for entry in reference_files:
            p=checkpoint/entry['path'];assert p.is_file() and not p.is_symlink()
            digest=sha(p);assert p.stat().st_size==entry['bytes'] and digest==entry['sha256']
            files.append(dict(path=entry['path'],bytes=p.stat().st_size,sha256=digest))
        record['checkpoint']=dict(path=str(checkpoint),files=files)
        plan=(PRIOR/'inputs/atom-target.switchplan.json').read_bytes()
        assert hashlib.sha256(plan).hexdigest()==expected['switchplan_sha256']
        (ROOT/'inputs/switchplan.json').write_bytes(plan)
        record.update(status='completed',verified=True,graph_sha256=expected['graph_sha256'],
                      switchplan_sha256=expected['switchplan_sha256'],source_commit=expected['source_commit'])
    except BaseException:
        record.update(status='failed',error=traceback.format_exc());raise
    finally:
        record['finished_utc']=time.time()
        temporary=output.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n')
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,output)
