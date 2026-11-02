import argparse,base64,gzip,hashlib,json
from pathlib import Path
from control import rpc,ROOT
p=argparse.ArgumentParser();p.add_argument('trial',choices=['env-01','smoke-01','main-01','inputs']);a=p.parse_args();trial=ROOT/'runs'/a.trial;trial.mkdir(parents=True,exist_ok=True)
code=f"""import base64,gzip,hashlib,json
from pathlib import Path
root=Path('/home/gspark/workspace/codex_atom_reuse_20261102')
run=root/'inputs' if {a.trial!r}=='inputs' else root/'runs'/{a.trial!r}
paths=sorted(run.glob('*.json'))
assert all(p.stat().st_size < 1000000 for p in paths)
files=[]
for p in paths:
 raw=p.read_bytes();files.append(dict(path=str(p.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
packed=gzip.compress(json.dumps(dict(files=files),sort_keys=True).encode(),mtime=0)
out=root/'runs'/({a.trial!r}+'-exact.json.gz')
assert not out.exists() or out.read_bytes()==packed
out.write_bytes(packed)
print(json.dumps(dict(bytes=len(packed),sha256=hashlib.sha256(packed).hexdigest(),files=len(files),base64=base64.b64encode(packed).decode())))
"""
r=rpc('rebellion',code,'exact-export-'+a.trial);raw=base64.b64decode(r.pop('base64'))
assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
(trial/'exact.json.gz').write_bytes(raw);(trial/'exact-manifest.json').write_text(json.dumps(r,indent=2))
packed=json.loads(gzip.decompress(raw))
for f in packed['files']:
 data=base64.b64decode(f['base64']);assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256']
 dest=ROOT/'exact'/f['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print(json.dumps(r))
