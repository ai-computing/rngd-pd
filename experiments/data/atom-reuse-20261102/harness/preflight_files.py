from pathlib import Path
import json
import os
root=Path('/home/gspark/workspace/codex_binding_conformance_20261030')
paths=['inputs/atom-fixtures.json','atom-artifacts/base','atom-artifacts/target','scripts/atom-b1-native-04.py',
       'runs/atom-b1-native-04/result.json','atom-runtime-source/src/rngd_pd/swap/atom_binding.py']
result=[]
for name in paths:
    p=root/name
    result.append(dict(path=str(p),exists=p.exists(),entries=[dict(name=x.name,bytes=x.stat().st_size,symlink=x.is_symlink())
        for x in sorted(p.iterdir())] if p.is_dir() else None))
provider=root/'inputs/atom-fixtures.json'
if provider.is_file():result.append(dict(provider=json.loads(provider.read_bytes())))
stats=os.statvfs('/home/gspark/workspace')
print(json.dumps(dict(paths=result,workspace_free_bytes=stats.f_bavail*stats.f_frsize)))
