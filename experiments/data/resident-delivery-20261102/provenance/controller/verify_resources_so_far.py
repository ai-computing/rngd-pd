import json
from pathlib import Path
r=Path('.');rows=[]
for p in sorted(r.glob('cdx-resident-*-cleanup.json')):
 d=json.loads(p.read_bytes());name=d['name'];create=json.loads((r/(name+'-create.json')).read_bytes());j=json.loads((r/(name+'-job.json')).read_bytes())
 assert d['job_absent'] and d['pods_absent'] and d['job_uid']==create['job_uid']
 spec=j['spec'];pod=spec['template']['spec'];req=[]
 for c in pod.get('containers',[])+pod.get('initContainers',[]):req.append(int(c.get('resources',{}).get('limits',{}).get('furiosa.ai/rngd',0)))
 assert max(req or [0]) in (0,1) and spec['activeDeadlineSeconds']>0 and spec['ttlSecondsAfterFinished']>0
 assert all(q['phase'] in ('Succeeded','Failed') for q in d['observations'])
 rows.append(dict(name=name,uid=d['job_uid'],card_limit=max(req or [0]),pod_phases=[q['phase'] for q in d['observations']],cleaned=True))
print(json.dumps(rows,indent=2))
