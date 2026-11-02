import copy
import json
from pathlib import Path
from stage import REMOTE,REPO
from control import ROOT
from cluster import NS

old=json.loads((REPO/'experiments/data/atom-binding-20261030/runs/atom-b1-native-04/job.json').read_bytes())
source=json.loads((ROOT/'source-manifest.json').read_bytes())
for trial in ('env-01','smoke-01','main-01'):
    run=REMOTE+'/runs/'+trial
    spec=copy.deepcopy(old['spec']['template']['spec'])
    spec['automountServiceAccountToken']=False
    spec['volumes'] += [dict(name='own-work',hostPath=dict(path=REMOTE,type='Directory')),
        dict(name='own-source',hostPath=dict(path=REMOTE+'/source',type='Directory')),
        dict(name='python-executable',hostPath=dict(path='/usr/bin/python3.10',type='File')),
        dict(name='python-stdlib',hostPath=dict(path='/usr/lib/python3.10',type='Directory'))]
    container=spec['containers'][0]
    for mount in container['volumeMounts']:
        if mount['name']=='binding-work':mount['readOnly']=True
    container['volumeMounts'] += [dict(name='own-work',mountPath=REMOTE),
        dict(name='own-source',mountPath=REMOTE+'/source',readOnly=True),
        dict(name='python-executable',mountPath='/usr/bin/python3.10',readOnly=True),
        dict(name='python-executable',mountPath='/usr/bin/python3',readOnly=True),
        dict(name='python-stdlib',mountPath='/usr/lib/python3.10',readOnly=True)]
    changes=dict(CCGRID_ROOT=REMOTE,CCGRID_SOURCE=REMOTE+'/source',CCGRID_RUN=run,
        PYTHONPATH=REMOTE+'/source/src:/home/gspark/workspace/01_pd_disag/experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/lib/python3.10/site-packages:/home/gspark/.local/lib/python3.10/site-packages',
        TMPDIR=run+'/tmp',XDG_CACHE_HOME=run+'/xdg')
    for entry in container['env']:
        if entry['name'] in changes:entry['value']=changes[entry['name']]
    python='/home/gspark/workspace/01_pd_disag/experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python'
    if trial=='env-01':
        spec.pop('initContainers',None)
        for category in ('requests','limits'):container['resources'][category].pop('rebellions.ai/ATOM')
        container['resources']['limits']['memory']='16Gi';container['resources']['requests']['memory']='4Gi'
        container['command']=['bash','-c','set -eu\nexec '+python+' '+REMOTE+'/scripts/env_probe.py\n']
    else:
        container['command']=['bash','-c','set -eu\nexec '+python+' '+REMOTE+'/source/bench/ccgrid_atom_reuse.py --config '+run+'/config.json\n']
    labels=dict(owner='codex',experiment='atom-reuse-20261102',trial=trial)
    job=dict(apiVersion='batch/v1',kind='Job',metadata=dict(name='cdx-atom-reuse-'+trial,namespace=NS,labels=labels),
        spec=dict(backoffLimit=0,ttlSecondsAfterFinished=600,activeDeadlineSeconds=300 if trial=='env-01' else 900 if trial=='smoke-01' else 1800,
            template=dict(metadata=dict(labels=labels),spec=spec)))
    path=ROOT/'runs'/trial;path.mkdir(parents=True,exist_ok=True)
    (path/'job.json').write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
    if trial!='env-01':
        config=dict(schema='ccgrid-atom-reuse-config.v1',phase='smoke' if trial=='smoke-01' else 'main',pairs=1 if trial=='smoke-01' else 5,
            run_dir=run,source_root=REMOTE+'/source',source_manifest=REMOTE+'/inputs/source-manifest.json',source_commit=source['source_commit'],
            base='/home/gspark/workspace/codex_binding_conformance_20261030/atom-artifacts/base',
            target='/home/gspark/workspace/codex_binding_conformance_20261030/atom-artifacts/target',
            checkpoint='/home/gspark/workspace/codex_swap_benchmark_20261023/fixtures/resident-base',
            graph_sha256='02b4eebd765c1acd3412517ea3804fe0ba38ddf2aa020a88db853245f2908ebe',
            input_receipt=REMOTE+'/inputs/input-receipt.json',switchplan=REMOTE+'/inputs/switchplan.json',
            expected_versions=dict(torch='2.11.0+cpu',rebel='0.11.0',compiler='0.11.0'))
        (path/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(jobs=['env-01','smoke-01','main-01'],source_commit=source['source_commit'])))
