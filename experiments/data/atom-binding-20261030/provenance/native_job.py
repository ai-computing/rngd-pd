"""Root-only, one-card, UID-guarded native Job launch/inspection/export."""
import argparse,base64,json
from control import c, ROOT, stage

p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status','collect']);p.add_argument('vendor',choices=['rngd','atom']);p.add_argument('trial');a=p.parse_args()
name='cdx-'+a.trial;run=c.REMOTE+'/runs/'+a.trial;dest=ROOT/'runs'/a.trial
if a.action=='start':
    source=c.REMOTE+('/runtime-source' if a.vendor=='rngd' else '/atom-runtime-source')
    if a.vendor=='atom':
        site='/home/gspark/.local/lib/python3.10/site-packages'
        venv='/home/gspark/workspace/01_pd_disag/experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv'
        c.rpc('rebellion',f"""import pathlib,json,subprocess
p=pathlib.Path({source!r})
if not p.exists():
 subprocess.run(['git','clone','--single-branch','--branch','exp/37-atom-binding','https://github.com/ai-computing/rngd-pd.git',str(p)],capture_output=True,check=True,timeout=70)
 subprocess.run(['git','-C',str(p),'checkout','--detach','fbfdb8bb9aa86750f144fa4ead276fe17f6db537'],capture_output=True,check=True)
head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip();assert head=='fbfdb8bb9aa86750f144fa4ead276fe17f6db537'
print(json.dumps(dict(head=head)))
""",'freeze-atom-source-01')
    script='run_rngd_b2.py' if a.vendor=='rngd' else 'run_atom_b1.py'
    remote_script=c.REMOTE+'/scripts/'+a.trial+'.py'
    stage(ROOT/script,remote_script)
    c.rpc('rebellion',f"""import pathlib,json,os
p=pathlib.Path({run!r});assert not p.exists();p.mkdir(parents=True);(p/'tmp').mkdir()
print(json.dumps(dict(run=str(p),uid=p.stat().st_uid,gid=p.stat().st_gid)))
""",'precreate-'+a.trial)
    j=json.loads((ROOT.parent/'ccgrid-npu-20261030/e7-refs-job.json').read_bytes())
    j['metadata']['name']=name;j['spec']['activeDeadlineSeconds']=1800
    for m in [j['metadata'],j['spec']['template']['metadata']]:m['labels']['experiment']=a.vendor+'-binding-conformance'
    spec=j['spec']['template']['spec'];container=spec['containers'][0]
    py='/home/gspark/venvs/furiosa-2026.4/bin/python' if a.vendor=='rngd' else '/home/gspark/workspace/01_pd_disag/experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv/bin/python'
    command=container['command'][-1]
    command=command[:command.index('exec ')]+'exec '+py+' '+remote_script+'\n'
    container['command'][-1]=command
    env={e['name']:e for e in container['env']}
    values=dict(CCGRID_ROOT=c.REMOTE,CCGRID_RUN=run,CCGRID_SOURCE=source,PYTHONPATH=source+'/src',TMPDIR=run+'/tmp',XDG_CACHE_HOME=run+'/xdg',BINDING_KIT=c.REMOTE+'/conformance-source/conformance/binding/conformance.py',BINDING_FIXTURES=c.REMOTE+'/inputs/'+a.vendor+'-fixtures.json',BINDING_REFERENCES=c.REMOTE+'/inputs/references.json',BINDING_CACHE='/home/gspark/workspace/codex_ccgrid_npu_20261030/cache/graphs',ATOM_CHECKPOINT='/home/gspark/workspace/codex_swap_benchmark_20261023/fixtures/resident-base',ATOM_REPEAT_N='5',BINDING_KIT_TIMEOUT='1400')
    for k,v in values.items():env[k]=dict(name=k,value=v)
    if a.vendor=='rngd' and a.trial.endswith('02'):
        env['BINDING_FIXTURES']['value']=c.REMOTE+'/inputs/rngd-interior-fixtures.json'
    container['env']=list(env.values())
    for m in container['volumeMounts']:
        if m['name']=='work':m['readOnly']=True
    extras=[('binding-work',c.REMOTE,False),('binding-source',source,True),('binding-kit',c.REMOTE+'/conformance-source',True)]
    extra_types={}
    if a.vendor=='atom':
        for e in container['env']:
            if e['name']=='PYTHONPATH':e['value']=source+'/src:'+venv+'/lib/python3.10/site-packages:'+site
        for r in [container['resources'],spec['initContainers'][0]['resources']]:
            for kind in ['requests','limits']:
                del r[kind]['furiosa.ai/rngd'];r[kind]['rebellions.ai/ATOM']='1'
        container['securityContext']={'privileged':True}
        container['env'].append(dict(name='ATOM_ALLOW_BROAD_VISIBILITY',value='1'))
        extras.append(('atom-sdk','/home/gspark/workspace/01_pd_disag/experiments/native_pd_2026_4/p2p_dma/devkit/torch-venv',True))
        extras.append(('atom-local',site,True))
        extras.append(('host-udev','/run/udev',True))
        if a.trial.endswith(('03','04')):
            container['securityContext']={'privileged':False}
            for e in container['env']:
                if e['name']=='ATOM_ALLOW_BROAD_VISIBILITY':e['value']='0'
            for i,n in enumerate(['librbln-thunk.so','librbln-thunk.so.3','librbln-thunk.so.3.0.0','librbln-ml.so']):
                vol='atom-runtime-lib-'+str(i);extras.append((vol,'/usr/lib/'+n,True));extra_types[vol]='File'
        init=spec['initContainers'][0];init['name']='assigned-atom'
        init['command']=['bash','-c',"set -eu\nfiles=(/dev/rbln*); [ -e \"${files[0]}\" ]; printf '%s\\n' \"${files[@]}\" | awk 'BEGIN { printf \"[\" } { if(NR>1)printf \",\"; printf \"\\\"%s\\\"\",$0 } END { print \"]\" }' > /run/swap/atom.json\ncat /run/swap/atom.json\n"]
    for vol,path,ro in extras:
        spec['volumes'].append(dict(name=vol,hostPath=dict(path=path,type=extra_types.get(vol,'Directory'))))
        container['volumeMounts'].append(dict(name=vol,mountPath=path,readOnly=ro))
    key='furiosa.ai/rngd' if a.vendor=='rngd' else 'rebellions.ai/ATOM'
    assert container['resources']['requests'][key]==container['resources']['limits'][key]=='1'
    assert not dest.exists();dest.mkdir(parents=True)
    c.save(dest/'job.json',j)
    blob=base64.b64encode(json.dumps(j).encode()).decode()
    code=c.KPRE+f"""
ps=json.loads(subprocess.check_output(k+['get','pods','-A','-o','json'],text=True))['items']
assert not [p for p in ps if p['metadata']['name'].startswith('cdx-') and p.get('status',{{}}).get('phase') not in ['Succeeded','Failed']]
assert get('job',{name!r}) is None
j=json.loads(subprocess.check_output(k+['create','-f','-','-o','json'],input=base64.b64decode({blob!r}).decode(),text=True))
print(json.dumps(dict(uid=j['metadata']['uid'],created=j['metadata']['creationTimestamp'])))
"""
    out=c.rpc('dell_0',code,'create-'+a.trial);c.save(dest/'create.json',out);print(json.dumps(out))
else:
    status=c.inspect_job(name);c.save(dest/'latest-api.json',status)
    print(json.dumps(status))
    if a.action=='collect':
        assert status['job'] and status['job']['uid']==json.loads((dest/'create.json').read_bytes())['uid']
        assert len(status['pods'])==1 and status['pods'][0]['phase'] in ['Succeeded','Failed']
        c.save(dest/'terminal-api.json',status)
        try:
            path,result=c.collect(a.trial);print(json.dumps(result))
        finally:
            c.cleanup_job(name,status,a.trial)
