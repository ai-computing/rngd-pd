"""Offline public embedded-weight ATOM artifact; runtime creation forbidden."""
import hashlib,json,os,resource,subprocess,time,traceback
from pathlib import Path
r=Path('/home/gspark/workspace/codex_binding_conformance_20261030')
out=r/'atom-public-session';out.mkdir(exist_ok=False)
record=dict(status='running',utc=time.time(),scope='CPU-only public rebel.compile_from_torch; create_runtime forbidden',
    checkpoint='/home/gspark/workspace/codex_swap_benchmark_20261023/fixtures/resident-base',
    graph=str(out/'session.rbln'),B=1,S=64,layers=32)
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):d.update(b)
    return d.hexdigest()
def save():(r/'evidence/atom-public-compile.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    assert not os.environ.get('RBLN_WEIGHT_FREE')
    import torch,rebel
    from importlib.metadata import version
    from rngd_pd.swap.atom import build_session
    from rngd_pd.swap.variants import Variant
    record.update(torch=torch.__version__,rebel=version('rebel-compiler'),source_commit=subprocess.check_output(['git','-C',os.environ['CCGRID_SOURCE'],'rev-parse','HEAD'],text=True).strip(),source_files={})
    for n in ['src/rngd_pd/swap/atom.py','src/rngd_pd/vendors/rbln_decode_batch.py','src/rngd_pd/engines/model.py']:
        record['source_files'][n]=sha(Path(os.environ['CCGRID_SOURCE'])/n)
    torch.set_num_threads(4)
    def forbidden(*a,**kw):raise RuntimeError('Runtime creation forbidden in CPU preparation')
    rebel.RBLNCompiledModel.create_runtime=forbidden
    started=time.monotonic()
    build_session(record['checkpoint'],record['graph'],[Variant(1,64,0)],layers=32,npu='RBLN-CA22')
    path=Path(record['graph']);assert path.is_file() and Path(str(path)+'.json').is_file()
    record.update(status='completed',compile_s=time.monotonic()-started,graph_sha256=sha(path),graph_bytes=path.stat().st_size,sidecar=json.loads(Path(str(path)+'.json').read_bytes()),sidecar_sha256=sha(str(path)+'.json'),runtime_creations=0,finished_utc=time.time(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save()
except BaseException:
    record.update(status='failed',traceback=traceback.format_exc(),finished_utc=time.time());save();raise
