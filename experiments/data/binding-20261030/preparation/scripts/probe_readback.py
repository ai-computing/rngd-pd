"""Actual public Torch H2D/strided writes/D2H and retained restoration."""
import hashlib,json,os,time,traceback
from pathlib import Path
run=Path(os.environ['CCGRID_RUN']);run.mkdir(parents=True,exist_ok=True)
record=dict(status='running',utc=time.time(),pod_uid=os.environ['POD_UID'],api_scope='torch.Tensor.to/cpu/copy_ and public furiosa.torch device/stream functions')
def save():
    (run/'result.json').write_text(json.dumps(record,indent=2)+'\n')
try:
    import torch,furiosa.torch as ft
    from rngd_pd import devices
    from importlib.metadata import version
    assert devices.allocated_cards()==[0] and devices.resolve_chip(-1)==0
    assert Path('/run/swap/rngd').read_text().strip()=='npu0'
    record.update(physical_host_index=0,torch_ordinal=0,sdk=version('furiosa-torch'),torch=torch.__version__)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    ft.set_fusion(8);ft.set_mesh_kind(ft.MeshKind.Mesh(2))
    sync=lambda:ft.current_stream('furiosa:0').synchronize()
    raw=lambda t:t.contiguous().view(torch.uint8).numpy().tobytes()
    original=torch.arange(24,dtype=torch.float32).reshape(2,3,4).to(torch.bfloat16)
    device=original.to('furiosa:0');sync()
    pointer=device.data_ptr();before=device.cpu();assert raw(before)==raw(original)
    old=before[0,:,1:3].t().contiguous().clone()
    target=torch.tensor([[101,102,103],[201,202,203]],dtype=torch.bfloat16)
    staged=target.t().contiguous().to('furiosa:0');sync()
    device[0,:,1:3].copy_(staged);sync()
    after=device.cpu();assert raw(after[0,:,1:3].t())==raw(target)
    assert torch.equal(after[1],original[1]) and torch.equal(after[0,:,0],original[0,:,0]) and torch.equal(after[0,:,3],original[0,:,3])
    journal=old.clone();assert journal.device.type=='cpu'
    device[0,:,1:3].copy_(journal.t().contiguous().to('furiosa:0'));sync()
    restored=device.cpu();assert raw(restored)==raw(original) and device.data_ptr()==pointer
    record.update(status='completed',projection_strided_write=True,actual_device_readback=True,retained_actual_readback_journal=True,restored_exactly=True,pointer_retained=True,
        before_sha256=hashlib.sha256(raw(before)).hexdigest(),target_sha256=hashlib.sha256(raw(target)).hexdigest(),after_sha256=hashlib.sha256(raw(after)).hexdigest(),restored_sha256=hashlib.sha256(raw(restored)).hexdigest(),completed_utc=time.time())
    del device,staged;sync();save()
except BaseException:
    record.update(status='failed',traceback=traceback.format_exc(),completed_utc=time.time());save();raise
finally:
    owner=run.stat()
    for p in run.glob('*'):os.chown(p,owner.st_uid,owner.st_gid)
