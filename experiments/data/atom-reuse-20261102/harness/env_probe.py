"""CPU-only SDK/source import probe. Runtime creation is never called."""
import json
from pathlib import Path
import os
import time
import traceback

run=Path('/home/gspark/workspace/codex_atom_reuse_20261102/runs/env-01')
record=dict(status='running',npu_requested=0,started_utc=time.time())
try:
    assert not list(Path('/dev').glob('rbln*'))
    import torch
    import rebel
    from importlib.metadata import version
    from rngd_pd.swap.atom_binding import AtomBinding
    original=rebel.RBLNCompiledModel.create_runtime
    rebel.RBLNCompiledModel.create_runtime=original
    record.update(status='completed',versions=dict(torch=torch.__version__,rebel=rebel.__version__,compiler=version('rebel-compiler')),
        owner_source=__import__('inspect').getfile(AtomBinding),rebel_source=rebel.__file__,
        runtime_created=False,compiled_model_created=False)
except BaseException:
    record.update(status='failed',error=traceback.format_exc());raise
finally:
    record['finished_utc']=time.time()
    temporary=run/'result.json.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n')
    os.replace(temporary,run/'result.json')
