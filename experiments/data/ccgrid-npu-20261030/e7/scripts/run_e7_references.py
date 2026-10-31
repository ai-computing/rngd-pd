"""Two fresh native target references, only inside one allocated post-E6 Job.

The existing f5fa133 native engine is the independent fresh-load baseline.
No cross-shape comparison is used for these same-B4 target fixtures.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

root = Path(os.environ['CCGRID_ROOT'])
run = Path(os.environ['CCGRID_RUN'])
work = run / 'work'
work.mkdir(exist_ok=False)
record = dict(schema='ccgrid-e7-fresh-references-v1', status='running',
    utc=time.time(), pod_uid=os.environ['POD_UID'], cases=[],
    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def save():
    data = json.dumps(record, indent=2) + '\n'
    (run / 'operational.json').write_text(data)
    if record['status'] in ['completed', 'failed']:
        (run / 'result.json').write_text(data)


save()
try:
    for role in ['sparse', 'lm']:
        output = work / (role + '-B4-S256-g2.json')
        checkpoint = '<private-path>' + role + '-new'
        argv = [os.sys.executable, str(root / 'scripts/native_reference.py'),
            '--variant', 'B4-S256-g2', '--checkpoint', checkpoint,
            '--prompt-offset', '0', '--out', str(output)]
        with (work / (role + '.log')).open('x') as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        row = json.loads(output.read_bytes()) if output.exists() else None
        record['cases'].append(dict(role=role, argv=argv, exit=process.returncode,
            raw_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None))
        save()
        assert process.returncode == 0 and row['status'] == 'completed'
        assert row['counts'] == dict(native_calls=255, native_output_checks=255, compilation_attempts=0)
        assert row['engine_closed'] is True
        assert row['source_commit'] == 'f5fa13377a3d967ba3d4022fb73ae59b1c87441c' and row['source_dirty'] == ''
        assert row['assignment']['host_index'] == row['assignment']['torch_ordinal'] == 0
        assert len(row['tokens']) == 4 and all(len(lane) == 8 for lane in row['tokens'])
        print(json.dumps(dict(event='fresh_reference_valid', role=role, utc=time.time())), flush=True)
    record.update(status='completed', finished_utc=time.time())
    save()
except BaseException:
    record.update(status='failed', traceback=traceback.format_exc(), finished_utc=time.time())
    save()
    raise
finally:
    owner = run.stat()
    for directory, dirs, files in os.walk(run, followlinks=False):
        for name in dirs + files:
            os.chown(Path(directory) / name, owner.st_uid, owner.st_gid, follow_symlinks=False)
        os.chown(directory, owner.st_uid, owner.st_gid, follow_symlinks=False)
