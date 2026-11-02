"""Task-local transport; all remote execution uses hub.orca.run_remote."""
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone
import argparse

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('atom_reuse_transport',
    'C:/Users/USER/orca/scratch/ccgrid-npu-20261030/e6_controller.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.LOCAL = ROOT


def rpc(environment, code, stem):
    compile(code, stem, 'exec')
    return transport.rpc(environment, code, stem+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', choices=['dell_0','rebellion'])
    parser.add_argument('script', type=Path)
    args = parser.parse_args()
    print(json.dumps(rpc(args.environment,args.script.read_text(encoding='utf-8'),args.script.stem),indent=2))
