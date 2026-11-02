"""Fresh processes exercise production KV with vLLM and rm-core unavailable."""
import os
from pathlib import Path
import subprocess
import sys


BLOCK_OPTIONAL = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('vllm', 'rm_core'):
            raise ModuleNotFoundError('blocked optional dependency: ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
'''


def run_child(code, root):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'),
               PD_SHM_ROOT=str(root), AIRM_KV_TRANSPORT='shm')
    return subprocess.run([sys.executable, '-c', BLOCK_OPTIONAL + code], env=env,
                          capture_output=True, text=True, timeout=20)


def test_shm_producer_exits_before_standalone_cluster_consumer(tmp_path):
    producer = run_child('''
import numpy as np
from rngd_pd.connector.worker import WorkerKVConnector, WorkerRole
from rngd_pd.connector import transport
transport.WIRE = 'bf16'
conn = WorkerKVConnector(WorkerRole.WORKER, req_id='replay',
                         transport=transport.select_transport(), n_layers=2)
for i in range(2):
    k = np.arange(1, 9, dtype=np.float32).reshape(1, 2, 4) + i * 16
    v = -np.arange(1, 9, dtype=np.float32).reshape(1, 4, 2) - i * 16
    conn.save_kv_layer('layer.' + str(i), (k, v))
conn.wait_for_save()
conn.t.close()
assert not any(n == 'vllm' or n.startswith('vllm.') for n in sys.modules)
''', tmp_path)
    assert producer.returncode == 0, producer.stderr
    assert len(list(tmp_path.iterdir())) == 4
    consumer = run_child('''
import numpy as np
from rngd_pd.serving import cluster
worker = cluster.DecodeWorkerBase.__new__(cluster.DecodeWorkerBase)
worker.nl, worker.NKV, worker.HD, worker.S = 2, 1, 2, 4
k, v, elapsed = worker._load_kv_now('replay')
assert k.tolist() == [[[[1.,2.,3.,4.],[5.,6.,7.,8.]]], [[[17.,18.,19.,20.],[21.,22.,23.,24.]]]]
assert v.tolist() == [[[[-1.,-2.],[-3.,-4.],[-5.,-6.],[-7.,-8.]]], [[[-17.,-18.],[-19.,-20.],[-21.,-22.],[-23.,-24.]]]]
assert elapsed >= 0
assert not any(n == 'vllm' or n.startswith('vllm.') for n in sys.modules)
''', tmp_path)
    assert consumer.returncode == 0, consumer.stderr
    assert list(tmp_path.iterdir()) == []


def test_vllm_plugin_without_vllm_fails_with_actionable_error(tmp_path):
    result = run_child('''
try:
    import rngd_pd.connector.rngd_kv
except ImportError as error:
    assert 'vLLM plugin requires vLLM' in str(error), str(error)
else:
    raise AssertionError('plugin unexpectedly loaded without vLLM')
''', tmp_path)
    assert result.returncode == 0, result.stderr
