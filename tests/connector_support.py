"""Import the real connector with a minimal vLLM import boundary when absent.

This exercises our whole-request transfer contract, not vLLM's paged scheduler.
With vLLM installed, its actual base classes are used instead.
"""
from contextlib import contextmanager
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock


@contextmanager
def connector_module():
    name = "rngd_pd.connector.rngd_kv"
    boundary = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    replacements = {}
    if importlib.util.find_spec("vllm") is None:
        base = types.ModuleType(boundary)

        class Role(Enum):
            WORKER = "worker"
            SCHEDULER = "scheduler"

        class Base:
            def bind_connector_metadata(self, metadata):
                self._connector_metadata = metadata

        base.KVConnectorRole = Role
        base.KVConnectorBase_V1 = Base
        base.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
        replacements[boundary] = base
    path = Path(__file__).resolve().parents[1] / "src/rngd_pd/connector/rngd_kv.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    replacements[name] = module
    with mock.patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
        yield module
