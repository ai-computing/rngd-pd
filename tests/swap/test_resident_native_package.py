import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('native_package', REPO / 'experiments/verify_resident_native_package.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)
DATA = REPO / 'experiments/data/resident-delivery-20261102/provenance'


def test_real_capacity_and_native_job_cleanup():
    result = package.verify(DATA, REPO)
    assert result['verified'] and result['native_jobs_uid_cleaned'] == 8
    assert result['capacity_native_calls'] == 9231


def test_modified_resource_receipt_is_rejected(tmp_path):
    shutil.copytree(DATA, tmp_path / 'provenance')
    path = tmp_path / 'provenance/controller/cdx-resident-native-capacity01-cleanup.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises((AssertionError, ValueError)):
        package.verify(tmp_path / 'provenance', REPO)
