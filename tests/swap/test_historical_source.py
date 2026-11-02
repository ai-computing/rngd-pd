"""Historical source identity must survive current runtime evolution."""
import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def sources():
    path = Path(__file__).parents[2] / "experiments/historical_source.py"
    spec = importlib.util.spec_from_file_location("historical_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_uses_exact_historical_git_blob_not_new_checkout(sources, tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args]).strip()
    git("init", "-q")
    path = tmp_path / "runtime.py"
    path.write_bytes(b"old runtime\n")
    git("-c", "core.autocrlf=false", "add", "runtime.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "old")
    pin = git("rev-parse", "HEAD").decode()
    path.write_bytes(b"new runtime\n")
    assert sources.source_bytes(tmp_path, pin, "runtime.py") == b"old runtime\n"
    (tmp_path / "new.py").write_bytes(b"not in historical source\n")
    with pytest.raises(ValueError, match="missing"):
        sources.source_bytes(tmp_path, pin, "new.py")


def test_archive_fallback_returns_checkout_for_caller_hash_validation(sources, tmp_path):
    (tmp_path / "runtime.py").write_bytes(b"historical CRLF\r\n")
    assert sources.source_bytes(tmp_path, "a"*40, "runtime.py") == b"historical CRLF\r\n"


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "C:/drive.py", "a\\b.py"])
def test_rejects_unsafe_source_names(sources, tmp_path, name):
    with pytest.raises(ValueError, match="path"):
        sources.source_bytes(tmp_path, "a"*40, name)
