"""Read measured source revisions without treating newer code as native evidence."""
from pathlib import PurePosixPath
import re
import subprocess


def source_bytes(repository, commit, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name
            or not relative.parts or re.fullmatch(r"[0-9a-f]{40}", commit) is None):
        raise ValueError("invalid historical source path or commit")
    try:
        exists = subprocess.run(["git", "-C", str(repository), "cat-file", "-e", commit+"^{commit}"],
                                capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        exists = False
    if not exists:
        # Archives/shallow checkouts still require exact measured hashes in callers.
        return (repository / name).read_bytes()
    result = subprocess.run(["git", "-C", str(repository), "show", commit+":"+name],
                            capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError("historical source missing from pinned commit: " + name)
    return result.stdout
