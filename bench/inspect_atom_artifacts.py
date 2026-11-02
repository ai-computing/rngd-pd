"""Collect public compiled-model metadata; does not create a native runtime."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(output, result):
    temporary = output / '.result.tmp'
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / 'result.json')
    finally:
        temporary.unlink(missing_ok=True)


def error(phase, exception):
    return dict(phase=phase, type=type(exception).__name__, message=str(exception))


def validate_manifest(raw):
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get('schema') != 'rngd-pd.atom-inspect-inputs.v1':
        raise ValueError('unsupported input manifest schema')
    rows = document.get('files')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 18:
        raise ValueError('manifest requires1–18 compiled files')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('file entry must be an object')
        name = row.get('path')
        if not isinstance(name, str) or not name or '\\' in name or ':' in name:
            raise ValueError('invalid input path')
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name or relative.suffix != '.rbln':
            raise ValueError('path must be a canonical relative .rbln path')
        if name in seen:
            raise ValueError('duplicate input path')
        seen.add(name)
        if type(row.get('bytes')) is not int or row['bytes'] <= 0:
            raise ValueError('input size must be a positive integer')
        if not isinstance(row.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('input SHA must be64 lowercase hex digits')
    return rows


def fingerprint(path):
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('input must be a regular file')
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def check_metadata(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('metadata dictionary keys must be strings')
            check_metadata(item)
    elif isinstance(value, list):
        for item in value:
            check_metadata(item)
    elif value is None or type(value) in (str, int, bool):
        pass
    elif type(value) is float and math.isfinite(value):
        pass
    else:
        raise TypeError('metadata contains a non-JSON value: ' + type(value).__name__)


def inspect_file(root, expected, inspect, row, checkpoint):
    row.update(path=expected['path'], expected_bytes=expected['bytes'], expected_sha256=expected['sha256'],
               status='running', phase='input_hash', started_utc=now(), native_api_called=False)
    checkpoint()
    phase = 'input_hash'
    started = time.perf_counter()
    try:
        path = root / expected['path']
        cursor = root
        for component in PurePosixPath(expected['path']).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                raise ValueError('symlink input is forbidden')
        before = fingerprint(path)
        row['input_stat_before'] = before
        if before[2] != expected['bytes']:
            raise ValueError('input size differs from frozen manifest')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        row['observed_sha256'] = digest.hexdigest()
        if fingerprint(path) != before:
            raise ValueError('input changed during hashing')
        if row['observed_sha256'] != expected['sha256']:
            raise ValueError('input SHA differs from frozen manifest')
        row['input_hash_s'] = time.perf_counter() - started
        # A pre-call checkpoint cannot prove whether an abruptly terminated
        # process actually entered the SDK, so its call outcome stays unknown.
        row.update(phase='inspect_pending', native_api_called=None)
        phase = 'output_snapshot'
        checkpoint()
        phase = 'inspect'
        row['native_api_called'] = True
        inspect_started = time.perf_counter()
        try:
            metadata = inspect(str(path))
        finally:
            row['inspect_s'] = time.perf_counter() - inspect_started
        phase = 'metadata'
        if not isinstance(metadata, dict):
            raise TypeError('inspect must return a metadata dictionary')
        check_metadata(metadata)
        if len(json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 8 * 1024 * 1024:
            raise ValueError('metadata exceeds8MiB')
        row['metadata'] = metadata
        phase = 'input_stability'
        row['input_stat_after'] = fingerprint(path)
        if row['input_stat_after'] != before:
            raise ValueError('input changed during inspection')
        row['status'] = 'completed'
    except Exception as exception:
        row['status'] = 'failed'
        row['error'] = error(phase, exception)
        if phase == 'input_hash':
            row['input_hash_s'] = time.perf_counter() - started
    row['phase'] = 'completed' if row['status'] == 'completed' else phase
    row['finished_utc'] = now()
    return row


def run_probe(manifest_path, input_root, output):
    root = Path(input_root).resolve(strict=True)
    destination = Path(output).resolve()
    if not root.is_dir() or destination == root or root in destination.parents:
        raise ValueError('output must be outside the input tree')
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(schema='rngd-pd.atom-inspect-result.v1', status='running', started_utc=now(),
                  command=sys.argv, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_root=str(root), manifest_path=str(Path(manifest_path).resolve()),
                  python=platform.python_version(), platform=platform.platform(),
                  runtime_api_called=False, coverage='unknown', files=[],
                  interpretation='Declared metadata only; not exhaustive coverage, live allocation, residency or serving correctness')
    save(destination, result)
    phase = 'manifest'
    try:
        with Path(manifest_path).open('rb') as stream:
            raw = stream.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024:
            raise ValueError('manifest exceeds256KiB')
        result['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
        (destination / 'inputs.json').write_bytes(raw)
        rows = validate_manifest(raw)
        result['expected_files'] = len(rows)
        phase = 'sdk'
        result['package_versions'] = {}
        for name in ('rebel-compiler', 'rebel'):
            try:
                result['package_versions'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result['package_versions'][name] = None
        started = time.perf_counter()
        try:
            rebel = importlib.import_module('rebel')
            inspect = rebel.RBLNCompiledModel.inspect
            if not callable(inspect):
                raise TypeError('public inspect API is not callable')
        finally:
            result['sdk_import_s'] = time.perf_counter() - started
        phase = 'output_snapshot'
        save(destination, result)
        for expected in rows:
            row = {}
            result['files'].append(row)
            inspect_file(root, expected, inspect, row, lambda: save(destination, result))
            save(destination, result)
        result['status'] = 'completed' if all(row['status'] == 'completed' for row in result['files']) else 'failed'
    except Exception as exception:
        result['status'] = 'failed'
        result['error'] = error(phase, exception)
    result['finished_utc'] = now()
    save(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.manifest, args.input_root, args.output)
    print(json.dumps(dict(status=result['status'], files=len(result['files']), coverage='unknown')))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
