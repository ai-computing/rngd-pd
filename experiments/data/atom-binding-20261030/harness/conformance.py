#!/usr/bin/env python3
# Copyright 2026 The onggi authors. SPDX-License-Identifier: Apache-2.0
"""Conformance tests for an onggi device binding (ADR-0010). Python 3.6+, standard library only.

    python3 conformance.py [--level B1|B2] [--fixtures FILE] [--report FILE] [-- BINDING COMMAND...]

Without a command it tests the reference fake binding. The binding command is run as
`COMMAND <operation> [args]` with JSON on stdin and stdout; see README.md.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_data  # noqa: E402

FAILS = []


def check(cond, what):
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        FAILS.append(what)
    return cond


def run(cmd, op, *args, stdin=None, expect_ok=True):
    p = subprocess.run(cmd + [op] + list(args), input=None if stdin is None else json.dumps(stdin).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if expect_ok and p.returncode != 0:
        raise RuntimeError("%s %s: exit %d: %s" % (op, " ".join(args), p.returncode, p.stderr.decode()[-500:]))
    try:
        return p.returncode, json.loads(p.stdout.decode()) if p.stdout.strip() else None
    except ValueError:
        raise RuntimeError("%s: stdout is not JSON: %r" % (op, p.stdout[:200]))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_size(data):
    """A file is either synthetic bytes or an external path; external bytes stay on disk."""
    return os.path.getsize(data) if isinstance(data, str) else len(data)


def interval_sha(data, offset, length):
    if not isinstance(data, str):
        return sha(data[offset:offset + length])
    digest = hashlib.sha256()
    with open(data, "rb") as f:
        f.seek(offset)
        while length:
            chunk = f.read(min(length, 1024 * 1024))
            if not chunk:
                raise ValueError("artifact shortened while hashing: %s" % data)
            digest.update(chunk)
            length -= len(chunk)
    return digest.hexdigest()


def intervals_equal(left, left_offset, right, right_offset, length):
    """Compare external file intervals without retaining model-sized buffers."""
    with open(left, "rb") as a, open(right, "rb") as b:
        a.seek(left_offset)
        b.seek(right_offset)
        while length:
            size = min(length, 1024 * 1024)
            achunk, bchunk = a.read(size), b.read(size)
            if len(achunk) != size or len(bchunk) != size:
                raise ValueError("artifact shortened while comparing intervals")
            if achunk != bchunk:
                return False
            length -= size
    return True


def artifacts_differ(sp, base, target):
    for f in sp["files"]:
        source = base.get(f["path"]) or base.get(f.get("base"))
        if source is None:
            if f["size"] > 0:
                return True
        elif (file_size(source) != f["size"]
              or not intervals_equal(target[f["path"]], 0, source, 0, f["size"])):
            return True
    return False


def artifact_files(root):
    if not os.path.isdir(root):
        raise ValueError("artifact directory does not exist: %s" % root)
    files = {}
    def walk_error(error):
        raise error
    for directory, dirs, names in os.walk(root, onerror=walk_error):
        for name in dirs + names:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError("artifact symlinks are not supported: %s" % name)
        for name in names:
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                raise ValueError("artifact entry is not a regular file: %s" % path)
            files[os.path.relpath(path, root).replace(os.sep, "/")] = path
    return files


def validate_switch(sp, base, target):
    """Check the provider's byte-space plan before trusting its coverage claims."""
    def require(condition, message):
        if not condition:
            raise ValueError("invalid switchplan: " + message)

    def integer(value, minimum=0):
        return type(value) is int and value >= minimum

    require(isinstance(sp, dict) and sp.get("schema") == "onggi.switchplan.v1", "schema")
    for field in ("base_manifest", "manifest"):
        require(isinstance(sp.get(field), str) and re.fullmatch(r"sha256:[0-9a-f]{64}", sp[field]), field)
    require(isinstance(sp.get("files"), list), "files must be a list")
    paths, totals = set(), {"size": 0, "same": 0, "moved": 0, "new": 0}
    for f in sp["files"]:
        require(isinstance(f, dict), "file must be an object")
        path = f.get("path")
        require(isinstance(path, str) and path in target and path not in paths, "unknown or duplicate target path")
        paths.add(path)
        for field in totals:
            require(integer(f.get(field)), "file " + field)
            totals[field] += f[field]
        require(f["size"] == file_size(target[path]), "target file size: " + path)
        if "base" in f:
            require(isinstance(f["base"], str) and f["base"] in base, "unknown paired base file")
        require(isinstance(f.get("ranges"), list), "ranges are required (use diff --ranges)")
        position, counts = 0, {"same": 0, "moved": 0, "new": 0}
        for r in f["ranges"]:
            require(isinstance(r, dict) and integer(r.get("offset")) and integer(r.get("length"), 1), "range interval")
            require(r["offset"] == position and position + r["length"] <= f["size"], "ranges must partition each file")
            kind = r.get("kind")
            require(kind in ("same", "elsewhere", "new"), "range kind")
            counts["moved" if kind == "elsewhere" else kind] += r["length"]
            position += r["length"]
            if kind == "same":
                require(f.get("base") in base and position <= file_size(base[f["base"]]), "same range exceeds base file")
                require(intervals_equal(target[path], r["offset"], base[f["base"]], r["offset"], r["length"]),
                        "same range bytes differ from paired base")
            if kind == "elsewhere":
                source = r.get("source_file")
                require(isinstance(source, str) and source in base and integer(r.get("source_offset")), "elsewhere source")
                require(r["source_offset"] + r["length"] <= file_size(base[source]), "elsewhere range exceeds base file")
                require(intervals_equal(target[path], r["offset"], base[source], r["source_offset"], r["length"]),
                        "elsewhere range bytes differ from source")
            else:
                require("source_file" not in r and "source_offset" not in r, "source fields require elsewhere")
        require(position == f["size"] and all(f[k] == counts[k] for k in counts), "range coverage or counts")
    require(paths == set(target), "plan must cover every target file")
    require(all(integer(sp.get(k)) and sp[k] == totals[k] for k in totals), "total sizes or counts")


def load_fixtures(path, level):
    directory = os.path.dirname(os.path.abspath(path))
    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("fixture paths must be nonempty strings")
        return os.path.abspath(os.path.join(directory, value))

    with open(path, encoding="utf-8") as f:
        provider = json.load(f)
    if not (isinstance(provider, dict) and provider.get("schema") == "onggi.binding-fixtures.v1"
            and isinstance(provider.get("cases"), list) and provider["cases"]):
        raise ValueError("fixtures require schema onggi.binding-fixtures.v1 and nonempty cases")
    cases, names, changed_units = [], set(), False
    for case in provider["cases"]:
        if not isinstance(case, dict):
            raise ValueError("fixture case must be an object")
        name, mode = case.get("name"), case.get("expect_mode")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("fixture names must be nonempty and unique")
        names.add(name)
        if mode not in ("units", "reload") or (level == "B1" and mode != "reload"):
            raise ValueError("fixture expect_mode must be units or reload (B1 requires reload)")
        bdir, tdir = resolve(case.get("base_dir")), resolve(case.get("target_dir"))
        base, target = artifact_files(bdir), artifact_files(tdir)
        sp = case.get("switchplan")
        if isinstance(sp, str):
            with open(resolve(sp), encoding="utf-8") as f:
                sp = json.load(f)
        validate_switch(sp, base, target)
        changed_units = changed_units or (mode == "units" and sp["moved"] + sp["new"] > 0
                                          and artifacts_differ(sp, base, target))
        cases.append((name, bdir, tdir, base, target, sp, mode))
    if level == "B2" and not changed_units:
        raise ValueError("B2 requires at least one changed switch with expect_mode units")
    return cases


def covers(update, switch):
    """Every byte that is not `same` in the switch plan lies in a unit of the same file."""
    units = {}
    for u in update["units"]:
        units.setdefault(u["file"], []).append((u["offset"], u["offset"] + u["length"]))
    for f in switch["files"]:
        for r in f["ranges"]:
            if r["kind"] == "same":
                continue
            pos, end = r["offset"], r["offset"] + r["length"]
            for a, b in sorted(units.get(f["path"], [])):
                if a <= pos < b:
                    pos = b
            if pos < end:
                return False
    return True


def verify_units(ver, files, what, update, op):
    """Verify every planned/restored interval; bytes_written is a nonnegative device counter."""
    mode = update["mode"]
    manifest = update["manifest" if op == "apply" else "base_manifest"]
    if not (isinstance(ver, dict) and ver.get("schema") == "onggi.verification.v1"
            and ver.get("ok") is True and ver.get("op") == op
            and mode in ("units", "reload") and ver.get("mode") == mode
            and isinstance(manifest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", manifest)
            and ver.get("manifest") == manifest and isinstance(ver.get("units"), list)):
        return check(False, what)

    def interval(unit):
        if not isinstance(unit, dict):
            return None
        path, off, length = unit.get("file"), unit.get("offset"), unit.get("length")
        if (not isinstance(path, str) or path not in files
                or type(off) is not int or type(length) is not int
                or off < 0 or length < 0 or off + length > file_size(files[path])
                or (length == 0 and not (mode == "reload" and off == 0 and file_size(files[path]) == 0))):
            return None
        return path, off, length

    # Reload rollback restores the base files, whose paths and sizes may differ from the target.
    required = update["units"] if mode == "units" else [
        {"file": path, "offset": 0, "length": file_size(data)} for path, data in files.items()]
    expected = [interval(u) for u in required]
    actual = [interval(u) for u in ver["units"]]
    ok = (None not in expected and None not in actual
          and len(set(expected)) == len(expected) and len(set(actual)) == len(actual)
          and set(actual) == set(expected))
    if ok:
        for unit, (path, off, length) in zip(ver["units"], actual):
            ok = ok and unit.get("sha256") == interval_sha(files[path], off, length)
        if "bytes_written" in ver:
            ok = (ok and type(ver["bytes_written"]) is int
                  and ver["bytes_written"] >= 0)
    return check(ok, what)


def synthetic_cases(work, level):
    for name in golden_data.CASES:
        base, target = golden_data.case(name)
        bdir, tdir = os.path.join(work, name, "base"), os.path.join(work, name, "target")
        golden_data.write(bdir, base)
        golden_data.write(tdir, target)
        with open(os.path.join(HERE, "golden", name + ".switchplan.json")) as f:
            sp = json.load(f)
        mode = "reload" if level == "B1" else "units" if name in ("identical", "inplace") else None
        yield name, bdir, tdir, base, target, sp, mode


def file_hashes(files):
    return {path: {"size": file_size(data), "sha256": interval_sha(data, 0, file_size(data))}
            for path, data in sorted(files.items())}


def run_cases(level, cmd, cases, report):
    for name, bdir, tdir, base, target, sp, mode in cases:
        start = len(FAILS)
        case_report = {"name": name, "base_dir": bdir, "target_dir": tdir, "expect_mode": mode,
                       "results": []}
        if report is not None:
            report["cases"].append(case_report)
            case_report.update(base_files=file_hashes(base), target_files=file_hashes(target),
                               switchplan_sha256=sha(json.dumps(sp, sort_keys=True, separators=(",", ":")).encode("utf-8")))

        def execute(op, *args, **kwargs):
            result = {"command": cmd + [op] + list(args), "stdin": kwargs.get("stdin")}
            if report is not None:
                case_report["results"].append(result)
            try:
                rc, value = run(cmd, op, *args, **kwargs)
                result.update(returncode=rc, stdout=value)
                return rc, value
            except Exception as error:
                result["error"] = str(error)
                raise

        print("== %s (%s)" % (name, level))
        try:
            _, prof = execute("describe", tdir)
            check(prof.get("schema") == "onggi.profile.v1", "describe returns onggi.profile.v1")

            execute("conformance-load", bdir, sp["base_manifest"])
            _, res = execute("resident")
            check(res.get("schema") == "onggi.resident.v1" and res.get("manifest") == sp["base_manifest"],
                  "resident names the loaded base")

            _, up1 = execute("plan", stdin=sp)
            _, up2 = execute("plan", stdin=sp)
            _, res2 = execute("resident")
            check(up1 == up2 and res2 == res, "plan is pure: same answer twice, resident state unchanged")
            check(up1.get("schema") == "onggi.updateplan.v1" and up1.get("mode") in ("units", "reload"),
                  "plan returns onggi.updateplan.v1 with mode units or reload")
            check(up1["base_manifest"] == sp["base_manifest"] and up1["manifest"] == sp["manifest"],
                  "update plan names the switch plan's base and target")
            if mode is not None:
                check(up1["mode"] == mode, "%s: expected mode %s" % (level, mode))
            if up1["mode"] == "units":
                check(covers(up1, sp), "units cover every byte that is not at the same offset")

            wrong_manifest = "sha256:" + ("1" if sp["base_manifest"] == "sha256:" + "0" * 64 else "0") * 64
            wrong = dict(sp, base_manifest=wrong_manifest)
            rc, upw = execute("plan", stdin=wrong, expect_ok=False)
            check(rc != 0 or (upw or {}).get("mode") == "reload", "plan against a base that is not resident refuses (reload)")

            if level == "B1":
                continue
            _, ver = execute("apply", "--target", tdir, stdin=up1)
            verify_units(ver, target, "apply verifies: every planned unit matches the target bytes", up1, "apply")
            _, res3 = execute("resident")
            check(res3.get("manifest") == sp["manifest"], "after apply the target is resident")

            _, rb = execute("rollback")
            verify_units(rb, base, "rollback restores: every planned unit matches the base bytes", up1, "rollback")
            _, res4 = execute("resident")
            check(res4.get("manifest") == sp["base_manifest"], "after rollback the base is resident")
            rc, _ = execute("rollback", expect_ok=False)
            check(rc != 0, "a second rollback has nothing to undo and fails")
        finally:
            case_report["failures"] = FAILS[start:]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("B1", "B2"), default="B2")
    parser.add_argument("--fixtures", help="external onggi.binding-fixtures.v1 provider JSON")
    parser.add_argument("--report", help="write a JSON run report to a new file")
    cmd = None
    if "--" in argv:
        cmd = argv[argv.index("--") + 1:]
        argv = argv[:argv.index("--")]
    options = parser.parse_args(argv)
    FAILS[:] = []
    report = {"schema": "onggi.binding-report.v1", "level": options.level,
              "argv": argv + (["--"] + cmd if cmd else []), "cases": []}
    report_file = None
    completed = False
    try:
        if options.report:
            report_file = open(options.report, "x", encoding="utf-8")
        if options.fixtures:
            if report_file is not None:
                report["fixtures"] = {"path": os.path.abspath(options.fixtures),
                                      "sha256": interval_sha(options.fixtures, 0, os.path.getsize(options.fixtures))}
            cases = load_fixtures(options.fixtures, options.level)
        work = tempfile.mkdtemp(prefix="onggi-binding-")
        if not options.fixtures:
            cases = synthetic_cases(work, options.level)
        if not cmd:
            os.environ["ONGGI_FAKE_DEVICE"] = os.path.join(work, "device")
            if options.level == "B1":
                os.environ["ONGGI_FAKE_LEVEL"] = "B1"
            cmd = [sys.executable, os.path.join(HERE, "fake_binding.py")]
        report["binding_command"] = cmd
        run_cases(options.level, cmd, cases, report if report_file is not None else None)
        completed = True
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
        report["error"] = str(error)
        check(False, str(error))
    finally:
        if report_file is not None:
            report.update(ok=completed and not FAILS, failures=list(FAILS))
            with report_file:
                json.dump(report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
    print("\n%d failed" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
