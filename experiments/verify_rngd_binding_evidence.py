"""Check packaged evidence hashes and derive claims; imports no runtime/SDK."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_file(root, entry):
    relative = PurePosixPath(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe evidence path")
    path = root.joinpath(*relative.parts)
    data = path.read_bytes()
    require(len(data) == entry["bytes"] and digest(data) == entry["sha256"], "hash mismatch: " + str(path))
    return data


def verify(root):
    package = load(root / "package-manifest.json")
    for entry in package["files"]:
        verify_file(root, entry)
    for name in ("rngd-b2-native-01", "rngd-b2-native-02"):
        run = root / "runs" / name
        exported = load(run / "export-manifest.json")
        archive = (run / "small-evidence.json.gz").read_bytes()
        require(len(archive) == exported["archive"]["bytes"]
                and digest(archive) == exported["archive"]["sha256"], "archive mismatch: " + name)
        packed = json.loads(gzip.decompress(archive))
        require(packed["trial"] == name and len(packed["files"]) == exported["archive"]["files"],
                "archive identity/count mismatch")
        entries = {entry["path"]: entry for entry in exported["files"]}
        require(set(entries) == {entry["path"] for entry in packed["files"]}, "archive/export inventory mismatch")
        for entry in packed["files"]:
            raw = base64.b64decode(entry["base64"], validate=True)
            require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], "packed entry digest mismatch")
            require(raw == verify_file(run, entries[entry["path"]]), "archive differs from extracted raw file")
    probe_root = root / "runs/b2-readback-probe-01"
    for entry in load(probe_root / "readonly-export-manifest.json")["files"]:
        verify_file(probe_root, entry)
    probe = load(probe_root / "result.json")
    require(probe["status"] == "completed" and probe["actual_device_readback"]
            and probe["restored_exactly"] and probe["pointer_retained"]
            and probe["before_sha256"] == probe["restored_sha256"], "public readback probe did not pass")
    failed = load(root / "runs/rngd-b2-native-01/result.json")
    result = load(root / "runs/rngd-b2-native-02/result.json")
    failed_kit = load(root / "runs/rngd-b2-native-01/work/kit-report.json")
    kit = load(root / "runs/rngd-b2-native-02/work/kit-report.json")
    require(failed["status"] == "failed" and failed_kit["ok"] is False, "failed attempt not preserved")
    require(failed["source_files"] == result["source_files"], "runtime changed between failed/successful attempts")
    repository = Path(__file__).resolve().parents[1]
    for name, expected in result["source_files"].items():
        data = (repository / name).read_bytes().replace(b"\r\n", b"\n")
        require(digest(data) == expected, "current runtime differs from measured native source: " + name)
    rejected = next(case for case in failed_kit["cases"] if case["name"] == "native-sparse")
    unsupported = next(step["stdout"] for step in rejected["results"]
                       if (step.get("stdout") or {}).get("schema") == "onggi.updateplan.v1")
    require(unsupported["mode"] == "reload" and unsupported["reason"], "unsupported coarse fixture was not refused")
    require(result["status"] == "completed" and result["native_evidence"] and result["engine_closed"]
            and result["kit_ok"] and kit["ok"] and not kit["failures"] and len(kit["cases"]) == 3,
            "successful native kit result missing")
    counts = result["native_counts"]
    require(counts == dict(native_calls=2805, native_output_checks=2805, compilation_attempts=0)
            and result["native_outputs_verified"], "native execution/compile guard did not pass")
    generations = result["generations"]
    require(len(generations) == 11 and result["generation_gates"] and all(
        row["status"] == "completed" and row["tokens_equal_fresh_reference"]
        and row["generation_preserved_identities"]
        and row.get("serving_allocations_programs_retained", True) for row in generations), "generation/identity gates failed")
    require(all(row["native_counts"] == dict(native_calls=255, native_output_checks=255, compilation_attempts=0)
                for row in generations), "per-generation native counts differ")
    fault = result["silent_corruption_gate"]
    require(fault["status"] == "completed" and fault["actual_readback_rejected_corruption"]
            and fault["restored_native_base_tokens"] and fault["identity_preserved"], "native corruption/restore gate failed")
    for key in ("references_actual",):
        require(digest((root / "runs/rngd-b2-native-02/work/refsactual.json").read_bytes())
                == result["inputs"][key]["sha256"], "actual reference input digest mismatch")
    fresh = load(root / "runs/rngd-b2-native-02/work/interior-fresh.json")
    require(fresh["status"] == "completed" and fresh["engine_closed"]
            and fresh["source_commit"] == package["independent_reference_commit"]
            and fresh["counts"] == dict(native_calls=255, native_output_checks=255, compilation_attempts=0),
            "independent original-engine interior reference did not pass")
    require(digest((root / "harness/run_rngd_b2.py").read_bytes()) == result["harness_sha256"], "native harness bytes differ")
    preparation = root / "preparation"
    selected = load(preparation / "selection-manifest.json")
    exported = load(preparation / "export-manifest.json")
    require(digest((preparation / "export-manifest.json").read_bytes()) == selected["source_export_manifest_sha256"],
            "preparation export manifest identity differs")
    exported_entries = {entry["path"]: entry for entry in exported["files"]}
    for entry in selected["files"]:
        require(entry == exported_entries[entry["path"]], "selection differs from original export entry")
        verify_file(preparation, entry)
    for name, measured in (("run_rngd_b2.py", failed), ("rngd-b2-native-02.py", result)):
        require(digest((preparation / "scripts" / name).read_bytes()) == measured["harness_sha256"],
                "measured harness differs: " + name)
    require((preparation / "scripts/rngd-b2-native-02.py").read_bytes() == (root / "harness/run_rngd_b2.py").read_bytes(),
            "current and measured successful harness copies differ")
    prep = load(preparation / "evidence/rngd-interior-preparation.json")
    require(prep["status"] == "completed"
            and digest((preparation / "scripts/prepare_rngd_interior.py").read_bytes()) == prep["script_sha256"],
            "measured fixture preparation source differs")
    probe_source = digest((preparation / "scripts/probe_readback.py").read_bytes())
    for field in ("adapter_sha256", "harness_sha256"):
        if field in probe:
            require(probe[field] == probe_source, "probe harness source differs")
    for name, measured, report in (("rngd-fixtures.json", failed, failed_kit),
                                    ("rngd-interior-fixtures.json", result, kit)):
        provider_path = preparation / "inputs" / name
        require(digest(provider_path.read_bytes()) == measured["inputs"]["fixtures"]["sha256"],
                "native provider bytes differ: " + name)
        report_cases = {case["name"]: case for case in report["cases"]}
        for case in load(provider_path)["cases"]:
            if case["name"] not in report_cases:
                continue  # Failed attempt stopped before its final LM case.
            actual_case = report_cases[case["name"]]
            require(case["base_dir"] == actual_case["base_dir"] and case["target_dir"] == actual_case["target_dir"],
                    "provider and kit artifact paths differ")
            plan = load(preparation / "inputs" / PurePosixPath(case["switchplan"]).name)
            actual_switch = next(step["stdin"] for step in actual_case["results"]
                                 if (step.get("stdin") or {}).get("schema") == "onggi.switchplan.v1")
            require(plan == actual_switch, "authentic plan differs from measured kit input")
    require(digest((preparation / "inputs/references.json").read_bytes()) == result["inputs"]["references"]["sha256"],
            "original reference input bytes differ")
    plan_path = preparation / "inputs/interior.switchplan.json"
    provider_path = preparation / "inputs/rngd-interior-fixtures.json"
    require(digest(plan_path.read_bytes()) == prep["plan_sha256"]
            and digest(provider_path.read_bytes()) == prep["provider_sha256"], "preparation/provider/plan identity differs")
    commands = prep["commands"]
    require(len(commands) == 3 and all(command["returncode"] == 0 for command in commands), "actual onggi preparation failed")
    require(commands[-1]["argv"][1] == "diff" and commands[-1]["stdout"].encode() == plan_path.read_bytes(),
            "published plan differs from actual onggi stdout")
    plan = load(plan_path)
    require(json.loads(commands[0]["stdout"])["manifest"] == prep["base_manifest"] == plan["base_manifest"]
            and json.loads(commands[1]["stdout"])["manifest"] == prep["manifest"] == plan["manifest"],
            "authentic push/diff/preparation identities disagree")
    def inventory(entries):
        return {entry["name"]: {"size": entry["bytes"], "sha256": entry["sha256"]} for entry in entries}
    base_files, interior_files = inventory(prep["base_files"]), inventory(prep["interior_files"])
    require(len(base_files) == len(interior_files) == 6, "incomplete fixture inventory")
    for case in kit["cases"]:
        require(case["base_files"] == base_files, "kit base differs from complete preparation inventory")
        if case["name"] == "native-interior-projection":
            require(case["target_files"] == interior_files, "kit interior differs from complete preparation inventory")
    require(rejected["target_files"] == inventory(prep["payload_source_files"]), "original E7 payload-source inventory differs")
    units = {}
    for case in kit["cases"]:
        update = next(step["stdout"] for step in case["results"]
                      if (step.get("stdout") or {}).get("schema") == "onggi.updateplan.v1")
        units[case["name"]] = {"logical_bytes": update["bytes"], "units": update["units"]}
    require(units["native-identical"]["logical_bytes"] == 0, "identical transaction wrote units")
    require(units["native-interior-projection"]["logical_bytes"] == 1949696
            and units["native-interior-projection"]["units"][0]["unit"] == "model.layers.0.mlp.down_proj.weight@125:193",
            "projection unit summary differs")
    require(units["native-lm"]["logical_bytes"] == 262144
            and units["native-lm"]["units"][0]["unit"] == "lm_head.weight@12037:12069", "LM unit summary differs")
    diagnostics = result["final_resident_before_close"]["diagnostics"]
    require(diagnostics["readback_calls"] == 9 and diagnostics["readback_payload_bytes"] == 4561305600,
            "readback payload summary differs")
    return {"schema": "rngd-binding-evidence-summary.v1", "verified": True,
            "n": 1, "conformance_attempts": 2, "passing_attempts": 1,
            "latency": {"median": None, "IQR": None, "reason": "conformance only; no repeated latency measurement"},
            "native_cases": 3, "generation_gates": len(generations), "native_counts": counts,
            "independent_fresh_reference_native_calls": fresh["counts"]["native_calls"],
            "units": units, "readback_calls": diagnostics["readback_calls"],
            "readback_payload_bytes": diagnostics["readback_payload_bytes"],
            "byte_scope": "logical serving writes and whole-parent readback payload; not physical DMA",
            "failed_attempt_retained": True, "runtime_source_unchanged_between_attempts": True,
            "preparation_provider_plans_verified": True, "measured_harnesses_verified": True,
            "probe_result_records_harness_hash": any(key in probe for key in ("adapter_sha256", "harness_sha256"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data/binding-20261030")
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
