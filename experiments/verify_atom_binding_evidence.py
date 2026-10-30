"""Verify exported ATOM B1 evidence and derive timings; standard library only."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_file(root, entry):
    path = PurePosixPath(entry["path"])
    require(not path.is_absolute() and ".." not in path.parts and ":" not in str(path), "unsafe evidence path")
    raw = root.joinpath(*path.parts).read_bytes()
    require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], "hash mismatch: " + str(path))
    return raw


def verify_export(root, archive_name, trial=None):
    manifest = load(root / "export-manifest.json")
    raw = (root / archive_name).read_bytes()
    require(len(raw) == manifest["archive"]["bytes"] and digest(raw) == manifest["archive"]["sha256"], "export archive mismatch")
    packed = json.loads(gzip.decompress(raw))
    if trial is not None:
        require(packed["trial"] == trial and len(packed["files"]) == manifest["archive"]["files"], "export trial/count mismatch")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    require(len(entries) == len(manifest["files"]) == len(packed["files"]), "duplicate/missing export entries")
    require(set(entries) == {entry["path"] for entry in packed["files"]}, "archive inventory mismatch")
    for entry in packed["files"]:
        data = base64.b64decode(entry["base64"], validate=True)
        require(len(data) == entry["bytes"] and digest(data) == entry["sha256"], "packed entry digest mismatch")
        require(data == verify_file(root, entries[entry["path"]]), "raw export differs from archive")


def stats(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return dict(n=len(values), median=statistics.median(values), q1=q1, q3=q3, iqr=q3 - q1)


def verify(root):
    package = load(root / "package-manifest.json")
    entries = package["files"]
    declared = {entry["path"] for entry in entries}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "package-manifest.json"}
    require(len(declared) == len(entries) and declared == actual, "package inventory mismatch")
    for entry in entries:
        verify_file(root, entry)
    freeze = load(root / "provenance/controller-freeze-atom-source-01.stdout")
    require(freeze["head"] == package["native_runtime_commit"] == "fbfdb8bb9aa86750f144fa4ead276fe17f6db537", "source freeze receipt differs")
    attempts = []
    for number in range(1, 5):
        name = "atom-b1-native-%02d" % number
        run = root / "runs" / name
        verify_export(run, "small-evidence.json.gz", name)
        result = load(run / "result.json")
        cleanup = load(run / "cleanup.json")
        terminal = load(run / "terminal-api.json")
        require(cleanup["job_absent"] and cleanup["pod_absent"]
                and cleanup["pod_uid"] == result["pod_uid"]
                and cleanup["job_uid"] == terminal["job"]["uid"], "cleanup UID/absence evidence differs")
        require(result["status"] == ("completed" if number == 4 else "failed"), "failed attempt history changed")
        if number < 4:
            require("Failed to retrieve information" in result["error"], "failed device selection evidence missing")
        attempts.append(result)
    run = root / "runs/atom-b1-native-04"
    for entry in load(run / "log-export-manifest.json")["files"]:
        verify_file(run, entry)
    require((run / "kit.stderr").read_bytes() == b"", "kit stderr was not empty")
    result = attempts[-1]
    kit = load(run / "kit-report.json")
    require(result["kit_returncode"] == 0 and kit["ok"] and kit["level"] == "B1"
            and not kit["failures"] and len(kit["cases"]) == 2, "native B1 kit did not pass two cases")
    assignment = result["assignment"]
    require(assignment["allocated_device_files"] == assignment["actual_device_files"] == ["/dev/rbln0"]
            and assignment["physical_device_id"] == assignment["sdk_device_id"] == 0
            and not assignment["broad_visibility_allowed"] and not assignment["broader_devices_exposed"], "successful assignment differs")
    job = load(run / "job.json")
    container = next(c for c in job["spec"]["template"]["spec"]["containers"] if c["name"] == "run")
    require(container["resources"]["limits"]["rebellions.ai/ATOM"] == "1"
            and container["resources"]["requests"]["rebellions.ai/ATOM"] == "1"
            and container["securityContext"]["privileged"] is False, "successful Job allocation/security differs")
    cleanup = load(run / "cleanup.json")
    require(cleanup["terminal_phase"] == "Succeeded" and all(c["state"]["terminated"]["exitCode"] == 0
            for c in cleanup["containers"]), "successful Job did not terminate cleanly")
    require(result["vendor_internal_apis_used"] == [] and result["experimental_constant_attachment"] is False
            and result["compile_attempt_count"] == 0 and not result["compile_attempts"]
            and "rebel.compile_from_torch" in result["guarded_compile_apis"], "public factory/compile guard failed")
    fresh, loads, repeats = result["fresh_reference"], result["loads"], result["repeats"]
    require(fresh["closed_before_binding_load"] and len(loads) == 7 and len(repeats) == 5, "fresh/load/repeat count differs")
    require([row["phase"] for row in loads] == ["kit"] * 2 + ["repeat"] * 5, "kit and repeat phases differ")
    require([dict(row, repeat=i) for i, row in enumerate(loads[2:])] == repeats, "repeat rows differ from load observations")
    graph = result["graph"]
    require(graph["sha256"] == graph["final_sha256"] and result["compiled_source_reused"], "compiled source changed")
    require(result["prompts"] == [[1, 42, 256, 1024, 7, 999, 2048, 13]] and result["n_new"] == 8, "generation input differs")
    for row in [fresh] + loads:
        require(row["run_attempts"] == row["run_completed"] == len(row["run_seconds"]) == 15
                and all(t >= 0 for t in row["run_seconds"])
                and row["native_runtime_type"] == "rebel.sync_runtime.SyncRuntime"
                and row["compiled_type"] == "rebel.compiled_model.RBLNCompiledModel"
                and row["compiled_device_allocation"] == 15210643456, "actual native runtime accounting differs")
        require(row["tokens"] == fresh["tokens"] and len(row["tokens"][0]) == 8, "fresh native token parity failed")
        if row is not fresh:
            require(row["tokens_equal_fresh"] and row["evidence"]["runtime_api"] == "public-AtomSession"
                    and row["evidence"]["sha256"] == graph["sha256"]
                    and row["resident"]["state"] == "ready" and row["resident"]["manifest"] == row["manifest"], "native load admission/parity failed")
    calls = sum(row["run_completed"] for row in [fresh] + loads)
    require(calls == result["native_runtime_calls_completed"] == 120, "native call total differs")
    timing = {field: stats([row[field] for row in repeats]) for field in result["repeat_summary"]}
    require(timing == result["repeat_summary"], "raw repeated timing summary differs from recomputation")
    timing.update({field: stats([row["session_timings"][field] for row in repeats]) for field in ("parse_s", "create_runtime_s")})
    timing["load_plus_post_load_ttft_s"] = stats([row["load_including_hash_s"] + row["ttft_s"] for row in repeats])
    preparation = root / "preparation"
    verify_export(preparation, "small-atom-preparation.json.gz")
    compiled = load(preparation / "evidence/atom-public-compile.json")
    require(compiled["status"] == "completed" and compiled["runtime_creations"] == 0
            and compiled["graph_sha256"] == graph["sha256"] and compiled["graph_bytes"] == graph["bytes"]
            and compiled["source_commit"] == package["graph_producer_commit"], "off-card graph identity differs")
    sources = dict(compiled["source_files"])
    for path, expected in result["source_hashes"].items():
        if "/src/" in path:
            sources["src/" + path.split("/src/", 1)[1]] = expected
    repository = Path(__file__).resolve().parents[1]
    for path, expected in sources.items():
        require(digest((repository / path).read_bytes().replace(b"\r\n", b"\n")) == expected,
                "current canonical LF source differs from native: " + path)
        require(package["source_identity_comparison"][path]["native_sha256"] == expected
                and package["source_identity_comparison"][path]["lf_equivalent"], "source comparison receipt differs")
    require(compiled["source_files"]["src/rngd_pd/swap/atom.py"] == sources["src/rngd_pd/swap/atom.py"], "graph producer/runtime source changed")
    for source, target in (("atom-b1-native-04.py", "run_atom_b1.py"), ("compile_atom_public.py", "compile_atom_public.py"),
                           ("prepare_atom_artifacts.py", "prepare_atom_artifacts.py")):
        require((preparation / "scripts" / source).read_bytes() == (root / "harness" / target).read_bytes(), "measured harness copy differs")
    def native_hash(suffix):
        matches = [value for path, value in result["source_hashes"].items() if path.endswith(suffix)]
        require(len(matches) == 1, "missing native input/source hash: " + suffix)
        return matches[0]
    require(digest((root / "harness/run_atom_b1.py").read_bytes()) == native_hash("/scripts/atom-b1-native-04.py"), "successful native harness digest differs")
    require(digest((root / "harness/conformance.py").read_bytes()) == native_hash("/conformance/binding/conformance.py"), "neutral kit source digest differs")
    provider_path = preparation / "inputs/atom-fixtures.json"
    require(digest(provider_path.read_bytes()) == native_hash("/inputs/atom-fixtures.json") == kit["fixtures"]["sha256"], "fixture provider digest differs")
    prep = load(preparation / "evidence/atom-artifact-preparation.json")
    failed_prep = load(preparation / "evidence/atom-artifact-preparation-attempt1.json")
    require(prep["status"] == "completed" and failed_prep["status"] == "failed"
            and prep["provider_sha256"] == digest(provider_path.read_bytes()), "failed/resumed preparation not retained")
    require(len(failed_prep["commands"]) == 2 and all(command["returncode"] == 0 for command in failed_prep["commands"]), "authentic artifact pushes failed")
    manifests = {command["stdout"]["ref"]: command["stdout"]["manifest"] for command in failed_prep["commands"]}
    plans = {}
    for command in prep["commands"]:
        path = preparation / "inputs" / PurePosixPath(command["output"]).name
        require(command["returncode"] == 0 and command["argv"][1] == "diff"
                and digest(path.read_bytes()) == command["sha256"], "authentic CLI diff output digest differs")
        plans[path.name] = load(path)
    require(set(plans) == {"atom-base.switchplan.json", "atom-target.switchplan.json"}, "incomplete authentic plans")
    for case, measured in zip(load(provider_path)["cases"], kit["cases"]):
        plan = plans[PurePosixPath(case["switchplan"]).name]
        require(case["name"] == measured["name"] and case["base_dir"] == measured["base_dir"]
                and case["target_dir"] == measured["target_dir"] and case["expect_mode"] == "reload", "measured provider case differs")
        require(plan["base_manifest"] == manifests["base"]
                and plan["manifest"] == manifests["base" if case["name"] == "native-identical" else "target"], "authentic push/diff manifest differs")
        observed = [step["stdin"] for step in measured["results"] if (step.get("stdin") or {}).get("schema") == "onggi.switchplan.v1"]
        require(observed[:2] == [plan, plan], "pure plan kit inputs differ from authentic CLI plan")
        updates = [step["stdout"] for step in measured["results"] if (step.get("stdout") or {}).get("schema") == "onggi.updateplan.v1"]
        require(len(updates) == 3 and updates[0] == updates[1]
                and all(u["mode"] == "reload" and u["reason"] and u["units"] == [] for u in updates), "B1 pure reload/refusal plan differs")
        for role, inventory in (("base", measured["base_files"]),
                                ("base" if case["name"] == "native-identical" else "target", measured["target_files"])):
            require(set(inventory) == {"session.rbln", "session.rbln.json", "atom-binding.json", "profile.json"}, "artifact inventory incomplete")
            require(inventory["session.rbln"] == dict(size=graph["bytes"], sha256=graph["sha256"]), "artifact graph SHA differs")
            for filename in ("session.rbln.json", "atom-binding.json", "profile.json"):
                raw = (preparation / "atom-artifacts" / role / filename).read_bytes()
                require(inventory[filename] == dict(size=len(raw), sha256=digest(raw)), "artifact metadata SHA differs")
    for role in ("base", "target"):
        require(digest((preparation / "atom-artifacts" / role / "session.rbln.json").read_bytes()) == compiled["sidecar_sha256"], "publisher graph sidecar differs")
    host = result["host_checkpoint"]["files"]
    require(result["host_checkpoint"]["path"] == compiled["checkpoint"], "host checkpoint path differs from graph producer")
    require(len(host) == 6 and {entry["path"] for entry in host} == {"config.json", "model.safetensors.index.json"}
            | {"model-%05d-of-00004.safetensors" % n for n in range(1, 5)}
            and all(entry["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) for entry in host), "host checkpoint hash inventory incomplete")
    summary = dict(schema="atom-binding-evidence-summary.v1", verified=True, conformance_attempts=4, passing_attempts=1,
        native_runtime_commit=package["native_runtime_commit"], graph_producer_commit=package["graph_producer_commit"],
        neutral_kit_commit=package["neutral_kit_commit"], native_cases=2, repeated_reload_n=5,
        native_runtime_calls=120, compile_attempts=0, graph_sha256=graph["sha256"],
        graph_bytes=graph["bytes"], timing_seconds=timing, quartile_method="statistics.quantiles inclusive",
        timing_scope="Serial repeated full loads including source hashing; post-load prompt TTFT measured separately; no cold-cache or trigger-to-first-token claim",
        host_checkpoint_files=host, host_checkpoint_payloads_packaged=False,
        input_verification="Artifact graph/metadata inventories match kit+off-card producer; packaged metadata/provider/CLI plans byte-verified. Host checkpoint payload hashes are recorded by native worker; payloads absent so not independently rehashed here.",
        successful_assignment=assignment, nonzero_physical_mapping_native_verified=False,
        failed_attempts_preserved=3, public_factory_only=True, conformance_level="B1")
    saved = root / "derived-summary.json"
    if saved.exists():
        require(load(saved) == summary, "published derived summary differs")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data/atom-binding-20261030")
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
