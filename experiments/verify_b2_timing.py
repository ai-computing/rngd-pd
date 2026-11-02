#!/usr/bin/env python3
"""Offline, stdlib-only current-B2 timing verification and inclusive statistics.

python experiments/verify_b2_timing.py --root experiments/data/b2-timing-20261102
    [--out NEW_SUMMARY.json]

Requires exact-byte package/archive exports, inputs/, configs/, runs/ and
control/ Job sidecars. Source checks alone canonicalize CRLF to LF, comparing
the measured baseline/harness with pinned Git objects. Without those objects,
the checkout must still contain the exact historical sources. Current runtime
edits are not a new measurement of the historical baseline.
Bulk artifact hashes are the preserved off-card manifest, not newly rehashed
device evidence. Every failed/incomplete main run has n=0 and null statistics;
no successful subset is filtered into an aggregate. Smoke is always separate.
Exit 0: fully verified smoke+five-pair main; 2: missing/incomplete; 1: invalid.
--out is exclusive-create and should be outside the immutable evidence root.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys


BASELINE = "4482faef87dabef8f586c6e7dff382b889cca115"
HARNESS_COMMIT = "6349df7"
HARNESS_SHA = "faf52c5dfa3d6a4bae231d5de2654e6a00ca7e43de128504e7c5f7bc7500b44c"
FRESH_SOURCE = "f5fa13377a3d967ba3d4022fb73ae59b1c87441c"
METRICS = ("trigger_to_first_token_s", "plan_s", "apply_or_reload_s", "prompt_ttft_s",
           "generate_full_continuation_s", "logical_device_written_bytes", "readback_payload_bytes", "readback_s")
COUNTS = dict(compilation_attempts=0, native_calls=255, native_output_checks=255)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def near(a, b, message):
    require(finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), message)


def statistics_for(values):
    require(all(finite(value) and value >= 0 for value in values), "invalid metric observation")
    if not values:
        return dict(n=0, median=None, q1=None, q3=None, IQR=None, values=[])
    q1, _, q3 = statistics.quantiles(values, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return dict(n=len(values), median=statistics.median(values), q1=q1, q3=q3, IQR=q3-q1, values=values)


def verify_file(root, entry):
    relative = PurePosixPath(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in entry["path"],
            "unsafe evidence path")
    path = root.joinpath(*relative.parts)
    require(path.is_file() and not path.is_symlink(), "missing/nonregular evidence: " + entry["path"])
    raw = path.read_bytes()
    require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], "hash mismatch: " + str(path))
    return raw


def verify_export(run):
    exported = load(run / "export-manifest.json")
    archive = (run / "small-evidence.json.gz").read_bytes()
    require(len(archive) == exported["archive"]["bytes"]
            and digest(archive) == exported["archive"]["sha256"], "archive hash mismatch")
    packed = json.loads(gzip.decompress(archive))
    entries = {entry["path"]: entry for entry in exported["files"]}
    require(len(entries) == len(exported["files"]) == exported["archive"]["files"]
            == len(packed["files"]), "archive/export count mismatch")
    require(set(entries) == {entry["path"] for entry in packed["files"]}, "archive/export inventory mismatch")
    for entry in packed["files"]:
        raw = base64.b64decode(entry["base64"], validate=True)
        require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], "packed entry hash mismatch")
        require(raw == verify_file(run, entries[entry["path"]]), "archive/extracted bytes differ")
    return set(entries)


def require_coverage(document, units):
    by_file = {}
    for unit in units:
        require(type(unit["offset"]) is int and type(unit["length"]) is int
                and unit["offset"] >= 0 and unit["length"] > 0, "invalid admitted span")
        by_file.setdefault(unit["file"], []).append((unit["offset"], unit["offset"] + unit["length"]))
    for item in document["files"]:
        for span in item["ranges"]:
            if span["kind"] == "same":
                continue
            cursor, stop = span["offset"], span["offset"] + span["length"]
            for begin, end in sorted(by_file.get(item["path"], [])):
                if begin <= cursor:
                    cursor = max(cursor, end)
            require(cursor >= stop, "admitted unit coverage misses changed bytes in " + item["path"])


def verify_clock(row):
    timing = row["timings"]
    trigger, generate = timing["trigger_perf_ns"], timing["generate_call_perf_ns"]
    require(type(trigger) is int and type(generate) is int and generate >= trigger, "invalid clock timestamps")
    before = (generate-trigger)/1e9
    ttft = timing["prompt_to_first_token_s"]
    require(finite(ttft) and ttft >= 0, "invalid prompt clock")
    near(timing["trigger_to_generate_call_s"], before, "pre-generation clock differs")
    near(timing["trigger_to_first_token_s"], before+ttft, "first-token clock formula differs")
    require(finite(timing["trigger_to_generate_return_s"])
            and timing["trigger_to_generate_return_s"] + 1e-6 >= before+ttft, "full-generation clock ends before first token")
    return before+ttft


def compare_pending(raw, final):
    require(raw.get("native_evidence") is False and raw.get("native_gate_status") == "pending_final_run_observer",
            "immutable trial did not preserve pending native gate")
    require(final.get("native_evidence") is True and final.get("native_gate_status") == "verified_by_final_run_observer",
            "final native gate did not pass")
    ignored = {"native_evidence", "native_gate_status"}
    require({k: v for k, v in raw.items() if k not in ignored}
            == {k: v for k, v in final.items() if k not in ignored}, "immutable trial differs from final row")


def check_source_bytes(actual, measured_hash, git_bytes=None):
    canonical = actual.replace(b"\r\n", b"\n")
    allowed = {digest(canonical), digest(canonical.replace(b"\n", b"\r\n"))}
    require(measured_hash in allowed, "measured source hash differs from canonical LF/CRLF forms")
    if git_bytes is not None:
        require(canonical == git_bytes.replace(b"\r\n", b"\n"), "canonical source differs from Git baseline")
    return digest(canonical)


def verify_sources(source_manifest, provenance, repository):
    require(source_manifest["source_commit"] == provenance["source_commit"] == BASELINE, "runtime baseline differs")
    require(provenance["harness_sha256"] == HARNESS_SHA, "measured harness identity differs")
    try:
        git_available = subprocess.run(["git", "-C", str(repository), "rev-parse", "--git-dir"],
                                       capture_output=True, check=False).returncode == 0
        if git_available:
            git_available = all(subprocess.run(["git", "-C", str(repository), "cat-file", "-e", commit+"^{commit}"],
                                              capture_output=True, check=False).returncode == 0
                                for commit in (BASELINE, HARNESS_COMMIT))
    except FileNotFoundError:
        git_available = False
    canonical_hashes = {}
    for name, expected in source_manifest["source_files"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        git_bytes = None
        if git_available:
            completed = subprocess.run(["git", "-C", str(repository), "show", BASELINE + ":" + name],
                                       capture_output=True, check=False)
            require(completed.returncode == 0, "Git baseline source missing: " + name)
            git_bytes = completed.stdout
        actual = git_bytes if git_bytes is not None else (repository/name).read_bytes()
        canonical_hashes[name] = check_source_bytes(actual, expected, git_bytes)
    for name, expected in provenance["source_files"].items():
        require(source_manifest["source_files"].get(name) == expected, "per-run source hash differs")
    git_bytes = None
    if git_available:
        completed = subprocess.run(["git", "-C", str(repository), "show", HARNESS_COMMIT + ":bench/ccgrid_b2.py"],
                                   capture_output=True, check=False)
        require(completed.returncode == 0, "Git harness baseline missing")
        git_bytes = completed.stdout
    canonical_hashes["bench/ccgrid_b2.py"] = check_source_bytes(
        git_bytes if git_bytes is not None else (repository/"bench/ccgrid_b2.py").read_bytes(), HARNESS_SHA, git_bytes)
    return dict(canonical_source_files_verified=True, git_objects_verified=git_available,
                git_object_verification="verified" if git_available else "unavailable: archive/shallow checkout lacks Git objects",
                canonical_lf_sha256=canonical_hashes,
                measured_hash_scope="exact LF or CRLF source bytes; raw evidence hashes are never normalized")


def verify_inputs(root, result, name, repository):
    provenance = result["provenance"]
    paths = {"input_manifest": root/"inputs/input-manifest.json", "source_manifest": root/"inputs/source-manifest.json",
             "fixture": root/"inputs/fixtures.json", "references": root/"inputs/references.json",
             "switchplan": root/"inputs/interior.switchplan.json"}
    for key, path in paths.items():
        require(digest(path.read_bytes()) == provenance[key+"_sha256"], "input sidecar hash differs: " + key)
    config = root / "configs" / (name + ".json")
    require(digest(config.read_bytes()) == result["config_sha256"] and load(config) == result["config"],
            "staged configuration differs")
    inputs, source_manifest, provider, refs, plan = [load(paths[key]) for key in paths]
    require(inputs["schema"] == "ccgrid-b2-inputs-v1" and inputs["source_commit"] == BASELINE
            and inputs["files"] == provenance["input_files"], "off-card input manifest differs")
    require(len(source_manifest["source_files"]) == 80 and len(provenance["source_files"]) == 7,
            "source inventory incomplete")
    require(len(inputs["files"]) == 25 and len({entry["path"] for entry in inputs["files"]}) == 25,
            "off-card input inventory incomplete or duplicated")
    for entry in inputs["files"]:
        require(PurePosixPath(entry["path"]).is_absolute() and type(entry["bytes"]) is int and entry["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "invalid full-file input identity")
    require(all(sum(entry.get("role") == role for entry in inputs["files"]) == 6 for role in ("base", "target")),
            "base/target inventory differs")
    source_verification = verify_sources(source_manifest, provenance, repository)
    require(provider["schema"] == "onggi.binding-fixtures.v1", "provider schema differs")
    case = next(case for case in provider["cases"] if case["name"] == "native-interior-projection")
    require(case["expect_mode"] == "units" and case["reference_role"] == "sparse", "fixture selection differs")
    for role, directory in (("base", case["base_dir"]), ("sparse", case["target_dir"])):
        ref = refs[role]
        require(ref["status"] == "completed" and ref["engine_closed"] is True and ref["source_commit"] == FRESH_SOURCE
                and ref["source_dirty"] == "" and ref["checkpoint"] == directory and ref["variant"] == "B4-S256-g2"
                and ref["continuation_length"] == 8 and ref["counts"] == COUNTS, "historical fresh reference differs")
        require(provenance["reference_assignments"][role] == ref["assignment"], "reference assignment differs")
        for values in (ref["tokens"], ref["prompts"]):
            require(isinstance(values, list) and len(values) == 4 and all(isinstance(lane, list) and len(lane) == 8
                    and all(type(token) is int and token >= 0 for token in lane) for lane in values),
                    "reference token/prompt shape differs")
        inventory_role = "base" if role == "base" else "target"
        require(all(PurePosixPath(entry["path"]).parent == PurePosixPath(directory)
                    for entry in inputs["files"] if entry.get("role") == inventory_role), "artifact input paths differ")
    indexed = {entry["path"]: entry for entry in inputs["files"]}
    for path, expected in ((result["config"]["fixtures"], provenance["fixture_sha256"]),
                           (result["config"]["references"], provenance["references_sha256"]),
                           (case["switchplan"], provenance["switchplan_sha256"])):
        require(indexed[path]["sha256"] == expected, "measured input path/hash differs from off-card manifest")
    require(refs["base"]["prompts"] == refs["sparse"]["prompts"], "historical prompts differ")
    require(plan["schema"] == "onggi.switchplan.v1" and result["config"]["variant"] == "B4-S256-g2",
            "switch plan/configuration differs")
    return refs, plan, source_verification


def verify_job(root, name, result):
    prefix = root/"control"/("cdx-b2-"+name)
    job, created, status, cleanup = [load(Path(str(prefix)+suffix))
                                    for suffix in ("-job.json", "-create.json", "-latest-status.json", "-cleanup.json")]
    require(job["metadata"]["name"] == created["name"] == cleanup["name"] == "cdx-b2-"+name,
            "Job identity differs")
    require(created["job_uid"] == status["job_uid"] == cleanup["job_uid"], "Job UID differs")
    require(cleanup["job_absent"] and cleanup["pods_absent"], "allocation cleanup not verified")
    require(len(status["pods"]) == 1 and len(cleanup["observations"]) == 1, "Job did not have exactly one Pod")
    pod = status["pods"][0]
    require(pod["uid"] == result["provenance"]["assignment"]["pod_uid"] and pod["phase"] == "Succeeded",
            "measured Pod UID differs")
    require(cleanup["observations"][0]["uid"] == pod["uid"]
            and cleanup["observations"][0]["phase"] == "Succeeded", "cleanup observed a different/nonterminal Pod")
    container = next(item for item in pod["containers"] if item["name"] == "run")
    require(container["restartCount"] == 0 and container["state"]["terminated"]["exitCode"] == 0,
            "native container failed/restarted")
    main = next(item for item in job["spec"]["template"]["spec"]["containers"] if item["name"] == "run")
    require(main["resources"]["limits"]["furiosa.ai/rngd"] == "1" and "--config" in main["command"]
            and main["command"][-1].endswith("/"+name+"-config.json"), "Job allocation/command differs")
    require(job["spec"]["backoffLimit"] == 0 and job["spec"]["template"]["spec"]["restartPolicy"] == "Never",
            "Job could retry unrecorded native attempts")
    return dict(job_uid=created["job_uid"], pod_uid=pod["uid"], terminal_pod_succeeded=True,
                cleanup_verified=True, job_controller_completion_observed=status["status"].get("succeeded") == 1)


def verify_trial(row, raw, result, refs, document):
    compare_pending(raw, row)
    require(row["status"] == "completed" and row["tokens_equal_historical_fresh"] is True
            and row["provenance"] == result["provenance"], "trial gates/provenance differ")
    require(row["tokens"] == refs["sparse"]["tokens"] and row["base_tokens"] == refs["base"]["tokens"],
            "base/target fresh tokens differ")
    arm, operations = row["arm"], row["operations"]
    action = "apply" if arm == "b2" else "reload"
    require(set(operations) == {"prepare_base_outside_trigger", "plan", action, "sync", "generate_full_continuation"},
            "operation inventory differs")
    for name, operation in operations.items():
        require(finite(operation["wall_s"]) and operation["wall_s"] >= 0, "invalid operation duration")
        expected = COUNTS if name in ("prepare_base_outside_trigger", "generate_full_continuation") else dict(
            compilation_attempts=0, native_calls=0, native_output_checks=0)
        require(operation["native_counts"] == expected, "native operation/compilation counts differ")
        before, after = operation["diagnostics_before"], operation["diagnostics_after"]
        for key, value in after.items():
            if finite(value):
                near(operation["diagnostics"][key], value-before[key], "operation diagnostic delta differs")
    total = verify_clock(row)
    require(row["timings"]["prompt_to_first_token_s"] <= operations["generate_full_continuation"]["wall_s"]+1e-6,
            "prompt TTFT exceeds full continuation")
    require(sum(operations[name]["wall_s"] for name in ("plan", action, "sync"))
            <= row["timings"]["trigger_to_generate_call_s"]+1e-6, "trigger omits timed operation")
    update = row["update_plan"]
    require(update["schema"] == "onggi.updateplan.v1" and update["mode"] == "units"
            and update["base_manifest"] == document["base_manifest"] and update["manifest"] == document["manifest"],
            "runtime did not admit the exported plan")
    require_coverage(document, update["units"])
    expected_unit = dict(file="model-00001-of-00004.safetensors", offset=1054274864, length=1949696,
                         unit="model.layers.0.mlp.down_proj.weight@125:193", source="target")
    require(update["units"] == [expected_unit] and update["bytes"] == 1949696, "admitted row unit differs")
    require(row["resident_after"]["manifest"] == document["manifest"] and row["resident_after"]["state"] == "ready",
            "target residency differs")
    device = result["expected_native_device"]
    for identities in (row["identities_before"], row["identities_after"]):
        require(identities["buffers"] and identities["programs"], "missing buffer/program identity observations")
        require(all(buffer["device"] == device["type"]+":"+str(device["index"])
                    and type(buffer["data_ptr"]) is int and buffer["data_ptr"] > 0
                    for buffer in identities["buffers"].values()), "buffer device identity differs")
    if arm == "b2":
        require(row["serving_allocations_programs_retained"] is True and row["identities_before"] == row["identities_after"],
                "B2 did not retain allocations/programs")
        verification = row["verification"]
        require(verification["ok"] is True and verification["op"] == "apply" and verification["mode"] == "units"
                and verification["manifest"] == document["manifest"] and verification["bytes_written"] == 1949696,
                "B2 target/readback verification differs")
        checked = verification["units"]
        require(len(checked) == 1 and all(checked[0][key] == expected_unit[key] for key in ("file", "offset", "length"))
                and re.fullmatch(r"[0-9a-f]{64}", checked[0]["sha256"]), "verified unit differs from admitted unit")
        parent = row["identities_before"]["buffers"]["shared/0/wd"]
        require(parent["shape"] == [2, 14336, 4096], "serving parent layout differs")
        require(row["logical_device_written_bytes"] == 68*parent["shape"][1]*2 == 1949696,
                "logical row-write bytes differ")
        expected_readback = 2*math.prod(parent["shape"])*2
        require(operations["apply"]["diagnostics"]["readback_calls"] == 2
                and row["readback_payload_bytes_during_trigger"] == expected_readback == 469762048,
                "whole-parent journaling/readback bytes differ")
        require(row["diagnostics_during_trigger"]["readback_calls"] == 2, "trigger readback calls differ")
    else:
        require(row["logical_device_written_bytes"] == 15011577856
                and row["readback_payload_bytes_during_trigger"] == 0, "reload logical/readback payload differs")
    require(row["readback_payload_bytes_during_trigger"] == sum(operation["diagnostics"]["readback_payload_bytes"]
                for name, operation in operations.items() if name != "prepare_base_outside_trigger"),
            "readback delta included base preparation or omitted an operation")
    return dict(trigger_to_first_token_s=total, plan_s=operations["plan"]["wall_s"],
                apply_or_reload_s=operations[action]["wall_s"], prompt_ttft_s=row["timings"]["prompt_to_first_token_s"],
                generate_full_continuation_s=operations["generate_full_continuation"]["wall_s"],
                logical_device_written_bytes=row["logical_device_written_bytes"],
                readback_payload_bytes=row["readback_payload_bytes_during_trigger"],
                readback_s=row["diagnostics_during_trigger"]["readback_s"])


def empty_run(phase, status, reason):
    return dict(status=status, phase=phase, verified=False, n_per_arm=0, observed_trials=0,
                reason=reason, excluded_from_main=phase == "smoke", arms={
                    arm: {metric: statistics_for([]) for metric in METRICS} for arm in ("b2", "reload")})


def verify_run(root, name, phase, repository):
    run = root/"runs"/name
    result = load(run/"result.json")
    require(result["status"] == "completed" and result["native_evidence"] is True
            and result["native_outputs_verified"] is True and result["tokens_equal_current_same_device_reload"] is True,
            "run native/final gates failed")
    require(result["config"]["phase"] == phase and result["config"]["source_commit"] == BASELINE,
            "phase/source configuration differs")
    pairs = 1 if phase == "smoke" else 5
    expected = [(pair, arm) for pair in range(1, pairs+1)
                for arm in (("b2", "reload") if pair % 2 else ("reload", "b2"))]
    require([(row["pair"], row["arm"]) for row in result["trials"]] == expected,
            "trial count/order differs; missing/duplicated observations cannot be filtered")
    require(result["native_counts"] == dict(compilation_attempts=0, native_calls=pairs*1020, native_output_checks=pairs*1020),
            "total native calls/compile guard differs")
    require(verify_export(run) == {"result.json", "events.jsonl", *[f"pair-{pair:02d}-{arm}.json" for pair, arm in expected]},
            "export omitted or added raw trials")
    refs, document, source_verification = verify_inputs(root, result, name, repository)
    job_verification = verify_job(root, name, result)
    assignment = result["provenance"]["assignment"]
    require(len(assignment["allocation_uuid"]) == 1 and assignment["host_index"] in assignment["visible_cards"]
            and result["expected_native_device"] == dict(type="furiosa", index=assignment["torch_ordinal"]),
            "native allocation/device mapping differs")
    require(result["sdk"] == "2026.4.0" and result["provenance"]["packages"]["torch"] == "2.10.0+cpu"
            and result["provenance"]["packages"]["furiosa-torch"] == "2026.4.0", "native package versions differ")
    samples = {arm: [] for arm in ("b2", "reload")}
    for row in result["trials"]:
        require(row["phase"] == phase, "trial stratum differs")
        raw = load(run/f"pair-{row['pair']:02d}-{row['arm']}.json")
        samples[row["arm"]].append(verify_trial(row, raw, result, refs, document))
    events = [json.loads(line) for line in (run/"events.jsonl").read_bytes().splitlines()]
    require(events[-1] == result and len(events) == 2+3*len(expected)+pairs, "incremental event history differs")
    complete = [event for event in events if event.get("schema") == "ccgrid-current-b2-trial-v1"
                and event.get("status") == "completed"]
    require(len(complete) == len(expected), "event history omitted trial completions")
    for raw, final in zip(complete, result["trials"]):
        compare_pending(raw, final)
    parity = [event for event in events if "tokens_equal_current_same_device_reload" in event and "pair" in event]
    require([event["pair"] for event in parity] == list(range(1, pairs+1))
            and all(event["tokens_equal_current_same_device_reload"] is True for event in parity), "current-pair gate missing")
    for pair in range(1, pairs+1):
        paired = [row for row in result["trials"] if row["pair"] == pair]
        require(paired[0]["tokens"] == paired[1]["tokens"], "current same-target pair tokens differ")
    return dict(status="completed", phase=phase, verified=True, n_per_arm=pairs, observed_trials=len(expected),
                excluded_from_main=phase == "smoke", job_uid=job_verification["job_uid"],
                job_verification=job_verification, native_counts=result["native_counts"],
                source_verification=source_verification,
                result_sha256=digest((run/"result.json").read_bytes()),
                arms={arm: {metric: statistics_for([sample[metric] for sample in samples[arm]])
                            for metric in METRICS} for arm in samples})


def verify(root):
    root = Path(root)
    summary = dict(schema="ccgrid-b2-timing-summary-v1", status="incomplete", verified=False,
        verifier_sha256=digest(Path(__file__).read_bytes()),
        smoke=empty_run("smoke", "incomplete", "smoke result missing"),
        main=empty_run("measure", "incomplete", "main result missing"),
        statistics="median and inclusive interpolated quartiles; smoke excluded from main",
        byte_scope="logical serving writes and whole-parent readback payload, not physical DMA",
        artifact_hash_scope="off-card full-file manifest identity; bulk weights are not rehashed offline",
        timing_scope="per-trial reconstructed trigger-to-native-first-token; overlapping diagnostics never summed",
        overlapping_diagnostics=["readback_s is included in apply/reload wall time, not an additional phase"])
    try:
        package = load(root/"package-manifest.json")
        require(package.get("schema") and isinstance(package["files"], list), "invalid package manifest")
        entries = {entry["path"]: entry for entry in package["files"]}
        require(len(entries) == len(package["files"]), "duplicate package entry")
        for entry in package["files"]:
            verify_file(root, entry)
        summary["package_manifest_sha256"] = digest((root/"package-manifest.json").read_bytes())
        summary["package_files"] = len(package["files"])
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
                  and path.name != "package-manifest.json"}
        require(actual == set(entries), "package file inventory omits or adds files")
        unexpected = [path.name for path in (root/"runs").glob("*") if path.is_dir()
                      and path.name not in ("smoke-01", "main-01")]
        require(not unexpected, "unaccounted run attempts: " + repr(unexpected))
        repository = Path(__file__).resolve().parents[1]
        for key, name, phase in (("smoke", "smoke-01", "smoke"), ("main", "main-01", "measure")):
            path = root/"runs"/name/"result.json"
            if not path.exists():
                continue
            try:
                summary[key] = verify_run(root, name, phase, repository)
            except Exception as error:
                summary[key] = empty_run(phase, "failed", str(error))
                summary[key]["observed_trials"] = len(load(path).get("trials", []))
        states = [summary[key]["status"] for key in ("smoke", "main")]
        summary["status"] = "failed" if "failed" in states else "completed" if states == ["completed"]*2 else "incomplete"
        summary["verified"] = summary["status"] == "completed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    summary = verify(args.root)
    text = json.dumps(summary, indent=2, allow_nan=False)+"\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return {"completed": 0, "incomplete": 2, "failed": 1}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
