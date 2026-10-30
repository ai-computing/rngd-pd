"""Verify the native drain/rollback-failure record without importing an SDK."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


MEASURED_HARNESS = "3e4da7803b4b2db1c4a4f6b81eb4ce20db4cf99249244aea5e1c3ad4389664bd"
RUNTIME = "a2ebc3281df4173a886bc6a035f2483eaa360bf5"
GENERATION_COUNTS = dict(native_calls=255, native_output_checks=255, compilation_attempts=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_bytes(root, entry):
    relative = PurePosixPath(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe evidence path")
    raw = root.joinpath(*relative.parts).read_bytes()
    require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"], "raw evidence hash differs: " + str(relative))
    return raw


def verify(root, repository):
    package = load(root / "package-manifest.json")
    for entry in package["files"]:
        file_bytes(root, entry)
    run = root / "runs/rngd-faults-native-01"
    exported = load(run / "export-manifest.json")
    archive = (run / "small-evidence.json.gz").read_bytes()
    require(len(archive) == exported["archive"]["bytes"] and digest(archive) == exported["archive"]["sha256"],
            "original export archive hash differs")
    packed = json.loads(gzip.decompress(archive))
    entries = {entry["path"]: entry for entry in exported["files"]}
    require(packed["trial"] == "rngd-faults-native-01"
            and len(packed["files"]) == exported["archive"]["files"] == 2
            and set(entries) == {entry["path"] for entry in packed["files"]}, "export archive inventory differs")
    for entry in packed["files"]:
        raw = base64.b64decode(entry["base64"], validate=True)
        require(len(raw) == entry["bytes"] and digest(raw) == entry["sha256"]
                and raw == file_bytes(run, entries[entry["path"]]), "archive/extracted record differs")
    require((run / "result.json").read_bytes() == (run / "operational.json").read_bytes(), "final operational/result records differ")
    result = load(run / "result.json")
    require(result["schema"] == "rngd-native-drain-rollback-failure-v1" and result["status"] == "completed"
            and result["native_evidence"] and result["engine_closed"], "native fault run did not pass")
    harness = (root / "harness/run_rngd_faults.py").read_bytes()
    provenance = load(root / "harness/run_rngd_faults.provenance.json")
    require(digest(harness) == result["harness_sha256"] == provenance["sha256"] == MEASURED_HARNESS
            and len(harness) == provenance["bytes"], "measured harness differs; do not substitute the later UID-only edit")
    require(result["runtime_commit"] == package["runtime_commit"] == RUNTIME, "frozen runtime identity differs")
    prior = repository / "experiments/data/binding-20261030"
    native02 = load(prior / "runs/rngd-b2-native-02/result.json")
    require(result["source_files"] == native02["source_files"], "runtime differs from the original B2 pass")
    for name, expected in result["source_files"].items():
        require(digest((repository / name).read_bytes().replace(b"\r\n", b"\n")) == expected,
                "current runtime differs from measured native LF source: " + name)
    input_values = {}
    for name, entry in package["reused_inputs"].items():
        raw = (repository / entry["path"]).read_bytes()
        require(digest(raw) == entry["sha256"] == result["inputs"][name]["sha256"], "committed input hash differs: " + name)
        input_values[name] = json.loads(raw)
    plan, refs = input_values["plan"], input_values["references"]
    for role in ("base", "sparse"):
        require(result["references"][role] == refs[role] and refs[role]["status"] == "completed"
                and refs[role]["engine_closed"] and refs[role]["source_dirty"] == ""
                and refs[role]["source_commit"] == "f5fa13377a3d967ba3d4022fb73ae59b1c87441c"
                and refs[role]["counts"] == GENERATION_COUNTS, "independent fresh reference differs")
    require(result["assignment"]["host_index"] == native02["assignment"]["host_index"] == 0
            and result["assignment"]["torch_ordinal"] == native02["assignment"]["torch_ordinal"] == 0
            and result["assignment"]["allocation_uuid"] == native02["assignment"]["allocation_uuid"]
            and result["expected_native_device"] == dict(type="furiosa", index=0), "physical/logical assignment differs")
    require(result["native_outputs_verified"] and result["native_counts"] == dict(
        native_calls=1275, native_output_checks=1275, compilation_attempts=0), "native output/compile counters differ")
    expected_gates = [("loaded-base", "base"), ("whole-request-held-base", "base"),
                      ("drained-apply-target", "sparse"), ("normal-rollback-base", "base"), ("second-apply-target", "sparse")]
    require([(row["operation"], row["reference_role"]) for row in result["generations"]] == expected_gates,
            "native generation gate sequence differs")
    for row in result["generations"]:
        require(row["status"] == "completed" and row["tokens_equal_reference"]
                and row["tokens"] == refs[row["reference_role"]]["tokens"]
                and row["native_counts"] == GENERATION_COUNTS
                and row["identities_before"] == row["identities_after"] == result["identities_loaded"],
                "generation parity, native counts or serving identities differ")
    drain = result["drain"]
    require(drain["status"] == "completed" and drain["worker_joined"] and drain["completion_after_request_release"]
            and not drain["finished_before_generation"] and not drain["finished_after_generation_before_release"]
            and drain["base_bytes_unchanged_before_release"]
            and drain["attempt_perf_ns"] < drain["request_release_marker_perf_ns"] < drain["finished_perf_ns"],
            "native whole-request drain gate failed")
    update = result["update"]
    require(update["mode"] == "units" and len(update["units"]) == 1
            and update["base_manifest"] == plan["base_manifest"] and update["manifest"] == plan["manifest"], "wrong authentic update")
    unit = update["units"][0]
    base_hashes, target_hashes = result["base_actual_d2h_sha256"], result["target_actual_d2h_sha256"]
    require(result["normal_rollback_actual_d2h_sha256"] == base_hashes, "normal rollback did not restore actual base bytes")
    for key, operation, manifest, hashes in (("drained_apply", "apply", plan["manifest"], target_hashes),
                                           ("normal_rollback", "rollback", plan["base_manifest"], base_hashes),
                                           ("second_apply", "apply", plan["manifest"], target_hashes)):
        verification = result[key]
        require(verification["schema"] == "onggi.verification.v1" and verification["ok"]
                and verification["op"] == operation and verification["manifest"] == manifest
                and verification["bytes_written"] == update["bytes"] == unit["length"] == 1949696
                and verification["units"] == [dict(file=unit["file"], offset=unit["offset"], length=unit["length"],
                                                    sha256=hashes[unit["unit"]])], "transaction verification differs")
    fault = result["rollback_failure"]
    require(fault["status"] == "completed" and fault["actual_partial_restore_verified"] and fault["request_refused"]
            and fault["engine_closed"] and fault["no_runtime_generation_after_failure"]
            and fault["resident_after_failure"]["state"] == "empty"
            and fault["resident_after_failure"]["manifest"] is None
            and fault["error"]["message"] == "rollback restore failed; engine unloaded", "rollback did not fail closed")
    require(fault["actual_mixed_sha256"] == fault["expected_mixed_sha256"]
            and fault["actual_mixed_sha256"] not in (fault["base_sha256"], fault["target_sha256"])
            and fault["base_sha256"] == base_hashes[unit["unit"]]
            and fault["target_sha256"] == target_hashes[unit["unit"]], "actual partial rollback mixed-byte proof differs")
    partial = fault["partial_unit"]
    require(partial["unit"] == "model.layers.0.mlp.down_proj.weight@125:159"
            and partial["file"] == unit["file"] and partial["offset"] == unit["offset"]
            and partial["length"] == 974848, "actual partial restore unit differs")
    diagnostics = result["final_diagnostics"]
    require(diagnostics == fault["resident_after_failure"]["diagnostics"]
            and diagnostics["readback_calls"] == 11 and diagnostics["readback_payload_bytes"] == 2583691264,
            "actual parent-readback payload accounting differs")
    created, terminal, cleanup, job = (load(run / name) for name in ("create.json", "terminal-api.json", "cleanup.json", "job.json"))
    require(created["uid"] == terminal["job"]["uid"] == cleanup["job_uid"] and len(terminal["pods"]) == 1, "Job UID differs")
    pod = terminal["pods"][0]
    require(pod["metadata"]["uid"] == cleanup["pod_uid"] == result["pod_uid"] == result["assignment"]["pod_uid"]
            and any(owner["uid"] == created["uid"] for owner in pod["metadata"]["ownerReferences"]), "Pod owner UID differs")
    require(cleanup["job_absent"] and cleanup["pod_absent"] and cleanup["terminal_phase"] == pod["phase"] == "Succeeded"
            and all(c["state"]["terminated"]["exitCode"] == 0 and c["restartCount"] == 0 for c in cleanup["containers"]),
            "owned Job/Pod cleanup or native exit failed")
    require(job["metadata"]["name"] == "cdx-rngd-faults-native-01" and job["spec"]["backoffLimit"] == 0
            and job["spec"]["activeDeadlineSeconds"] == 1800 and job["spec"]["ttlSecondsAfterFinished"] == 600,
            "bounded native Job specification differs")
    spec = job["spec"]["template"]["spec"]
    for container in spec["containers"] + spec["initContainers"]:
        require(container["resources"]["requests"]["furiosa.ai/rngd"] == container["resources"]["limits"]["furiosa.ai/rngd"] == "1",
                "native Job did not request/limit exactly one card")
    return dict(schema="rngd-fault-evidence-summary.v1", verified=True, native_generation_gates=5,
                native_counts=result["native_counts"], whole_request_drain=True, actual_partial_rollback_failure=True,
                actual_mixed_sha256=fault["actual_mixed_sha256"], fail_closed=True,
                measured_harness_sha256=MEASURED_HARNESS, current_runtime_matches_native_canonical_source=True,
                owned_uid_cleanup_verified=True, readback_payload_bytes=diagnostics["readback_payload_bytes"],
                scope="n=1 compliance; readback payload is not physical DMA; no timing claim")


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository / "experiments/data/binding-faults-20261030")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, repository), indent=2))
