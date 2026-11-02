#!/usr/bin/env python3
"""One E6 B/S switch: python bench/ccgrid_e6.py trial --config FILE --out NEWFILE.

The config supplies trial_id, arm (full|delta|resident), cache_state
(empty|prepopulated), onggi_binary, store, base_ref, target_ref,
consumable_base_source, immutable_checkpoint, base_cache_dir, base_variant,
target_variant, chip, host_workers, intra_op, reference_file,
pod_metadata_file, clock_basis and a previously absent work_dir.
References contain prompts, base_prompts, tokens and base_tokens (B integer
lists each). Bundle layout is checkpoint/ and cache/graphs/. All input paths
should be absolute. Additional config provenance is retained verbatim.

Every arm starts with the same freshly constructed base engine and independently
copied consumable base files. The live base reads immutable_checkpoint, never
the directory Onggi claims. Target graphs are not prepared before the trigger.
Full/delta restart the engine subprocess after delivery; resident retains it.
Prepopulated means an actual target pull filled this trial's fresh chunk cache.
OS page cache remains uncontrolled/shared; this is not a globally cold start.

Both actual API Pod creation-to-token and same-host perf_counter trigger-to-token
are recorded. Native generate() supplies the first generated token timestamp;
the complete continuation is checked, and its remaining generation is excluded
from first-token latency. No retry, allocator wait subtraction, HTTP echo token,
compilation, additional target warmup or aggregate statistics are involved.
Failure evidence and all worker files/logs remain in the owned work directory.
The caller must provide an exclusive allocated card and a bounded Kubernetes Job.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


ENTRY_PERF_NS = time.perf_counter_ns()


def stamp():
    return dict(perf_ns=time.perf_counter_ns(), wall_ns=time.time_ns())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest())


def first_token_partition(trigger_ns, delivered_ns, target):
    points = (trigger_ns, delivered_ns, target["generation_start_perf_ns"],
              target["first_token_perf_ns"])
    if list(points) != sorted(points):
        raise ValueError("first-token timestamps are not ordered on the shared host clock")
    return dict(artifact_delivery_s=(points[1] - points[0]) / 1e9,
        delivery_to_generation_start_s=(points[2] - points[1]) / 1e9,
        target_prompt_to_first_token_s=(points[3] - points[2]) / 1e9)


def validate_worker_identity(base, target, arm):
    # Windows virtualenv launchers can have a different PID from the interpreter.
    base_id = (base["pid"], base["entry_perf_ns"])
    target_id = (target["pid"], target["entry_perf_ns"])
    if (base_id == target_id) != (arm == "resident"):
        raise RuntimeError("actual worker identity does not match the requested restart/resident arm")
    return dict(base=dict(pid=base_id[0], entry_perf_ns=base_id[1]),
                target=dict(pid=target_id[0], entry_perf_ns=target_id[1]),
                same_interpreter=base_id == target_id)


def pull(cfg, destination, cache, record, base_dir=None):
    argv = [cfg["onggi_binary"], "pull", "--store", cfg["store"],
            "--cache", str(cache), "--json"]
    if base_dir is not None:
        argv += ["--base", cfg["base_ref"], "--base-dir", str(base_dir)]
    argv += [cfg["target_ref"], str(destination)]
    record.update(argv=argv, started=stamp())
    process = subprocess.run(argv, capture_output=True, text=True, check=False)
    record.update(finished=stamp(), returncode=process.returncode,
                  stdout=process.stdout, stderr=process.stderr)
    if process.returncode:
        raise RuntimeError("onggi pull failed; raw stdout/stderr retained")
    record["report"] = json.loads(process.stdout)
    stats = record["report"]["stats"]
    if base_dir is not None and not (stats.get("base") or stats.get("local_base")):
        raise RuntimeError("base-aware pull did not use the offered base")
    return record["report"]


def validate(cfg):
    # A missing Pod timestamp fails before any device/work-directory operation.
    metadata = read_json(cfg["pod_metadata_file"]).get("metadata", {})
    if not all(metadata.get(k) for k in ("creationTimestamp", "uid", "name")):
        raise ValueError("actual Pod metadata requires creationTimestamp, uid and name")
    created = datetime.fromisoformat(metadata["creationTimestamp"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise ValueError("Pod creationTimestamp must include a UTC offset")
    if not cfg.get("clock_basis"):
        raise ValueError("clock_basis must describe the API/host UTC synchronization basis")
    if cfg.get("arm") not in ("full", "delta", "resident"):
        raise ValueError("arm must be full, delta or resident")
    if cfg.get("cache_state") not in ("empty", "prepopulated"):
        raise ValueError("cache_state must be empty or prepopulated")
    work = Path(cfg["work_dir"]).resolve()
    if work.exists():
        raise FileExistsError("work_dir already exists; refusing to consume it")
    for key in ("trial_id", "onggi_binary", "store", "base_ref", "target_ref"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError("missing nonempty config field: " + key)
    for key in ("base_ref", "target_ref"):
        if cfg[key].startswith("-"):
            raise ValueError("artifact refs cannot begin with '-'")
    if type(cfg.get("chip")) is not int or cfg["chip"] < 0:
        raise ValueError("chip must be an explicit nonnegative allocated ordinal")
    if cfg.get("host_workers", 2) not in (1, 2) or type(cfg.get("host_workers", 2)) is not int:
        raise ValueError("host_workers must be 1 or 2")
    if type(cfg.get("intra_op", 4)) is not int or cfg.get("intra_op", 4) < 1:
        raise ValueError("intra_op must be positive")
    from rngd_pd.swap.variants import Variant
    base, target = (Variant.parse(cfg[k]) for k in ("base_variant", "target_variant"))
    if base == target or base.group != target.group:
        raise ValueError("E6 needs different B/S variants sharing one group")
    reference = read_json(cfg["reference_file"])
    for prefix, variant in (("base_", base), ("", target)):
        for suffix in ("prompts", "tokens"):
            rows = reference.get(prefix + suffix)
            if (not isinstance(rows, list) or len(rows) != variant.B or not rows
                    or not all(isinstance(row, list) and row and all(type(t) is int and t >= 0
                        for t in row) for row in rows) or len({len(r) for r in rows}) != 1):
                raise ValueError("reference needs B nonempty equal-length integer lists: " + prefix + suffix)
        if len(reference[prefix + "prompts"][0]) + len(reference[prefix + "tokens"][0]) > variant.S:
            raise ValueError("reference prompt/continuation exceed variant context")
    for key in ("consumable_base_source", "immutable_checkpoint", "base_cache_dir"):
        source = Path(cfg[key]).resolve()
        if not source.is_dir() or source == work or work.is_relative_to(source) or source.is_relative_to(work):
            raise ValueError("work_dir must be independent of existing input directory: " + key)
    source = Path(cfg["consumable_base_source"])
    if not (source / "checkpoint").is_dir() or not (source / "cache/graphs").is_dir():
        raise ValueError("base bundle requires checkpoint/ and cache/graphs/")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("consumable base source must contain real files, not symlinks")
    return metadata, created.timestamp(), reference


def weight_buffers(engine):
    tensors = [("group%d/%s" % (i, name), tensor) for i, group in enumerate(engine.shared)
               for name, tensor in sorted(group.items())]
    tensors.append(("lm_head", engine.wlm_d))
    if len(tensors) != engine.ngroups * 7 + 1:
        raise RuntimeError("resident shared-weight allocation count differs from runtime layout")
    return [dict(name=name, object_id=id(t), data_ptr=t.data_ptr(), shape=list(t.shape),
                 dtype=str(t.dtype), device=str(t.device), bytes=t.numel() * t.element_size())
            for name, t in tensors]


def graph_buffers(engine):
    return {v.name: {name: id(state[name]) for name in ("m", "lmh")}
            for v, state in engine.variants.items()}


def validate_native_outputs(value, expected_device):
    import torch
    devices = set()
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device != expected_device:
                raise RuntimeError("native output device %s differs from allocated %s" % (item.device, expected_device))
            devices.add(str(item.device))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    if not devices:
        raise RuntimeError("native module returned no tensor outputs")
    return sorted(devices)


def observe_native(chip):
    """Count original operations; never replace native results or graph inputs."""
    from rngd_pd.engines import fsdk, artifact_cache
    from rngd_pd.swap.rngd import SwappableDecode
    import torch
    expected_device = torch.device(fsdk.dev_str(chip))
    counts = dict(compilation_attempts=0, native_calls=0, graph_loads=0, shared_uploads=0,
                  native_output_devices=[])
    def no_compile(*args, **kwargs):
        counts["compilation_attempts"] += 1
        raise RuntimeError("compilation forbidden in E6")
    fsdk.CompileModule.from_module = staticmethod(no_compile)
    original_call = fsdk.CompileModule.__call__
    def call(module, *args, **kwargs):
        counts["native_calls"] += 1
        output = original_call(module, *args, **kwargs)
        counts["native_output_devices"] = validate_native_outputs(output, expected_device)
        return output
    fsdk.CompileModule.__call__ = call
    original_load = artifact_cache.load
    def load(*args, **kwargs):
        counts["graph_loads"] += 1
        return original_load(*args, **kwargs)
    artifact_cache.load = load
    original_upload = SwappableDecode._upload_shared
    def upload(engine, *args, **kwargs):
        counts["shared_uploads"] += 1
        return original_upload(engine, *args, **kwargs)
    SwappableDecode._upload_shared = upload
    return counts, fsdk.SDK


def generate_target(engine, cfg, reference, counts, row):
    from rngd_pd.swap.variants import Variant
    target = Variant.parse(cfg["target_variant"])
    if target in engine.variants:
        raise RuntimeError("target variant was prepared before its first-use measurement")
    calls_before = counts["native_calls"]
    row["target_prepare_started"] = stamp()
    row["graph"] = engine.swap(target)
    engine._sync()
    row["target_prepared"] = stamp()
    row["generation_started"] = stamp()
    tokens, ttft = engine.generate(reference["prompts"], len(reference["tokens"][0]))
    engine._sync()
    row["generation_done_perf_ns"] = time.perf_counter_ns()
    row["generation_start_perf_ns"] = row["generation_started"]["perf_ns"]
    row["first_token_perf_ns"] = row["generation_start_perf_ns"] + round(ttft * 1e9)
    row["first_token_wall_ns"] = row["generation_started"]["wall_ns"] + round(ttft * 1e9)
    row.update(tokens=tokens, tokens_equal_reference=tokens == reference["tokens"],
        prompt_ttft_s=ttft, counts=dict(counts), weights=dict(engine.timings),
        weight_buffers_after=weight_buffers(engine), graph_buffers_after=graph_buffers(engine),
        active_variant=engine.active.name,
        first_token_timing_method="generate start + native engine-reported prompt TTFT")
    row["target_native_calls"] = counts["native_calls"] - calls_before
    row["expected_target_native_calls"] = (len(reference["prompts"][0]) + len(reference["tokens"][0]) - 1) * (engine.ngroups + 1)
    if (not row["tokens_equal_reference"] or counts["compilation_attempts"]
            or row["target_native_calls"] != row["expected_target_native_calls"]
            or counts["shared_uploads"] != 1):
        raise RuntimeError("target token/counter validity gate failed")


def worker(cfg, work, role):
    work = Path(work)
    row = dict(role=role, pid=os.getpid(), entry_perf_ns=ENTRY_PERF_NS,
               started=stamp(), status="failed")
    engine = None
    try:
        import torch
        torch.set_num_threads(cfg.get("intra_op", 4))
        from rngd_pd.swap.rngd import SwappableDecode
        from rngd_pd.swap.variants import Variant
        counts, sdk = observe_native(cfg["chip"])
        row.update(sdk=sdk, torch=str(torch.__version__), counts=counts)
        reference = read_json(cfg["reference_file"])
        checkpoint = cfg["immutable_checkpoint"] if role == "base" else str(work / "target/checkpoint")
        cache = cfg["base_cache_dir"] if role == "base" else str(work / "target/cache/graphs")
        row["construct_started"] = stamp()
        engine = SwappableDecode(checkpoint, group=Variant.parse(cfg["base_variant"]).group,
            chip=cfg["chip"], cache_dir=cache, allow_compile=False,
            host_workers=cfg.get("host_workers", 2))
        row["constructed"] = stamp()
        row.update(checkpoint=checkpoint, cache_dir=cache, weights=dict(engine.timings))
        if role == "base":
            row["base_graph"] = engine.swap(Variant.parse(cfg["base_variant"]))
            row["base_generation_started"] = stamp()
            tokens, ttft = engine.generate(reference["base_prompts"], len(reference["base_tokens"][0]))
            engine._sync()
            row.update(base_tokens=tokens, base_tokens_equal_reference=tokens == reference["base_tokens"],
                base_prompt_ttft_s=ttft, base_ready=stamp(),
                weight_buffers_before=weight_buffers(engine), graph_buffers_before=graph_buffers(engine),
                base_counts=dict(counts))
            expected_calls = (len(reference["base_prompts"][0]) + len(reference["base_tokens"][0]) - 1) * (engine.ngroups + 1)
            if counts["native_calls"] != expected_calls:
                raise RuntimeError("base native call count differs from expected graph execution")
            if not row["base_tokens_equal_reference"]:
                raise RuntimeError("base tokens differ from independent fresh reference")
            if Variant.parse(cfg["target_variant"]) in engine.variants:
                raise RuntimeError("target unexpectedly prepared during common setup")
            publish(work / "base-ready.json", row)
            action_file = work / "base-action.json"
            while not action_file.exists():
                time.sleep(0.01)
            action = read_json(action_file)
            if action["action"] == "target":
                engine.cache_dir = str(work / "target/cache/graphs")
                row["cache_dir"] = engine.cache_dir
                generate_target(engine, cfg, reference, counts, row)
                row["resident_weights_preserved"] = row["weight_buffers_before"] == row["weight_buffers_after"]
                row["base_graphs_preserved"] = all(row["graph_buffers_after"].get(v) == ids
                    for v, ids in row["graph_buffers_before"].items())
                if not row["resident_weights_preserved"] or not row["base_graphs_preserved"]:
                    raise RuntimeError("resident weight/base graph identities changed")
                row["status"] = "completed"
                publish(work / "target-result.json", row)
            elif action["action"] != "close":
                raise ValueError("unknown base worker action")
        else:
            generate_target(engine, cfg, reference, counts, row)
            row["status"] = "completed"
            publish(work / "target-result.json", row)
        row["status"] = "completed"
    except Exception as exc:
        row.update(error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        row["close_started"] = stamp()
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                row.update(status="failed", close_error=dict(type=type(exc).__name__, message=str(exc)))
        row["closed"] = stamp()
        publish(work / (role + "-finished.json"), row)
    return 0 if row["status"] == "completed" else 1


def wait_file(path, process, deadline):
    while not Path(path).exists():
        if process.poll() is not None:
            raise RuntimeError("native worker exited before " + Path(path).name)
        if time.monotonic() > deadline:
            raise TimeoutError("native worker exceeded trial deadline")
        time.sleep(0.02)
    return read_json(path)


def trial(cfg, result):
    metadata, created_s, reference = validate(cfg)
    work = Path(cfg["work_dir"]).resolve()
    result.update(pod={k: metadata[k] for k in ("name", "uid", "creationTimestamp")},
        clock_basis=cfg["clock_basis"], reference=reference,
        reference_file=fingerprint(cfg["reference_file"]),
        os_page_cache="uncontrolled/shared; application-cache state only",
        allow_compile=False, allocator_retries=0, events={}, phase_s={}, workers={}, pulls={})
    work.mkdir(parents=False)
    cfg = dict(cfg, work_dir=str(work))
    publish(work / "config.json", cfg)
    events = result["events"]
    events["setup_started"] = stamp()
    shutil.copytree(cfg["consumable_base_source"], work / "consumable-base")
    events["base_files_copied"] = stamp()
    cache = work / "node-cache"
    if cfg["cache_state"] == "prepopulated":
        pull(cfg, work / "cache-fill", cache, result["pulls"].setdefault("cache_prepopulation", {}))
        shutil.rmtree(work / "cache-fill")
    events["cache_setup_finished"] = stamp()
    deadline = time.monotonic() + cfg.get("worker_timeout_s", 3600)
    processes, logs = [], []
    def launch(role):
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--config",
                   str(work / "config.json"), "--role", role]
        log = (work / (role + ".log")).open("x", encoding="utf-8")
        logs.append(log)
        started = stamp()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        result["workers"][role] = dict(argv=command, pid=process.pid,
            launcher_pid=process.pid, pid_basis="Popen launcher; actual interpreter identity is in worker evidence",
            launched=started)
        processes.append((process, result["workers"][role]))
        return process
    try:
        base = launch("base")
        ready = wait_file(work / "base-ready.json", base, deadline)
        result["base"] = ready
        events["base_drained_barrier"] = stamp()
        events["trigger"] = stamp()
        pull(cfg, work / "target", cache, result["pulls"].setdefault("target", {}),
             base_dir=None if cfg["arm"] == "full" else work / "consumable-base")
        events["target_files_ready"] = stamp()
        if not (work / "target/checkpoint").is_dir() or not (work / "target/cache/graphs").is_dir():
            raise RuntimeError("delivered target lacks checkpoint/ or cache/graphs/")
        if cfg["arm"] == "resident":
            publish(work / "base-action.json", dict(action="target"))
            target_process = base
        else:
            events["base_shutdown_requested"] = stamp()
            publish(work / "base-action.json", dict(action="close"))
            base.wait(timeout=max(0.01, deadline - time.monotonic()))
            events["base_exited"] = stamp()
            if base.returncode:
                raise RuntimeError("base engine failed during close")
            target_process = launch("target")
        target = wait_file(work / "target-result.json", target_process, deadline)
        result["target"] = target
        target_process.wait(timeout=max(0.01, deadline - time.monotonic()))
        events["target_worker_exited"] = stamp()
        if target_process.returncode:
            raise RuntimeError("target worker failed; worker evidence retained")
        if not target["tokens_equal_reference"]:
            raise RuntimeError("target token parity failed")
        result["worker_identity"] = validate_worker_identity(ready, target, cfg["arm"])
        first_ns = target["first_token_perf_ns"]
        partition = first_token_partition(events["trigger"]["perf_ns"],
                                         events["target_files_ready"]["perf_ns"], target)
        pod_seconds = target["first_token_wall_ns"] / 1e9 - created_s
        if pod_seconds < 0:
            raise ValueError("first target token precedes actual Pod creation; clock evidence invalid")
        result["phase_s"] = dict(trigger_to_first_partition=partition,
            common_setup_to_barrier_s=(events["base_drained_barrier"]["perf_ns"] - events["setup_started"]["perf_ns"]) / 1e9,
            base_copy_s=(events["base_files_copied"]["perf_ns"] - events["setup_started"]["perf_ns"]) / 1e9,
            cache_setup_s=(events["cache_setup_finished"]["perf_ns"] - events["base_files_copied"]["perf_ns"]) / 1e9,
            target_engine_init_s=0 if cfg["arm"] == "resident" else (target["constructed"]["perf_ns"] - target["construct_started"]["perf_ns"]) / 1e9,
            base_weights=ready["weights"],
            target_weights=dict(host_weights_s=0, upload_weights_s=0) if cfg["arm"] == "resident" else target["weights"],
            target_graph=target["graph"],
            target_full_generation_s=(target["generation_done_perf_ns"] - target["generation_start_perf_ns"]) / 1e9)
        result.update(status="completed", validity="valid", tokens_equal_reference=True,
            metrics=dict(pod_creation_to_first_token_s=pod_seconds,
                reconfiguration_to_first_token_s=(first_ns - events["trigger"]["perf_ns"]) / 1e9))
    finally:
        # A failed pull must not leave the base waiting indefinitely for an action.
        for process, process_record in processes:
            if process.poll() is None:
                if not (work / "base-action.json").exists():
                    publish(work / "base-action.json", dict(action="close"))
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            process_record.update(returncode=process.returncode, parent_observed_final_state=stamp())
        for log in logs:
            log.close()
        result["worker_evidence"] = {p.name: read_json(p) for p in sorted(work.glob("*-finished.json"))}
        result["worker_files"] = sorted(p.name for p in work.iterdir() if p.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("trial", "worker"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--role", choices=("base", "target"))
    args = parser.parse_args(argv)
    if args.command == "worker":
        if not args.role:
            parser.error("worker requires --role")
        cfg = read_json(args.config)
        return worker(cfg, cfg["work_dir"], args.role)
    if not args.out:
        parser.error("trial requires --out")
    with open(args.out, "x", encoding="utf-8") as destination:
        result = dict(schema="ccgrid-e6-trial-v1", experiment="E6", status="failed",
            validity="invalid", tokens_equal_reference=False, metrics={},
            entry_perf_ns=ENTRY_PERF_NS, launched=stamp(), script=fingerprint(__file__),
            python=sys.version, hostname=platform.node(), perf_counter=dict(
                implementation=time.get_clock_info("perf_counter").implementation,
                monotonic=time.get_clock_info("perf_counter").monotonic))
        try:
            cfg = read_json(args.config)
            result.update(config=cfg, config_file=fingerprint(args.config),
                **{key: cfg.get(key) for key in ("trial_id", "arm", "cache_state", "base_variant", "target_variant")})
            trial(cfg, result)
        except Exception as exc:
            result.update(status="failed", validity="invalid", metrics={},
                error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        result["finished"] = stamp()
        json.dump(result, destination, indent=2)
        destination.write("\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
