#!/usr/bin/env python3
"""Controlled resident delivery trial. See experiments/RESIDENT-DELIVERY-PROTOCOL.md.

One CLI invocation owns one native engine and exits after cleanup. CPU validation:
  python bench/ccgrid_resident_delivery.py --check-config --config FILE
Allocated Linux trial:
  python bench/ccgrid_resident_delivery.py trial --config FILE --out NEWFILE
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

import ccgrid_e6 as e6


ARMS = ("graph-copy", "graph-csi", "full-csi-no-rebase", "full-csi-rebase", "graph-csi-active", "full-csi-active")
ENTRY = e6.stamp()


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def validate(cfg):
    """No SDK import, device operation, hashing or mutation at this boundary."""
    if cfg.get("arm") not in ARMS:
        raise ValueError("unknown arm")
    if cfg.get("phase") not in ("smoke", "measure"):
        raise ValueError("phase must be smoke or measure")
    for key in ("trial_id", "store", "onggi_binary", "csi_client"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip() or cfg[key].startswith("-"):
            raise ValueError("missing or unsafe config field: " + key)
    for key in ("base_ref", "target_ref"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", cfg.get(key, "")):
            raise ValueError(key + " must be an immutable sha256 ref")
    if cfg["base_ref"] == cfg["target_ref"]:
        raise ValueError("base and target refs must be different")
    if type(cfg.get("chip")) is not int or cfg["chip"] < 0:
        raise ValueError("chip must be an allocated nonnegative ordinal")
    if type(cfg.get("host_workers")) is not int or cfg["host_workers"] != 2:
        raise ValueError("host_workers must be 2")
    if type(cfg.get("intra_op")) is not int or cfg["intra_op"] != 4:
        raise ValueError("intra_op must be 4")
    if not isinstance(cfg.get("provenance"), dict):
        raise ValueError("provenance must be a dictionary")
    work = Path(cfg["work_dir"])
    if not work.is_absolute():
        raise ValueError("work_dir must be absolute")
    work = work.resolve()
    if work.exists():
        raise FileExistsError("work_dir already exists; refusing to consume it")
    for key in ("immutable_checkpoint", "base_cache_dir", "base_bundle_source", "target_bundle_source"):
        source = Path(cfg[key])
        if not source.is_absolute() or not source.is_dir() or overlap(source.resolve(), work):
            raise ValueError("work_dir must be independent of existing absolute input: " + key)
    reference_path = Path(cfg["reference_file"])
    if not reference_path.is_absolute() or not reference_path.is_file() or overlap(reference_path.resolve(), work):
        raise ValueError("reference_file must be an independent existing absolute file")
    from rngd_pd.swap.variants import Variant
    base, target = (Variant.parse(cfg[key]) for key in ("base_variant", "target_variant"))
    if base == target or base.group != target.group or base.group <= 0:
        raise ValueError("different variants must share one positive group")
    refs = e6.read_json(reference_path)
    for prefix, variant in (("base_", base), ("", target)):
        for suffix in ("prompts", "tokens"):
            key = prefix + suffix
            rows = refs.get(key)
            if (not isinstance(rows, list) or len(rows) != variant.B or not rows
                    or not all(isinstance(row, list) and row and all(type(t) is int and t >= 0 for t in row)
                               for row in rows) or len({len(row) for row in rows}) != 1):
                raise ValueError("invalid reference shape/integer parity input: " + key)
        if len(refs[prefix + "prompts"][0]) + len(refs[prefix + "tokens"][0]) > variant.S:
            raise ValueError("reference exceeds variant context")
    for key in ("base_bundle_source", "target_bundle_source"):
        root = Path(cfg[key])
        if not (root / "cache/graphs").is_dir() or not (root / "variant.json").is_file():
            raise ValueError("bundle needs cache/graphs and variant.json: " + key)
        paths = list(root.rglob("*"))
        if root.is_symlink() or any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):
            raise ValueError("bundle inputs must contain ordinary files/directories")
        if cfg["arm"].startswith("graph-"):
            if any(p.relative_to(root).as_posix() not in ("cache", "cache/graphs", "variant.json")
                   and not p.relative_to(root).as_posix().startswith("cache/graphs/") for p in paths):
                raise ValueError("graph-only bundle may contain only cache/graphs and variant.json")
        elif not (root / "checkpoint").is_dir():
            raise ValueError("whole bundle requires checkpoint/")
    prepared = cfg.get("prepared_materializer")
    if prepared:
        data = Path(prepared)
        if (cfg["arm"] in ("graph-copy", "graph-csi-active") or not data.is_absolute()
                or not data.is_dir() or overlap(work, data.resolve())
                or any(overlap(data.resolve(), Path(cfg[k]).resolve()) for k in
                       ("immutable_checkpoint", "base_cache_dir", "base_bundle_source", "target_bundle_source"))):
            raise ValueError("prepared_materializer must be independent owned idle state")
        receipt = Path(cfg["preparation_receipt"])
        if not receipt.is_absolute() or not receipt.is_file() or overlap(work, receipt.resolve()):
            raise ValueError("preparation_receipt must be an independent existing absolute file")
    return refs


def file_metadata(root):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise RuntimeError("symlink in delivered artifact")
    return {p.relative_to(root).as_posix(): dict(size=(s := p.stat()).st_size,
        inode=s.st_ino, device=s.st_dev, blocks=getattr(s, "st_blocks", None),
        realpath=str(p.resolve()), nlink=s.st_nlink, mode=s.st_mode)
        for p in sorted(root.rglob("*")) if p.is_file()}


def inventory(root):
    root = Path(root)
    result = file_metadata(root)
    for name, row in result.items():
        row["sha256"] = e6.fingerprint(root / name)["sha256"]
    return result


def verify_inventory(root, expected):
    actual = file_metadata(root)
    if set(actual) != set(expected):
        raise RuntimeError("delivered inventory differs: missing=%s extra=%s" %
            (sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))))
    for name, row in actual.items():
        row["sha256"] = e6.fingerprint(Path(root) / name)["sha256"]
        if row["size"] != expected[name]["size"] or row["sha256"] != expected[name]["sha256"]:
            raise RuntimeError("delivered file hash/size mismatch: " + name)
    return actual


def timing(events, target):
    points = [events[k]["perf_ns"] for k in ("started", "demand", "delivery_finished", "hash_finished")]
    points += [target["generation_start_perf_ns"], target["first_token_perf_ns"]]
    if points != sorted(points):
        raise ValueError("trial timestamps are not ordered")
    return dict(delivery_s=(points[2] - points[1]) / 1e9,
        hash_validation_s=(points[3] - points[2]) / 1e9,
        hash_to_generation_s=(points[4] - points[3]) / 1e9,
        prompt_to_first_token_s=(points[5] - points[4]) / 1e9,
        demand_to_first_token_s=(points[5] - points[1]) / 1e9,
        harness_entry_to_first_token_s=(points[5] - points[0]) / 1e9,
        excluded_pre_demand_setup_s=(points[1] - points[0]) / 1e9)


def exact_mount(target, raw=None):
    raw = Path("/proc/self/mountinfo").read_text() if raw is None else raw
    for line in reversed(raw.splitlines()):
        left, right = line.split(" - ", 1)
        fields = left.split()
        mountpoint = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if mountpoint == str(target):
            return dict(mount_id=fields[0], mountinfo=line, options=fields[5].split(","))
    return None


def readonly_mount(target, raw=None):
    mounted = exact_mount(target, raw)
    if mounted:
        if "ro" not in mounted["options"]:
            raise RuntimeError("target mount is not read-only")
        return mounted
    raise RuntimeError("no exact target mount in current namespace")


def verify_cleanup(data, mounts, raw=None):
    for target in mounts:
        if exact_mount(target, raw):
            raise RuntimeError("owned mount remains after unpublish/unmount: " + str(target))
    for name in ("leases", "volumes"):
        if any(path.is_file() for path in (data / name).rglob("*")):
            raise RuntimeError("owned materializer lease/volume state remains after cleanup")


def allocated_device(cfg):
    from rngd_pd import devices
    if platform.system() != "Linux":
        raise RuntimeError("native trial requires Linux and a real allocated card")
    cards = devices.allocated_cards()
    if cards is None or len(cards) != 1:
        raise RuntimeError("exactly one allocated RNGD card is required")
    chip, physical = devices.resolve_chip(-1), cards[0]
    if chip != cfg["chip"]:
        raise RuntimeError("configured chip ordinal differs from actual allocation")
    if Path("/run/swap/rngd").read_text().strip() != "npu" + str(physical):
        raise RuntimeError("init/main card assignments differ")
    uid = os.environ["POD_UID"]
    uuids = devices.allocated_uuids(uid, os.environ.get("CONTAINER_NAME", "run"))
    if len(uuids) != 1:
        raise RuntimeError("ambiguous allocation UUID")
    return dict(pod_uid=uid, host_index=physical, torch_ordinal=chip,
        visible_cards=devices.visible_cards(), allocation_uuid=uuids)


def delivery_log_proof(log, target_ref, base_ref):
    def same(value, ref):
        return value in (ref, ref.split(":")[1])
    row = dict(pull_stats=None, claimed_base=False, left_in_place_bytes=0,
        busy_donor_refusal=False, idle_donor_refusal=False, target_lines=[])
    for line in log.splitlines():
        materialized = re.search(r"materialized (\S+) in .*?: (\{.*\})", line)
        target = re.search(r"pull of (\S+?)(?:: | took over )", line)
        if materialized and same(materialized[1], target_ref):
            row["pull_stats"] = json.loads(materialized[2])
            row["target_lines"].append(line)
        if target and same(target[1], target_ref):
            row["target_lines"].append(line)
            claim = re.search(r"took over (\S+): (\d+) B left in place", line)
            if claim and same(claim[1], base_ref):
                row.update(claimed_base=True, left_in_place_bytes=int(claim[2]))
            row["busy_donor_refusal"] |= "no artifact to take over" in line and bool(re.search(r"\([1-9]\d* in use,", line))
            row["idle_donor_refusal"] |= "took over nothing:" in line
    return row


def snapshot(data, roots=()):
    row = dict(at=e6.stamp(), pid=os.getpid(), files={}, disk_totals={})
    for name in ("status", "smaps_rollup"):
        path = Path("/proc/self") / name
        if path.exists():
            row[name] = path.read_text()
    row["mountinfo"] = Path("/proc/self/mountinfo").read_text() if Path("/proc/self/mountinfo").exists() else None
    row["cgroup"] = {}
    path = Path("/proc/self/cgroup")
    if path.exists():
        row["cgroup_membership"] = path.read_text()
        for line in row["cgroup_membership"].splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0":
                for name in ("memory.current", "memory.peak", "memory.stat", "memory.events", "memory.max"):
                    for root in (Path("/sys/fs/cgroup") / relative.lstrip("/"), Path("/sys/fs/cgroup")):
                        candidate = root / name
                        if candidate.is_file():
                            row["cgroup"][str(candidate)] = candidate.read_text()
                            break
    for root in roots:
        root = Path(root)
        files = file_metadata(root) if root.exists() else None
        row["files"][str(root)] = files
        if files is not None:
            unique = {(item["device"], item["inode"]): item for item in files.values()}
            row["disk_totals"][str(root)] = dict(logical_file_entry_bytes=sum(item["size"] for item in files.values()),
                unique_inode_allocated_bytes=sum(item["blocks"] * 512 for item in unique.values())
                    if all(item["blocks"] is not None for item in unique.values()) else None,
                scope="files only; directory/metadata allocation excluded; roots may share inodes")
    row["materializer_state"] = {}
    if data.exists():
        for sub in ("volumes", "leases"):
            directory = data / sub
            if directory.exists():
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        row["materializer_state"][path.relative_to(data).as_posix()] = path.read_text()
        row["materializer_used"] = {p.name: dict(mtime_ns=p.stat().st_mtime_ns,
            text=p.read_text()) for p in sorted((data / "artifacts").glob("*.used"))}
    return row


def command(argv, row, timeout=310):
    row.update(argv=list(map(str, argv)), started=e6.stamp())
    try:
        process = subprocess.run(row["argv"], capture_output=True, text=True, timeout=timeout, check=False)
        row.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
        if process.returncode:
            raise RuntimeError("command failed; exact argv/stdout/stderr retained")
    except subprocess.TimeoutExpired as exc:
        row.update(timeout=True, stdout=str(exc.stdout), stderr=str(exc.stderr))
        raise
    finally:
        row["finished"] = e6.stamp()
    return row


class CSI:
    def __init__(self, cfg, work, data, result):
        self.cfg, self.work, self.data, self.result = cfg, work, data, result
        self.socket = work / "csi.sock"
        self.process = self.log = self.row = None

    def start(self, label):
        if self.socket.exists():
            self.socket.unlink()
        cache = self.work / (label + "-cache")
        cache.mkdir()
        idle = "0" if self.cfg["arm"] == "full-csi-no-rebase" else "1s"
        argv = [self.cfg["onggi_binary"], "csi", "--store", self.cfg["store"],
            "--node-id", "cdx-exp", "--endpoint", "unix://" + str(self.socket),
            "--data-dir", str(self.data), "--cache", str(cache), "--workers", "8", "--pulls", "1",
            "--gc-every", "0", "--evict-after", "0", "--rebase-idle", idle]
        self.row = dict(argv=argv, started=e6.stamp(), log=str(self.work / (label + "-csi.log")), cache_initial_entries=[])
        self.result.setdefault("servers", []).append(self.row)
        self.log = open(self.row["log"], "x", encoding="utf-8")
        self.process = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT)
        self.row["pid"] = self.process.pid
        deadline = time.monotonic() + 30
        while not self.socket.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("CSI server did not create socket; retained child log")
            time.sleep(0.02)
        self.row["socket_ready"] = e6.stamp()

    def rpc(self, operation, target, volume, ref, label):
        row = self.result.setdefault("rpcs", {}).setdefault(label, {})
        command([self.cfg["csi_client"], "--socket", str(self.socket), "--operation", operation,
            "--volume-id", volume, "--target", str(target), "--ref", ref, "--timeout", "300s"], row)
        row["report"] = json.loads(row["stdout"])
        return row

    def stop(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.row["forced_kill"] = True
                    self.process.kill()
                    self.process.wait(timeout=10)
            self.row.update(returncode=self.process.returncode, stopped=e6.stamp())
            if self.row.get("forced_kill") or self.process.returncode not in (0, -signal.SIGTERM):
                raise RuntimeError("CSI server failed clean shutdown")
        finally:
            self.log.close()
            self.row["log_sha256"] = e6.fingerprint(self.row["log"])["sha256"]
            self.process = None


def live_paths(cfg, base_mount):
    checkpoint = str(base_mount / "checkpoint") if cfg["arm"] == "full-csi-active" else cfg["immutable_checkpoint"]
    cache = str(base_mount / "cache/graphs") if cfg["arm"].endswith("csi-active") else cfg["base_cache_dir"]
    return checkpoint, cache


def parity_base(engine, reference, counts):
    before = counts["native_calls"]
    tokens, ttft = engine.generate(reference["base_prompts"], len(reference["base_tokens"][0]))
    engine._sync()
    expected = (len(reference["base_prompts"][0]) + len(reference["base_tokens"][0]) - 1) * (engine.ngroups + 1)
    row = dict(tokens=tokens, tokens_equal_reference=tokens == reference["base_tokens"],
        native_calls=counts["native_calls"] - before, expected_native_calls=expected, ttft_s=ttft, finished=e6.stamp())
    if not row["tokens_equal_reference"] or row["native_calls"] != expected:
        raise RuntimeError("base token/native call parity failed: " + json.dumps(row))
    return row


def trial(cfg, result):
    reference = validate(cfg)
    work = Path(cfg["work_dir"]).resolve()
    work.mkdir()
    e6.publish(work / "config.json", cfg)
    data = Path(cfg.get("prepared_materializer", str(work / "materializer")))
    events = result.setdefault("events", {})
    events["started"] = ENTRY
    result.update(reference=reference, reference_file=e6.fingerprint(cfg["reference_file"]),
        os_page_cache="uncontrolled/shared", allow_compile=False, allocator_retries=0,
        counter_scope="file metadata describes logical bytes and filesystem blocks; no DMA/write counter claim")
    expected = {name: inventory(cfg[name + "_bundle_source"]) for name in ("base", "target")}
    result["expected_inventory"] = expected
    events["expected_hashes_finished"] = e6.stamp()
    base_mount, target_mount = work / "base-mount", work / "target-mount"
    donor = data / "artifacts" / cfg["base_ref"].split(":")[1]
    csi = CSI(cfg, work, data, result)
    engine = None
    cleanup_errors = []
    target_attempted = base_attempted = bound = False
    result["snapshots"] = {}
    try:
        result["assignment"] = allocated_device(cfg)
        if cfg["arm"] != "graph-copy":
            if cfg.get("prepared_materializer"):
                receipt = e6.read_json(cfg["preparation_receipt"])
                result["external_preparation"] = dict(receipt=receipt,
                    receipt_file=e6.fingerprint(cfg["preparation_receipt"]), excluded_from_allocation_timer=True)
                if (receipt.get("base_ref") != cfg["base_ref"]
                        or Path(receipt.get("materializer", "")).resolve() != data.resolve()
                        or receipt.get("status") != "completed"):
                    raise RuntimeError("preparation receipt identity/status mismatch")
                initial = snapshot(data, [donor])
                result["snapshots"]["external_preparation_received"] = initial
                if initial["materializer_state"]:
                    raise RuntimeError("prepared materializer has active leases or volume state")
                if {p.name for p in (data / "artifacts").iterdir() if p.is_dir()} != {donor.name}:
                    raise RuntimeError("prepared materializer must contain only the configured base artifact")
                result["donor_prepared_inventory"] = verify_inventory(donor, expected["base"])
                witnesses = receipt.get("inventory")
                if not witnesses or any(witnesses.get(name) != row for name, row in result["donor_prepared_inventory"].items()):
                    raise RuntimeError("preparation receipt full hash/inode inventory mismatch")
            else:
                events["donor_preparation_started"] = e6.stamp()
                csi.start("prep")
                base_attempted = True
                csi.rpc("publish", base_mount, "base", cfg["base_ref"], "base-publish")
                result["base_mount_evidence"] = readonly_mount(base_mount)
                result["base_published_inventory"] = verify_inventory(base_mount, expected["base"])
                result["donor_prepared_inventory"] = verify_inventory(donor, expected["base"])
                if not cfg["arm"].endswith("csi-active"):
                    csi.rpc("unpublish", base_mount, "base", cfg["base_ref"], "base-idle")
                    base_attempted = False
                csi.stop()
                events["donor_preparation_finished"] = e6.stamp()
            csi.start("demand")
            if cfg["arm"] == "full-csi-active" and cfg.get("prepared_materializer"):
                base_attempted = True
                csi.rpc("publish", base_mount, "base", cfg["base_ref"], "base-publish")
                result["base_mount_evidence"] = readonly_mount(base_mount)
                result["base_published_inventory"] = verify_inventory(base_mount, expected["base"])
            events["real_idle_wait_started"] = e6.stamp()
            time.sleep(1.05)
            events["real_idle_wait_finished"] = e6.stamp()
        else:
            result["donor_setup"] = "skipped for local graph copy; lower setup cost recorded"
        import torch
        torch.set_num_threads(cfg["intra_op"])
        torch.set_grad_enabled(False)
        import rngd_pd.swap.rngd as runtime
        from importlib.metadata import version
        source = Path(__file__).resolve().parents[1]
        expected_runtime = source / "src/rngd_pd/swap/rngd.py"
        if Path(runtime.__file__).resolve() != expected_runtime.resolve():
            raise RuntimeError("import selected a different native runtime source")
        result["source_files"] = {str(path.relative_to(source)): e6.fingerprint(path)
            for path in [Path(__file__).resolve(), source / "bench/ccgrid_e6.py"]
            + sorted((source / "src/rngd_pd").rglob("*.py"))}
        result["packages"] = {name: version(name) for name in ("torch", "numpy", "furiosa-torch", "furiosa-tcc")}
        from rngd_pd.swap.rngd import SwappableDecode
        from rngd_pd.swap.variants import Variant
        counts, sdk = e6.observe_native(cfg["chip"])
        result.update(sdk=sdk, torch=str(torch.__version__), counts=counts)
        events["construct_started"] = e6.stamp()
        checkpoint, cache = live_paths(cfg, base_mount)
        result.update(live_checkpoint=checkpoint, live_base_graph_cache=cache)
        engine = SwappableDecode(checkpoint, group=Variant.parse(cfg["base_variant"]).group,
            chip=cfg["chip"], cache_dir=cache, allow_compile=False, host_workers=2)
        events["constructed"] = e6.stamp()
        result["base_graph"] = engine.swap(Variant.parse(cfg["base_variant"]))
        result["base_generation"] = parity_base(engine, reference, counts)
        if Variant.parse(cfg["target_variant"]) in engine.variants:
            raise RuntimeError("target prepared before demand")
        if counts["shared_uploads"] != 1 or counts["compilation_attempts"]:
            raise RuntimeError("base upload/compilation gate failed")
        result.update(weight_buffers_before=e6.weight_buffers(engine), graph_buffers_before=e6.graph_buffers(engine),
            base_counts=dict(counts), weights=dict(engine.timings))
        result["snapshots"]["base_ready"] = snapshot(data, [donor, base_mount])
        engine._sync()
        events["zero_inflight_drain_barrier"] = e6.stamp()
        events["demand"] = e6.stamp()
        if cfg["arm"] == "graph-copy":
            command_row = result.setdefault("local_copy_bind", {})
            command_row["copy_started"] = e6.stamp()
            copied = work / "target-copy"
            shutil.copytree(cfg["target_bundle_source"], copied)
            command_row["copy_finished"] = e6.stamp()
            target_mount.mkdir()
            bound = True
            command(["mount", "--bind", str(copied), str(target_mount)], command_row.setdefault("bind", {}))
            command(["mount", "-o", "remount,bind,ro", str(target_mount)], command_row.setdefault("readonly", {}))
        else:
            target_attempted = True
            csi.rpc("publish", target_mount, "target", cfg["target_ref"], "target-publish")
        events["delivery_finished"] = e6.stamp()
        result["target_mount_evidence"] = readonly_mount(target_mount)
        result["delivered_inventory"] = verify_inventory(target_mount, expected["target"])
        events["hash_finished"] = e6.stamp()
        if cfg["arm"] != "graph-copy":
            result["delivery_log_proof"] = delivery_log_proof(Path(csi.row["log"]).read_text(),
                cfg["target_ref"], cfg["base_ref"])
            stats = result["delivery_log_proof"]["pull_stats"]
            if not stats or stats.get("bytes") != sum(row["size"] for row in expected["target"].values()):
                raise RuntimeError("target materialized PullStats missing or logical target byte count differs")
        result["snapshots"]["delivery"] = snapshot(data, [donor, base_mount, target_mount])
        if cfg["arm"].endswith("csi-active"):
            preserved = verify_inventory(base_mount, expected["base"])
            result["active_base_preserved"] = preserved == result["base_published_inventory"]
            if not result["active_base_preserved"] or not donor.exists():
                raise RuntimeError("active base mount/donor changed")
            result["active_base_recheck"] = parity_base(engine, reference, counts)
            result["counts_after_active_base_recheck"] = dict(counts)
        engine.cache_dir = str(target_mount / "cache/graphs")
        target = result.setdefault("target", {})
        e6.generate_target(engine, cfg, reference, counts, target)
        events["target_generation_finished"] = e6.stamp()
        result["snapshots"]["target_ready"] = snapshot(data, [donor, base_mount, target_mount])
        result["resident_weights_preserved"] = result["weight_buffers_before"] == target["weight_buffers_after"]
        result["base_graphs_preserved"] = all(target["graph_buffers_after"].get(v) == ids
            for v, ids in result["graph_buffers_before"].items())
        if not result["resident_weights_preserved"] or not result["base_graphs_preserved"]:
            raise RuntimeError("resident weights/base graph identities changed")
        before = result.get("donor_prepared_inventory", {})
        after = result["delivered_inventory"]
        continuous = [name for name, row in before.items() if name in after and
            row["inode"] == after[name]["inode"] and row["device"] == after[name]["device"]]
        result["claim_witness"] = dict(donor_removed=not donor.exists(), inode_continuity_files=continuous,
            donor_path=str(donor), logs=[row["log"] for row in result.get("servers", [])])
        proof = result.get("delivery_log_proof", {})
        if cfg["arm"] == "full-csi-rebase" and (donor.exists() or not continuous
                or not proof.get("claimed_base") or not proof.get("left_in_place_bytes")
                or not proof.get("pull_stats")):
            raise RuntimeError("full rebase did not prove PullStats/claim/removal/inode continuity; stop main")
        if cfg["arm"] == "graph-csi-active" and (proof.get("claimed_base") or not proof.get("busy_donor_refusal")):
            raise RuntimeError("active-base CSI did not prove busy-donor refusal/fallback")
        if cfg["arm"] == "full-csi-no-rebase" and (proof.get("claimed_base") or not donor.exists()):
            raise RuntimeError("no-rebase control unexpectedly consumed its donor")
        result["metrics"] = timing(events, target)
        result["native_gates_passed"] = True
    finally:
        events["close_started"] = e6.stamp()
        owner_closed = engine is None
        if engine is not None:
            try:
                engine.close()
                owner_closed = True
            except Exception:
                cleanup_errors.append(dict(stage="engine.close", traceback=traceback.format_exc()))
        if owner_closed:
            events["engine_closed"] = e6.stamp()
        result["file_release_blocked"] = not owner_closed
        result["snapshots"]["close"] = snapshot(data, [donor, base_mount, target_mount])
        # Failed sync can leave the owner alive; keep its mounts and leases until process exit.
        for attempted, target, volume, ref in ((target_attempted, target_mount, "target", cfg["target_ref"]),
                (base_attempted, base_mount, "base", cfg["base_ref"])):
            if owner_closed and attempted:
                try:
                    if csi.process is None:
                        csi.start("cleanup")
                    csi.rpc("unpublish", target, volume, ref, volume + "-cleanup")
                except Exception:
                    cleanup_errors.append(dict(stage=volume + "-unpublish", traceback=traceback.format_exc()))
        if owner_closed and bound:
            try:
                command(["umount", str(target_mount)], result.setdefault("local_unmount", {}))
            except Exception:
                cleanup_errors.append(dict(stage="umount", traceback=traceback.format_exc()))
        try:
            csi.stop()
        except Exception:
            cleanup_errors.append(dict(stage="server-stop", traceback=traceback.format_exc()))
        if owner_closed:
            try:
                verify_cleanup(data, [base_mount, target_mount])
            except Exception:
                cleanup_errors.append(dict(stage="cleanup-state-gate", traceback=traceback.format_exc()))
        result["snapshots"]["cleanup"] = snapshot(data, [donor, base_mount, target_mount])
        events["cleanup_finished"] = e6.stamp()
        result["cleanup_errors"] = cleanup_errors
        result["cleanup_passed"] = not cleanup_errors
        e6.publish(work / "partial-result.json", result)
        if cleanup_errors:
            raise RuntimeError("cleanup failed; all partial evidence retained")
    result.update(status="completed", validity="valid")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("trial",))
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.check_config:
        try:
            validate(e6.read_json(args.config))
        except Exception as exc:
            print(json.dumps(dict(status="config_invalid", error=str(exc))), file=sys.stderr)
            return 1
        print(json.dumps(dict(status="config_valid")))
        return 0
    if args.command != "trial" or not args.out:
        parser.error("trial requires --out NEWFILE")
    with open(args.out, "x", encoding="utf-8") as stream:
        result = dict(schema="resident-delivery-v1", status="failed", validity="invalid", metrics={},
            started=ENTRY, script=e6.fingerprint(__file__), pid=os.getpid(), python=sys.version,
            hostname=platform.node(), perf_counter_implementation=time.get_clock_info("perf_counter").implementation)
        try:
            cfg = e6.read_json(args.config)
            result.update(config=cfg, config_file=e6.fingerprint(args.config))
            trial(cfg, result)
        except Exception as exc:
            result.update(status="failed", validity="invalid", metrics={}, error=dict(
                type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        result["finished"] = e6.stamp()
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
