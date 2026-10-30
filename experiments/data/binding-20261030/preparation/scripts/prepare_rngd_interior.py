"""Off-card fixture preparation only; never imports Torch or opens an NPU.

Creates new paths exclusively. Existing base/E7 artifacts and original plans
remain untouched; emitted switch ranges are the frozen onggi binary's output.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import traceback


OLD_ROOT = Path("/home/gspark/workspace/codex_ccgrid_npu_20261030")
BINARY_SHA = "47bfaae6bace003bc29c9223521a80b44526c25835dc36d350b2979585abc114"
TENSOR = "model.layers.0.mlp.down_proj.weight"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_info(path, name):
    with Path(path).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        require(header_size <= min(100_000_000, path.stat().st_size - 8), "invalid public tensor header")
        info = json.loads(stream.read(header_size))[name]
    require(info["dtype"] == "BF16" and len(info["shape"]) == 2, "fixture requires 2D BF16 down_proj")
    start, end = info["data_offsets"]
    rows, columns = info["shape"]
    require(end - start == rows * columns * 2 and rows >= 192, "unsupported projection shape")
    return dict(start=8 + header_size + start, end=8 + header_size + end,
                shape=info["shape"], row_bytes=columns * 2)


def main():
    root = Path(os.environ.get("CCGRID_ROOT", "/home/gspark/workspace/codex_binding_conformance_20261030"))
    base = Path(os.environ.get("BINDING_BASE", "/home/gspark/workspace/codex_swap_benchmark_20261023/fixtures/resident-base"))
    original = Path(os.environ.get("BINDING_E7_SPARSE", "/home/gspark/workspace/codex_rngd_opt_20261023/fixtures/sparse-new"))
    binary = Path(os.environ.get("BINDING_ONGGI", str(OLD_ROOT / "tools/onggi-v1")))
    artifact, store = root / "rngd-artifacts/interior", root / "rngd-interior-store"
    output = root / "evidence/rngd-interior-preparation.json"
    with output.open("x") as stream:
        stream.write("{}\n")
    record = dict(status="running", utc=time.time(), scope="CPU-only new fixture/store; no existing artifact mutations",
                  script_sha256=sha(__file__), base=str(base), payload_source=str(original),
                  artifact=str(artifact), store=str(store), commands=[])
    def save():
        output.write_text(json.dumps(record, indent=2) + "\n")
    hashes = {}
    def inventory(directory):
        rows = []
        for path in sorted(directory.iterdir()):
            require(path.is_file() and not path.is_symlink(), "fixture must contain regular files only")
            st = path.stat()
            key = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns
            if key not in hashes:
                hashes[key] = sha(path)
            rows.append(dict(path=str(path), name=path.name, bytes=st.st_size, sha256=hashes[key],
                             device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns))
        return rows
    def command(argv, timeout):
        started = time.monotonic()
        process = subprocess.run(argv, capture_output=True, timeout=timeout,
                                 env=dict(os.environ, GOMAXPROCS="2"))
        row = dict(argv=argv, returncode=process.returncode, elapsed_s=time.monotonic() - started,
                   stdout=process.stdout.decode(), stderr=process.stderr.decode())
        record["commands"].append(row)
        save()
        require(process.returncode == 0, "onggi command failed; preserve command evidence")
        return process.stdout, json.loads(process.stdout)
    save()
    try:
        require(sha(binary) == BINARY_SHA, "frozen onggi-v1 binary differs")
        require(not artifact.exists() and not store.exists(), "refusing to reuse an existing fixture/store")
        record["binary"] = dict(path=str(binary), sha256=BINARY_SHA)
        index = read_json(base / "model.safetensors.index.json")["weight_map"]
        original_index = read_json(original / "model.safetensors.index.json")["weight_map"]
        shard = index[TENSOR]
        require(Path(shard).name == shard and original_index[TENSOR] == shard, "unexpected projection shard path")
        info, source_info = tensor_info(base / shard, TENSOR), tensor_info(original / shard, TENSOR)
        require(info == source_info, "original projection headers differ")
        record["base_files"] = inventory(base)
        record["payload_source_files"] = inventory(original)
        require(len(record["base_files"]) == 6, "expected frozen six-file base artifact")
        artifact.mkdir(parents=True, exist_ok=False)
        for entry in record["base_files"]:
            source, destination = Path(entry["path"]), artifact / entry["name"]
            if source.name == shard:
                started = time.monotonic()
                with source.open("rb") as stream, destination.open("xb") as target:
                    shutil.copyfileobj(stream, target, length=8 << 20)
                require((source.stat().st_dev, source.stat().st_ino) !=
                        (destination.stat().st_dev, destination.stat().st_ino), "patched shard must have its own inode")
                record["shard_copy"] = dict(source=str(source), target=str(destination), bytes=source.stat().st_size,
                                             elapsed_s=time.monotonic() - started)
            else:
                os.link(source, destination)
        length = 64 * info["row_bytes"]
        source_offset, target_offset = source_info["start"], info["start"] + 128 * info["row_bytes"]
        with (original / shard).open("rb") as stream:
            stream.seek(source_offset)
            payload = stream.read(length)
        require(len(payload) == length, "source projection payload truncated")
        with (artifact / shard).open("r+b") as stream:
            stream.seek(target_offset)
            previous = stream.read(length)
            require(previous != payload, "interior replacement unexpectedly equals base rows")
            stream.seek(target_offset)
            require(stream.write(payload) == length, "short interior write")
            stream.flush()
            os.fsync(stream.fileno())
        record["patch"] = dict(tensor=TENSOR, file=shard, source_rows=[0, 64], target_rows=[128, 192],
                               source_offset=source_offset, target_offset=target_offset, bytes=length,
                               payload_sha256=hashlib.sha256(payload).hexdigest(),
                               replaced_base_rows_sha256=hashlib.sha256(previous).hexdigest(), tensor_info=info)
        record["interior_files"] = inventory(artifact)
        require(tensor_info(artifact / shard, TENSOR) == info, "fixture metadata changed")
        store.mkdir(exist_ok=False)
        pushed = {}
        for role, directory in (("base", base), ("interior", artifact)):
            raw, pushed[role] = command([str(binary), "push", "--store", str(store), "--sub", "0",
                "--compress", "none", "--workers", "2", "--uploads", "2", "--quiet", "--json", "--ref", role,
                str(directory)], 1200)
        raw, plan = command([str(binary), "diff", "--store", str(store), "--ranges", "--json", "base", "interior"], 120)
        require(plan["schema"] == "onggi.switchplan.v1", "actual diff did not return v1")
        require(plan["base_manifest"] == pushed["base"]["manifest"]
                and plan["manifest"] == pushed["interior"]["manifest"], "push/diff manifest disagreement")
        changed = []
        for file in plan["files"]:
            for span in file["ranges"]:
                if span["kind"] != "same":
                    require(file["path"] == shard and info["start"] <= span["offset"]
                            and span["offset"] + span["length"] <= info["end"],
                            "actual coarse changed range escaped supported down_proj bytes")
                    changed.append(span)
        require(changed, "new projection fixture has no actual changed ranges")
        plan_path = root / "inputs/interior.switchplan.json"
        with plan_path.open("xb") as stream:
            stream.write(raw)
        identical_path, lm_path = root / "inputs/identical.switchplan.json", root / "inputs/lm.switchplan.json"
        identical, lm = read_json(identical_path), read_json(lm_path)
        require(identical["base_manifest"] == identical["manifest"] == lm["base_manifest"],
                "reused authentic identical/LM plans disagree about their own base")
        cases = [dict(name="native-identical", reference_role="base", base_dir=str(base), target_dir=str(base),
                      switchplan=str(identical_path), expect_mode="units"),
                 dict(name="native-interior-projection", reference_role="sparse", base_dir=str(base), target_dir=str(artifact),
                      switchplan=str(plan_path), expect_mode="units"),
                 dict(name="native-lm", reference_role="lm", base_dir=str(base),
                      target_dir="/home/gspark/workspace/codex_rngd_opt_20261023/fixtures/lm-new",
                      switchplan=str(lm_path), expect_mode="units")]
        provider = root / "inputs/rngd-interior-fixtures.json"
        with provider.open("x") as stream:
            json.dump(dict(schema="onggi.binding-fixtures.v1", cases=cases), stream, indent=2)
            stream.write("\n")
        record.update(status="completed", finished_utc=time.time(), changed_ranges=changed,
                      provider=str(provider), provider_sha256=sha(provider),
                      plan=str(plan_path), plan_sha256=sha(plan_path),
                      reused_plans={str(path): sha(path) for path in (identical_path, lm_path)},
                      base_identity_evidence={"old_store_manifest": identical["base_manifest"],
                                              "new_store_manifest": plan["base_manifest"],
                                              "same_base_bytes": "all cases use the exact same base directory/files inventoried above",
                                              "new_store_parameters": {"sub": 0, "compress": "none", "workers": 2, "uploads": 2}},
                      base_manifest=plan["base_manifest"], manifest=plan["manifest"])
        save()
    except BaseException:
        record.update(status="failed", traceback=traceback.format_exc(), finished_utc=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
