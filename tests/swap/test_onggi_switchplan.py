"""Consumer safety and legacy compatibility for onggi's versioned byte plans."""
import copy
import json

import pytest

from rngd_pd.swap import delta


@pytest.fixture
def plan():
    return {
        "schema": "onggi.switchplan.v1",
        "base_manifest": "sha256:" + "a" * 64,
        "manifest": "sha256:" + "b" * 64,
        "size": 8, "same": 3, "moved": 3, "new": 2,
        "files": [{
            "path": "weights/target.bin", "base": "weights/base.bin",
            "size": 8, "same": 3, "moved": 3, "new": 2,
            "ranges": [
                {"offset": 0, "length": 2, "kind": "same"},
                {"offset": 2, "length": 3, "kind": "elsewhere",
                 "source_file": "other/base.bin", "source_offset": 9000},
                {"offset": 5, "length": 2, "kind": "new"},
                {"offset": 7, "length": 1, "kind": "same"},
            ],
        }],
    }


def test_versioned_loader_uses_canonical_target_ranges_without_changed(plan):
    assert delta.load_onggi_ranges(plan) == {"weights/target.bin": [(2, 5)]}


@pytest.mark.parametrize("consumer", [delta.parse_onggi_switchplan, delta.load_onggi_ranges])
def test_same_bytes_require_a_paired_base_file(plan, consumer):
    del plan["files"][0]["base"]
    with pytest.raises(ValueError, match="paired base"):
        consumer(plan)


@pytest.mark.parametrize("kind", ["new", "elsewhere"])
def test_unpaired_file_may_be_entirely_new_or_elsewhere(plan, kind):
    fd = plan["files"][0]
    del fd["base"]
    span = {"offset": 0, "length": 8, "kind": kind}
    if kind == "elsewhere":
        span.update(source_file="other/base.bin", source_offset=9000)
    fd["ranges"] = [span]
    for obj in [plan, fd]:
        obj.update(same=0, moved=8 if kind == "elsewhere" else 0,
                   new=8 if kind == "new" else 0)
    parsed = delta.parse_onggi_switchplan(plan)
    assert "base" not in parsed["files"][0]
    assert parsed["files"][0]["ranges"] == [span]
    assert delta.load_onggi_ranges(plan) == {"weights/target.bin": [(0, 8)]}


def test_parser_preserves_manifest_and_source_metadata_in_independent_copy(plan):
    plan["files"][0]["path"] = "./weights\\target.bin"
    plan["files"][0]["base"] = "weights//base.bin"
    plan["files"][0]["ranges"][1]["source_file"] = "./other\\base.bin"
    plan["future"] = {"nested": [1]}
    before = copy.deepcopy(plan)
    parsed = delta.parse_onggi_switchplan(plan)
    assert parsed["base_manifest"] == "sha256:" + "a" * 64
    assert parsed["manifest"] == "sha256:" + "b" * 64
    assert parsed["files"][0]["path"] == "weights/target.bin"
    assert parsed["files"][0]["base"] == "weights/base.bin"
    assert parsed["files"][0]["ranges"][1] == {
        "offset": 2, "length": 3, "kind": "elsewhere",
        "source_file": "other/base.bin", "source_offset": 9000,
    }
    parsed["files"][0]["ranges"][1]["source_offset"] = 0
    parsed["future"]["nested"].append(2)
    assert plan == before


@pytest.mark.parametrize("kind", ["string", "path"])
def test_versioned_loader_keeps_json_and_file_inputs(plan, tmp_path, kind):
    doc = json.dumps(plan)
    if kind == "path":
        doc = tmp_path / "plan.json"
        doc.write_text(json.dumps(plan), encoding="utf-8")
    assert delta.load_onggi_ranges(doc) == {"weights/target.bin": [(2, 5)]}


@pytest.mark.parametrize("schema", ["onggi.switchplan.v2", "", None, 1])
def test_loader_rejects_unknown_schema_instead_of_treating_it_as_legacy(plan, schema):
    plan["schema"] = schema
    plan["files"][0]["changed"] = [[2, 5]]
    with pytest.raises(ValueError):
        delta.load_onggi_ranges(plan)


@pytest.mark.parametrize("key", ["schema", "base_manifest", "manifest", "files",
                                "size", "same", "moved", "new"])
def test_parser_requires_v1_envelope_fields(plan, key):
    del plan[key]
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("key", ["base_manifest", "manifest"])
@pytest.mark.parametrize("value", [None, 12, "a" * 64, "sha256:" + "A" * 64,
                                   "sha256:" + "a" * 63, "sha256:" + "a" * 64 + "\n"])
def test_parser_rejects_malformed_manifest_digests(plan, key, value):
    plan[key] = value
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("scope,keys", [
    ("root", ("size", "same", "moved", "new")),
    ("file", ("size", "same", "moved", "new")),
    ("range", ("offset", "length", "source_offset")),
])
@pytest.mark.parametrize("value", [True, -1, 1.5, "8", None])
def test_parser_rejects_non_integer_or_negative_byte_counts(plan, scope, keys, value):
    for key in keys:
        doc = copy.deepcopy(plan)
        obj = {"root": doc, "file": doc["files"][0],
               "range": doc["files"][0]["ranges"][1]}[scope]
        obj[key] = value
        with pytest.raises(ValueError):
            delta.parse_onggi_switchplan(doc)


@pytest.mark.parametrize("key", ["path", "size", "same", "moved", "new"])
def test_parser_requires_file_fields(plan, key):
    del plan["files"][0][key]
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("scope", ["root", "file"])
@pytest.mark.parametrize("key", ["same_blocks", "changed_runs", "largest_run"])
def test_known_optional_metrics_must_be_nonnegative_integers(plan, scope, key):
    (plan if scope == "root" else plan["files"][0])[key] = True
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("field", ["path", "base", "source_file"])
@pytest.mark.parametrize("path", ["", ".", "../a", "a/../b", "/a", "\\a",
                                  "C:\\a", "a\x00b", None, 42])
def test_parser_rejects_unsafe_target_and_source_paths(plan, field, path):
    obj = plan["files"][0]
    if field == "source_file":
        obj = obj["ranges"][1]
    obj[field] = path
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("alias", ["weights/target.bin", "./weights/target.bin",
                                  "weights\\target.bin", "weights//target.bin"])
def test_parser_rejects_duplicate_normalized_target_paths(plan, alias):
    duplicate = copy.deepcopy(plan["files"][0])
    duplicate["path"] = alias
    plan["files"].append(duplicate)
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("key", ["offset", "length", "kind", "source_file", "source_offset"])
def test_elsewhere_requires_full_range_and_source_fields(plan, key):
    del plan["files"][0]["ranges"][1][key]
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("range_index", [0, 2])
@pytest.mark.parametrize("field,value", [("source_file", "base.bin"), ("source_offset", 0)])
def test_same_and_new_forbid_source_fields(plan, range_index, field, value):
    plan["files"][0]["ranges"][range_index][field] = value
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("change", ["missing", "empty", "gap", "overlap", "unsorted",
                                   "truncated", "overflow", "zero", "unknown_kind"])
def test_ranges_must_partition_entire_target_in_order(plan, change):
    fd = plan["files"][0]
    if change == "missing":
        del fd["ranges"]
    elif change == "empty":
        fd["ranges"] = []
    elif change == "gap":
        fd["ranges"][1]["offset"] = 3
    elif change == "overlap":
        fd["ranges"][1]["offset"] = 1
    elif change == "unsorted":
        fd["ranges"].reverse()
    elif change == "truncated":
        fd["ranges"].pop()
    elif change == "overflow":
        fd["ranges"][-1]["length"] = 2
    elif change == "zero":
        fd["ranges"][1]["length"] = 0
    else:
        fd["ranges"][1]["kind"] = "moved"
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("scope", ["root", "file"])
@pytest.mark.parametrize("key", ["size", "same", "moved", "new"])
def test_classification_lengths_and_aggregate_totals_are_exact(plan, scope, key):
    (plan if scope == "root" else plan["files"][0])[key] += 1
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


def test_equal_total_does_not_hide_incorrect_classification(plan):
    for obj in [plan, plan["files"][0]]:
        obj["same"] = 4
        obj["new"] = 1
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


@pytest.mark.parametrize("changed", [[[2, 5]], [[5, 2], [2, 4]], [[2, 3], [5, 2]]])
def test_optional_changed_may_have_equivalent_normalized_windows(plan, changed):
    plan["files"][0]["changed"] = changed
    assert delta.load_onggi_ranges(plan) == {"weights/target.bin": [(2, 5)]}


@pytest.mark.parametrize("changed", [[], [[0, 8]], [[2, 4]], [[2, 7]], [[True, 5]],
                                    [[2, -1]], [[2]], None, {}])
def test_optional_changed_cannot_disagree_with_canonical_ranges(plan, changed):
    plan["files"][0]["changed"] = changed
    with pytest.raises(ValueError):
        delta.load_onggi_ranges(plan)


@pytest.mark.parametrize("ranges", [None, []])
def test_zero_size_file_may_omit_ranges(plan, ranges):
    empty = {"path": "empty", "size": 0, "same": 0, "moved": 0, "new": 0}
    if ranges is not None:
        empty["ranges"] = ranges
    plan["files"].append(empty)
    assert delta.load_onggi_ranges(plan) == {"weights/target.bin": [(2, 5)], "empty": []}


@pytest.mark.parametrize("field,value", [("files", None), ("files", [None]),
                                        ("ranges", None), ("ranges", [None]),
                                        ("ranges", {})])
def test_parser_rejects_malformed_container_shapes(plan, field, value):
    (plan if field == "files" else plan["files"][0])[field] = value
    with pytest.raises(ValueError):
        delta.parse_onggi_switchplan(plan)


def test_unversioned_legacy_retains_unbounded_changed_and_unpaired_behavior():
    assert delta.load_onggi_ranges({"files": [
        {"path": "paired", "base": "paired", "size": 1, "changed": [[100, 3]]},
        {"path": "unpaired", "size": 5},
    ]}) == {"paired": [(100, 3)], "unpaired": [(0, 5)]}
