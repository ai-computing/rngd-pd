"""Whole-request connector contract; vLLM is stubbed only while importing when absent."""
import types

import numpy as np
import pytest

from rngd_pd.connector import transport
from connector_support import connector_module


@pytest.fixture
def connector():
    with connector_module() as module:
        yield module


def test_scheduler_metadata_preserves_request_and_external_token_count(connector):
    conn = connector.RngdKVConnector(connector.KVConnectorRole.SCHEDULER, n_layers=2)
    conn.update_state_after_alloc(types.SimpleNamespace(request_id="actual"), None, 17)
    meta = conn.build_connector_meta(None)
    assert meta.req_id == "actual"
    assert meta.num_external_tokens == 17
    assert meta.load_layers == ["layer.0", "layer.1"]
    # A later step cannot replay a stale allocation.
    idle = conn.build_connector_meta(None)
    assert idle.load_layers == []
    assert idle.num_external_tokens == 0


def test_bound_metadata_drives_real_save_and_load_keys(connector):
    tr = transport.InProcTransport()
    worker = connector.RngdKVConnector(connector.KVConnectorRole.WORKER,
                                       req_id="constructor", transport=tr, n_layers=1)
    meta = connector.RngdKVConnectorMetadata(req_id="actual", save_layers=["layer.0"],
                                            load_layers=["layer.0"], num_external_tokens=3)
    worker.bind_connector_metadata(meta)
    k = np.array([[[1., 2., 3.]]], np.float32)
    v = -k
    worker.save_kv_layer("layer.0", (k, v))
    assert set(tr._store) == {"actual/layer.0/k", "actual/layer.0/v"}
    dk, dv = np.zeros_like(k), np.zeros_like(v)
    worker.register_kv_caches(([dk], [dv]))
    worker.start_load_kv()
    worker.wait_for_layer_load("layer.0")
    np.testing.assert_array_equal(dk, k)
    np.testing.assert_array_equal(dv, v)


def test_bound_metadata_layer_selection_is_respected(connector):
    tr = transport.InProcTransport()
    worker = connector.RngdKVConnector(connector.KVConnectorRole.WORKER, transport=tr, n_layers=2)
    worker.bind_connector_metadata(connector.RngdKVConnectorMetadata(req_id="actual"))
    worker.save_kv_layer("layer.1", (np.ones(1), np.ones(1)))
    worker.start_load_kv()
    worker.wait_for_layer_load("layer.1")
    assert tr._store == {}


def test_standalone_request_constructor_still_transfers_whole_layers(connector):
    tr = transport.InProcTransport()
    worker = connector.RngdKVConnector(connector.KVConnectorRole.WORKER,
                                       req_id="custom-request", transport=tr, n_layers=1)
    worker.save_kv_layer("layer.0", (np.ones(2), np.zeros(2)))
    meta = worker.build_connector_meta(None)
    assert meta.num_external_tokens == 0
    worker.bind_connector_metadata(meta)
    worker.start_load_kv()
    k, v = worker.wait_for_layer_load("layer.0")
    np.testing.assert_array_equal(k, np.ones(2))
    np.testing.assert_array_equal(v, np.zeros(2))


def test_multiple_pending_requests_rejected_without_replacing_first(connector):
    conn = connector.RngdKVConnector(connector.KVConnectorRole.SCHEDULER, n_layers=1)
    conn.update_state_after_alloc(types.SimpleNamespace(request_id="first"), None, 5)
    with pytest.raises(NotImplementedError, match="single|one request"):
        conn.update_state_after_alloc(types.SimpleNamespace(request_id="second"), None, 7)
    assert conn.build_connector_meta(None).req_id == "first"


@pytest.mark.parametrize("blocks", [[1, 2], ([1, 2],), types.SimpleNamespace(blocks=[1])])
def test_paged_block_mappings_are_explicitly_unsupported(connector, blocks):
    conn = connector.RngdKVConnector(connector.KVConnectorRole.SCHEDULER, n_layers=1)
    with pytest.raises(NotImplementedError, match="block|paged"):
        conn.update_state_after_alloc(types.SimpleNamespace(request_id="paged"), blocks, 5)


def test_vllm_scheduler_output_is_rejected_without_a_block_adapter(connector):
    conn = connector.RngdKVConnector(connector.KVConnectorRole.SCHEDULER, n_layers=1)
    output = types.SimpleNamespace(scheduled_new_reqs=[types.SimpleNamespace(req_id="paged")],
                                   scheduled_cached_reqs=[], num_scheduled_tokens={"paged": 5})
    with pytest.raises(NotImplementedError, match="scheduler|paged"):
        conn.build_connector_meta(output)


@pytest.mark.parametrize("method", ["save", "load"])
def test_paged_worker_context_is_rejected_before_transport(connector, method):
    tr = transport.InProcTransport()
    worker = connector.RngdKVConnector(connector.KVConnectorRole.WORKER, transport=tr, n_layers=1)
    with pytest.raises(NotImplementedError, match="paged|layout adapter"):
        if method == "save":
            worker.save_kv_layer("layer.0", (np.ones(1), np.zeros(1)),
                                 attn_metadata=types.SimpleNamespace(slot_mapping=[1]))
        else:
            worker.start_load_kv(forward_context=types.SimpleNamespace(attn_metadata={}))
    assert tr._store == {}


@pytest.mark.parametrize("req_id,count", [(None, 3), (["a", "b"], 3), ("r", -1), ("r", 1.5)])
def test_invalid_bound_metadata_is_rejected(connector, req_id, count):
    worker = connector.RngdKVConnector(connector.KVConnectorRole.WORKER, n_layers=1)
    meta = connector.RngdKVConnectorMetadata(req_id=req_id, load_layers=["layer.0"],
                                            num_external_tokens=count)
    with pytest.raises(ValueError, match="request|tokens"):
        worker.bind_connector_metadata(meta)
