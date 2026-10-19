"""Card selection: --chip -1 must pick the card k8s allocated, even in a privileged pod that sees every card.

    PYTHONPATH=src python -m pytest -q tests/test_devices.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rngd_pd import devices  # noqa: E402

# A node with cards npu0, npu1, npu3 (npu2 absent), as the device plugin and sysfs show it.
UUIDS = {0: "C0413086-0708-4606-8242-4947484F4646", 1: "89413086-070A-4702-834A-434B4D4F4D44",
         3: "89413086-0709-4707-8349-4645404A4442"}
POD = "62866677-caf9-4d2b-915a-6e757b05872f"


class Node:
    def __init__(self, visible, allocations):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.dev = os.path.join(root, "dev")
        self.sysfs = os.path.join(root, "sys")
        self.checkpoint = os.path.join(root, "kubelet_internal_checkpoint")
        os.makedirs(self.dev)
        for i in visible:
            open(os.path.join(self.dev, f"npu{i}mgmt"), "w").close()
            open(os.path.join(self.dev, f"npu{i}pe0"), "w").close()
        for i, u in UUIDS.items():
            d = os.path.join(self.sysfs, f"rngd!npu{i}mgmt")
            os.makedirs(d)
            with open(os.path.join(d, "device_uuid"), "w") as f:
                f.write(u.lower() + "\n")
        entries = [{"PodUID": pod, "ContainerName": c, "ResourceName": r, "DeviceIDs": ids}
                   for pod, c, r, ids in allocations]
        with open(self.checkpoint, "w") as f:
            json.dump({"Data": {"PodDeviceEntries": entries, "RegisteredDevices": {}}, "Checksum": 0}, f)

    def resolve(self, want=-1, env=None):
        return devices.resolve_chip(want, env=env or {}, dev_root=self.dev,
                                    checkpoint=self.checkpoint, sysfs=self.sysfs)


class ResolveChip(unittest.TestCase):
    def test_explicit_chip_wins(self):
        self.assertEqual(Node([0, 1, 3], []).resolve(want=1), 1)

    def test_privileged_pod_uses_its_allocation(self):
        node = Node([0, 1, 3], [
            ("other-pod", "worker", devices.RESOURCE, {"0": [UUIDS[0]]}),
            (POD, "worker", devices.RESOURCE, {"1": [UUIDS[3]]}),
        ])
        self.assertEqual(node.resolve(env={"POD_UID": POD}), 3)

    def test_flat_device_list_and_container_filter(self):
        node = Node([0, 1, 3], [
            (POD, "init", devices.RESOURCE, [UUIDS[0]]),
            (POD, "worker", devices.RESOURCE, [UUIDS[1]]),
        ])
        self.assertEqual(node.resolve(env={"POD_UID": POD, "CONTAINER_NAME": "worker"}), 1)

    def test_other_resources_ignored(self):
        node = Node([0, 1, 3], [(POD, "worker", "rebellions.ai/ATOM", {"0": ["rbln0"]}),
                                (POD, "worker", devices.RESOURCE, {"0": [UUIDS[0]]})])
        self.assertEqual(node.resolve(env={"POD_UID": POD}), 0)

    def test_unprivileged_pod_sees_one_card(self):
        self.assertEqual(Node([3], []).resolve(), 3)

    def test_many_visible_without_record_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "no allocation record"):
            Node([0, 1, 3], []).resolve()

    def test_unknown_uuid_is_an_error(self):
        node = Node([0, 1, 3], [(POD, "worker", devices.RESOURCE, {"0": ["00000000-0000-0000-0000-000000000000"]})])
        with self.assertRaisesRegex(RuntimeError, "not found"):
            node.resolve(env={"POD_UID": POD})

    def test_no_device(self):
        with self.assertRaisesRegex(RuntimeError, "no RNGD device"):
            Node([], []).resolve()


if __name__ == "__main__":
    unittest.main()
