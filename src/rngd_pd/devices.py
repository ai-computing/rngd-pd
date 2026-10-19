"""Which RNGD card this worker should use.

A pod that is granted one card by the Furiosa device plugin sees only that card,
under its host index (/dev/rngd/npu2* for the third card), so a hardcoded
rngd:0 fails inside k8s. The prefill pod has to be privileged (npuNp2pmem is
outside the plugin's allowlist), and a privileged pod sees every card on the
node, so "the first visible card" collides with the other pods.

The kubelet records each allocation in its device-plugin checkpoint, keyed by pod
UID, with the card's UUID as the device ID; sysfs maps that UUID to the index.
The chart mounts the checkpoint directory read-only and passes POD_UID, which
makes the allocated card recoverable even from a privileged pod.
"""
import glob
import json
import os
import re

RESOURCE = "furiosa.ai/rngd"
CHECKPOINT = "/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint"
SYSFS = "/sys/class/rngd_mgmt"


def visible_cards(dev_root="/dev/rngd"):
    """Host indices of the cards whose device nodes this container can see."""
    return sorted({int(m.group(1)) for p in glob.glob(os.path.join(dev_root, "npu*mgmt"))
                   for m in [re.search(r"npu(\d+)mgmt$", p)] if m})


def card_uuids(sysfs=SYSFS):
    """{device UUID (upper case): host index} from sysfs."""
    out = {}
    for p in glob.glob(os.path.join(sysfs, "*npu*mgmt", "device_uuid")):
        m = re.search(r"npu(\d+)mgmt", p)
        if m:
            with open(p) as f:
                out[f.read().strip().upper()] = int(m.group(1))
    return out


def allocated_uuids(pod_uid, container=None, checkpoint=CHECKPOINT, resource=RESOURCE):
    """Device IDs the kubelet allocated to this pod (and container, if given)."""
    with open(checkpoint) as f:
        data = json.load(f)["Data"]
    ids = []
    for entry in data.get("PodDeviceEntries") or []:
        if entry.get("PodUID") != pod_uid or entry.get("ResourceName") != resource:
            continue
        if container and entry.get("ContainerName") != container:
            continue
        devs = entry.get("DeviceIDs") or {}
        # Newer kubelets key DeviceIDs by NUMA node; older ones store a flat list.
        for v in (devs.values() if isinstance(devs, dict) else [devs]):
            ids.extend(v)
    return [d.upper() for d in ids]


def allocated_cards(env=os.environ, checkpoint=CHECKPOINT, sysfs=SYSFS):
    """Host indices of the cards k8s gave this pod, or None if that can't be known here."""
    pod_uid = env.get("POD_UID")
    checkpoint = env.get("KUBELET_DEVICE_CHECKPOINT", checkpoint)
    if not pod_uid or not os.path.exists(checkpoint):
        return None
    uuids = allocated_uuids(pod_uid, env.get("CONTAINER_NAME"), checkpoint)
    if not uuids:
        return None
    index = card_uuids(sysfs)
    missing = [u for u in uuids if u not in index]
    if missing:
        raise RuntimeError(f"allocated card(s) {missing} not found under {sysfs}")
    return sorted(index[u] for u in uuids)


def resolve_chip(want, env=os.environ, dev_root="/dev/rngd", checkpoint=CHECKPOINT, sysfs=SYSFS):
    """--chip N pins a card; --chip -1 means "the card k8s allocated to this pod".

    Order: the kubelet allocation (works in privileged pods), else the only
    visible card. Several visible cards and no allocation record is an error,
    because picking one would collide with the other pods on the node.
    """
    if want >= 0:
        return want
    cards = allocated_cards(env, checkpoint, sysfs)
    source = "allocated by k8s"
    if cards is None:
        cards = visible_cards(dev_root)
        source = "visible"
        if len(cards) > 1:
            raise RuntimeError(
                f"{len(cards)} RNGD cards visible ({cards}) and no allocation record: "
                "set POD_UID and mount the kubelet device-plugins directory, or pass --chip")
    if not cards:
        raise RuntimeError(f"no RNGD device visible under {dev_root}")
    print(f"[chip] {source}: {cards} -> using rngd:{cards[0]}", flush=True)
    return cards[0]
