"""Explicit caller of the existing binding contract, not a vendor SDK API.

The caller authenticates immutable target files/manifests and a complete,
truthful byte plan before entry. Range parsing alone does not verify bytes.
"""
from .delta import parse_onggi_switchplan


def transition(owner, switch_plan, target):
    """Serialize a verified-plan transition with owner requests and replacement.

    Both concrete owners use reentrant locks. Only a declared reload plan may
    use lifecycle replacement; an admitted apply failure propagates unchanged.
    """
    parsed = parse_onggi_switchplan(switch_plan)
    with owner._lock:
        before = owner.resident()
        if before["state"] != "ready" or before["manifest"] != parsed["base_manifest"]:
            raise ValueError("switch base manifest is not ready and resident")
        profile = owner.describe(target)
        update = owner.plan(parsed)
        if (update["base_manifest"] != parsed["base_manifest"]
                or update["manifest"] != parsed["manifest"]):
            raise ValueError("binding plan identities disagree with switch plan")
        verification = None
        if update["mode"] == "units":
            verification = owner.apply(update, target)
            if (not isinstance(verification, dict) or verification.get("ok") is not True
                    or verification.get("manifest") != parsed["manifest"]):
                owner.close()
                raise RuntimeError("binding did not verify the target; owner closed")
        elif update["mode"] == "reload":
            owner.load(target, parsed["manifest"])
        else:
            raise ValueError("unsupported binding transition mode")
        after = owner.resident()
        if after["state"] != "ready" or after["manifest"] != parsed["manifest"]:
            owner.close()
            raise RuntimeError("binding did not admit the target; owner closed")
        return {"schema": "onggi.transition.v1", "base_manifest": parsed["base_manifest"],
                "manifest": parsed["manifest"], "mode": update["mode"],
                "reason": update.get("reason"), "profile": profile,
                "resident": after, "verification": verification}
