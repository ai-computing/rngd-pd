"""Shim: patch torch.library.register_fake so torchvision import survives.

torchvision 0.22 + torch 2.10 mismatch: `torchvision::nms` is not registered
at the C++ level (extension ABI broken), so `@register_fake("torchvision::nms")`
fails at module-import time, blocking ALL transformers/furiosa_llm imports.

Import this module FIRST, before any furiosa/transformers imports.
"""
import torch.library

_orig_rf = torch.library.register_fake


def _safe_register_fake(op, func=None, *, lib=None, _stacklevel=1, allow_override=False):
    out = _orig_rf(op, func, lib=lib, _stacklevel=_stacklevel,
                   allow_override=allow_override)
    if not isinstance(op, str):
        return out

    def safe_decorator(fn):
        try:
            return out(fn)
        except RuntimeError as e:
            if "does not exist" in str(e):
                return fn
            raise
    return safe_decorator


torch.library.register_fake = _safe_register_fake
