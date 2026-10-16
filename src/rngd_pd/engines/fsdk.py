"""One place that knows which furiosa-torch is installed.

2026.3.0 moved the pieces this repo compiles against and dropped two of them:

    furiosa.torch.native_device            -> furiosa.torch
    furiosa.torch.custom_ops.CompileModule -> furiosa.torch.CompileModule
    furiosa.torch.custom_ops.TacticKernelModule -> furiosa.torch.TclModule,
        and the DSL grammar dropped inline axis sizes (see _symbolic_axes)
    furiosa.torch.custom_ops.edf.EdfModule -> gone

Fifteen call sites imported those directly, so an upgrade meant touching all
of them and there was no way to measure the new backend without committing to
it. Importing from here instead lets both versions run, which is what a
before/after on the same kernels needs.
"""
import torch  # noqa: F401  — must precede furiosa.torch: it loads libc10

import furiosa.torch as _ft

try:
    import importlib.metadata as _md
    SDK = _md.version("furiosa-torch")
except Exception:                                     # pragma: no cover
    SDK = "unknown"

NEW = tuple(int(x) for x in SDK.split(".")[:2] if x.isdigit()) >= (2026, 3)

if NEW:
    import re as _re

    from furiosa.torch import (CompileModule, DfgModule, TclModule,        # noqa
                               set_fusion, set_mesh_kind, MeshKind)
    from furiosa.torch import ir                                           # noqa

    _AXES = _re.compile(r"\[([A-Za-z0-9_=,\s]+)\]")

    def _symbolic_axes(dsl):
        """[B=4,N=4096]/bf16 -> [B,N]/bf16.

        2026.3.0 takes axis extents from the traced input shapes instead of
        the signature, and its parser rejects the `=size` form outright. Every
        kernel here already gets its real shapes from the sample it is
        exported with, so dropping the sizes is the whole port.
        """
        return _AXES.sub(
            lambda m: "[" + ",".join(p.split("=")[0].strip()
                                     for p in m.group(1).split(",")) + "]",
            dsl)

    def TacticKernelModule(dsl):
        return TclModule(_symbolic_axes(dsl))

    EdfModule = None            # no EDF container in 2026.3.0
else:
    from furiosa.torch.custom_ops import CompileModule, TacticKernelModule  # noqa
    from furiosa.torch.custom_ops.dfg import DfgModule                      # noqa
    from furiosa.torch.custom_ops.edf import EdfModule                      # noqa
    from furiosa.torch import native_device as _nd
    set_fusion, set_mesh_kind, MeshKind = (_nd.set_fusion, _nd.set_mesh_kind,
                                           _nd.MeshKind)

# Config was renamed CompilerConfig in 2026.3.0 and the enums moved into the
# native config module. The field set is identical either way.
try:
    from furiosa.torch.compiler import (Config as FCConfig,                # noqa
                                        TacticHintConfig, TacticSortingPolicy)
except ImportError:
    from furiosa.torch._C.config.compiler import (                         # noqa
        CompilerConfig as FCConfig, TacticHintConfig, TacticSortingPolicy)


# 2026.3.0 renamed the LLM hints along the axis that actually decides the
# tactic -- Decode/Prefill became IOBound/ComputeBound -- so the names this
# repo and DECODE_TACTIC_HINT use no longer resolve. Map them by what they
# meant: a decode step is memory-bound, a prefill pass is compute-bound.
_HINT_ALIAS = {
    "ForLlmModelDecode": "ForLlmModelIOBound",
    "ForLlmModelPrefill": "ForLlmModelComputeBound",
    "ForLlmModelPrefillLongSequence": "ForLlmModelComputeBound",
} if NEW else {}


def hint(name):
    """TacticHintConfig by name, accepting either SDK's spelling.

    Which hint is fastest is measured, not implied by the name -- on 2026.2.0
    the winner at B=8 was ForMiscModel -- so callers still choose. This only
    keeps a name from failing to resolve.
    """
    return getattr(TacticHintConfig, _HINT_ALIAS.get(name, name))


# 2026.3.0 renamed the torch device backend as well: `rngd:0` no longer parses.
DEV = "furiosa" if NEW else "rngd"


def dev_str(chip=0):
    """The torch device string for a card index, on whichever SDK is loaded."""
    return f"{DEV}:{chip}"


if NEW:
    from furiosa.torch import current_stream as _current_stream               # noqa

    def sync(chip=0):
        """Block until the card has finished the work already submitted.

        2026.3.0 runs the module asynchronously: a call returns as soon as it
        is queued, so timing one without this measures the enqueue, not the
        kernel (0.08 ms against a real 1.7). 2026.2.0 was synchronous and this
        is a no-op there.
        """
        _current_stream(dev_str(chip)).synchronize()
else:
    def sync(chip=0):
        pass


def open_device(fusion=8, mesh=2):
    """The 8-PE fused device every engine here runs on.

    RNGD's fused nodes top out at four PEs, so eight is two groups joined as a
    mesh -- which is why this pair of calls has to precede any device tensor.
    """
    set_fusion(fusion)
    set_mesh_kind(MeshKind.Mesh(mesh))
