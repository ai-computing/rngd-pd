"""Delta load: switch a running engine to another compiled variant (#17).

- variants: Variant (B, S, group) and BatchSelector (queue depth -> batch, with hysteresis)
- delta: changed byte ranges (onggi diff --ranges) -> changed safetensors tensors
- rngd: SwappableDecode, RNGD decode with one resident weight copy shared by every variant
- atom: ATOM sessions (bucketing: variants compiled together), HostPool (fast reload of the next session)
- atom_inputs: ATOM weights as torch-rbln device inputs (measured, not a serving path on rebel 0.11)
"""
from rngd_pd.swap.variants import Variant, BatchSelector  # noqa: F401
