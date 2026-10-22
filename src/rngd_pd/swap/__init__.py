"""Delta load: switch a running engine to another compiled variant (#17).

- variants: Variant (B, S, group) and BatchSelector (queue depth -> batch, with hysteresis)
- delta: changed byte ranges (onggi diff --ranges) -> changed safetensors tensors
- rngd: SwappableDecode, RNGD decode with one resident weight copy shared by every variant
"""
from rngd_pd.swap.variants import Variant, BatchSelector  # noqa: F401
