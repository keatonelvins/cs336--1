"""
uv run pytest -k test_linear
"""

import torch
import torch.nn as nn
from typing import Optional

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        super().__init__()

        std_dev = (2 / (in_features + out_features))**0.5
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, 0, std_dev, -3*std_dev, 3*std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T # use row-major order