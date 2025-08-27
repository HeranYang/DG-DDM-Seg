# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # b x hw x c
        x = self.layernorm(x)  # b x hw x c
        x = x.transpose(1, 2).view(b, c, h, w)  # b x c x h x w
        return x
