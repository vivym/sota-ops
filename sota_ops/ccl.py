from typing import Tuple

import torch


@torch.no_grad()
def connected_components_labeling(
    indices: torch.Tensor,
    edges: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.sota_ops.connected_components_labeling(indices, edges)
