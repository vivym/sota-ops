from typing import Optional

import torch


@torch.no_grad()
def connected_components_labeling(
    indices: torch.Tensor,
    edges: torch.Tensor,
    compacted: bool = True,
) -> torch.Tensor:
    indices = indices.contiguous()
    edges = edges.contiguous()

    return torch.ops.sota_ops.connected_components_labeling(
        indices, edges, compacted
    )
