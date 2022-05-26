from typing import Tuple

import torch


@torch.no_grad()
def ball_query(
    points: torch.Tensor,
    query: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    radius: float,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.sota_ops.ball_query(
        points, query, batch_indices, batch_offsets, radius, num_samples
    )
