from typing import Tuple, Optional

import torch


@torch.no_grad()
def ball_query(
    points: torch.Tensor,
    query: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    radius: float,
    num_samples: int,
    point_labels: Optional[torch.Tensor] = None,
    query_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    points = points.contiguous()
    query = query.contiguous()
    batch_indices = batch_indices.contiguous()
    batch_offsets = batch_offsets.contiguous()

    if point_labels is not None:
        point_labels = point_labels.contiguous()

    if query_labels is not None:
        query_labels = query_labels.contiguous()

    return torch.ops.sota_ops.ball_query(
        points, query, batch_indices, batch_offsets, radius, num_samples,
        point_labels, query_labels,
    )
