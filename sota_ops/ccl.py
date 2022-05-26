import torch


@torch.no_grad()
def connected_components_labeling(
    indices: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.sota_ops.connected_components_labeling(indices, edges)


def test():
    indices = torch.as_tensor([
        0, 1, 1, 3, 4, 6, 8, 10
    ])
    edges = torch.as_tensor([
        # 0
        4,
        # 1
        # 2
        5, 6,
        # 3
        4,
        # 4
        0, 3,
        # 5
        6, 2,
        # 6,
        5, 2
    ])
    indices = indices.cuda()
    edges = edges.cuda()
    labels = connected_components_labeling(indices, edges)
    print(labels)


if __name__ == "__main__":
    test()
