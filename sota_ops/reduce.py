import torch


@torch.no_grad()
def segmented_reduce(
    values: torch.Tensor,
    segment_offsets_begin: torch.Tensor,
    segment_offsets_end: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    values = values.contiguous()
    segment_offsets_begin = segment_offsets_begin.contiguous()
    segment_offsets_end = segment_offsets_end.contiguous()

    if mode == "sum":
        mode = 0
    elif mode == "min":
        mode = 1
    elif mode == "max":
        mode = 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.ops.sota_ops.segmented_reduce(
        values, segment_offsets_begin, segment_offsets_end, mode
    )


def test():
    import random

    num_segments = 128

    values = []
    segment_offsets = [0]
    for i in range(num_segments):
        num_values_per_seg = random.randint(1, 1023)
        values.append(torch.randn(num_values_per_seg, 3, dtype=torch.float64, device="cuda"))
        segment_offsets.append(segment_offsets[-1] + num_values_per_seg)

    values = torch.cat(values, dim=0)
    segment_offsets_begin = torch.tensor(segment_offsets[:-1], dtype=torch.int32, device="cuda")
    segment_offsets_end = torch.tensor(segment_offsets[1:], dtype=torch.int32, device="cuda")

    result = segmented_reduce(values, segment_offsets_begin, segment_offsets_end, mode="sum")

    for i in range(num_segments):
        values_per_seg = values[segment_offsets[i]:segment_offsets[i + 1]]
        assert torch.allclose(result[i], values_per_seg.sum(0)), i

    print("done")


if __name__ == "__main__":
    test()
