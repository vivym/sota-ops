import functools
import operator
import random

import numpy as np
import torch

from .reduce import segmented_reduce


@torch.no_grad()
def segmented_reduce_test1(
    values: torch.Tensor,
    segment_offsets_begin: torch.Tensor,
    segment_offsets_end: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    values = values.contiguous()
    segment_offsets_begin = segment_offsets_begin.contiguous()
    segment_offsets_end = segment_offsets_end.contiguous()

    if mode == "sum":
        mode_id = 0
    elif mode == "min":
        mode_id = 1
    elif mode == "max":
        mode_id = 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.ops.sota_ops.segmented_reduce_test1(
        values, segment_offsets_begin, segment_offsets_end, mode_id
    )


def gen_sparse_csr(n: int, m: int, device):
    crow_indices = [0]
    col_indices = []
    for i in range(n):
        nnz = random.randint(1, m)
        crow_indices.append(crow_indices[-1] + nnz)
        col_indices.append(torch.arange(nnz, dtype=torch.int64, device=device))

    crow_indices = torch.as_tensor(crow_indices, dtype=torch.int64, device=device)
    col_indices = torch.cat(col_indices)
    values = torch.randn_like(col_indices, dtype=torch.float32, device=device)

    return torch.sparse_csr_tensor(crow_indices, col_indices, values), crow_indices, col_indices


def test_spmv(sp: torch.Tensor):
    n, m = sp.shape
    dense = torch.ones(m, 1, dtype=sp.dtype, device=sp.device)

    for _ in range(10):
        result = sp @ dense

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    elapsed_times = []
    for _ in range(100):
        start_event.record()
        result = sp @ dense
        end_event.record()
        torch.cuda.synchronize()
        elapsed_times.append(start_event.elapsed_time(end_event))

    print("spmv", np.mean(elapsed_times))

    return result.view(-1)


def test_pytorch(sp: torch.Tensor, crow_indices, col_indices):
    lengths = crow_indices[1:] - crow_indices[:-1]
    values = sp.values()

    for _ in range(10):
        result = torch.segment_reduce(values, "sum", lengths=lengths)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    elapsed_times = []
    for _ in range(100):
        start_event.record()
        result = torch.segment_reduce(values, "sum", lengths=lengths)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_times.append(start_event.elapsed_time(end_event))

    print("pytorch", np.mean(elapsed_times))

    return result


def test_cub(sp: torch.Tensor, crow_indices, col_indices):
    values = sp.values().view(-1, 1)

    for _ in range(10):
        result = segmented_reduce(values, crow_indices[:-1], crow_indices[1:])

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    elapsed_times = []
    for _ in range(100):
        start_event.record()
        result = segmented_reduce(values, crow_indices[:-1], crow_indices[1:])
        end_event.record()
        torch.cuda.synchronize()
        elapsed_times.append(start_event.elapsed_time(end_event))

    print("cub", np.mean(elapsed_times))

    return result.view(-1)


def test_moderngpu(sp: torch.Tensor, crow_indices, col_indices):
    values = sp.values()
    crow_indices = crow_indices.to(dtype=torch.int32)
    col_indices = col_indices.to(dtype=torch.int32)

    for _ in range(10):
        result = segmented_reduce_test1(values, crow_indices[:-1], crow_indices[1:])

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    elapsed_times = []
    for _ in range(100):
        start_event.record()
        result = segmented_reduce_test1(values, crow_indices[:-1], crow_indices[1:])
        end_event.record()
        torch.cuda.synchronize()
        elapsed_times.append(start_event.elapsed_time(end_event))

    print("moderngpu", np.mean(elapsed_times))

    return result


def test_seg_reduce():
    n = 100000
    m = 64

    sp, crow_indices, col_indices = gen_sparse_csr(n, m, torch.device("cuda"))

    res_spmv = test_spmv(sp)

    res_pytorch = test_pytorch(sp, crow_indices, col_indices)

    res_cub = test_cub(sp, crow_indices, col_indices)

    res_moderngpu = test_moderngpu(sp, crow_indices, col_indices)

    print(torch.allclose(res_spmv, res_pytorch, rtol=1e-3, atol=1e-5))
    print(torch.allclose(res_spmv, res_cub, rtol=1e-3, atol=1e-5))


if __name__ == "__main__":
    test_seg_reduce()
