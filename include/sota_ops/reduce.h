#pragma once

#include <torch/types.h>

namespace sota_ops::reduce {

// 0: sum; 1: min; 2: max
at::Tensor segmented_reduce(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode);

} // namespace sota_ops::reduce
