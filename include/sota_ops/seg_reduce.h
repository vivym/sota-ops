#pragma once

#include <torch/types.h>

namespace sota_ops::seg_reduce {

at::Tensor segmented_reduce_test1(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode);

} // namespace sota_ops::seg_reduce
