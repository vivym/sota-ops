#pragma once

#include <torch/types.h>

namespace sota_ops::iou {

at::Tensor instance_seg_iou_csr(
    const at::Tensor& proposal_indices,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance);

at::Tensor instance_seg_iou(
    const at::Tensor& proposal_indices_begin,
    const at::Tensor& proposal_indices_end,
    const at::Tensor& instance_labels,
    const at::Tensor& num_points_per_instance);

} // namespace sota_ops::iou