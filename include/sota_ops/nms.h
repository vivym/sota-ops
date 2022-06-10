#pragma once

#include <torch/types.h>

namespace sota_ops::nms {

at::Tensor nms(const at::Tensor& ious, const at::Tensor& scores, double threshold);

} // namespace sota_ops::nms
