#pragma once

#include <torch/types.h>

namespace sota_ops::ccl {

at::Tensor connected_components_labeling(
    const at::Tensor& indices,
    const at::Tensor& edges,
    bool compacted);

} // namespace sota_ops::ccl
