#pragma once

#include <torch/types.h>

namespace sota_ops::ccl {

at::Tensor connected_components_labeling(at::Tensor indices, at::Tensor edges);

} // namespace sota_ops::ccl
