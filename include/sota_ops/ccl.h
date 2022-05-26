#pragma once

#include <torch/types.h>
#include <vector>
#include <tuple>

namespace sota_ops::ccl {

std::tuple<at::Tensor, at::Tensor> connected_components_labeling(at::Tensor indices, at::Tensor edges);

} // namespace sota_ops::ccl
