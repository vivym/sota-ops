#pragma once

#include <torch/types.h>
#include <tuple>

namespace sota_ops::ball_query {

std::tuple<at::Tensor, at::Tensor> ball_query(
    at::Tensor points,
    at::Tensor query,
    at::Tensor batch_indices,
    at::Tensor batch_offsets,
    double radius,
    int64_t num_samples);

} // namespace sota_ops::ball_query
