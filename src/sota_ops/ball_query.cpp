#include <torch/library.h>
#include "sota_ops/ball_query.h"

namespace sota_ops::ball_query {

std::tuple<at::Tensor, at::Tensor> ball_query(
    at::Tensor points,
    at::Tensor query,
    at::Tensor batch_indices,
    at::Tensor batch_offsets,
    double radius,
    int64_t num_samples) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::ball_query", "")
                       .typed<decltype(ball_query)>();
  return op.call(points, query, batch_indices, batch_offsets, radius, num_samples);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sota_ops::ball_query(Tensor points, Tensor query, Tensor batch_indices, "
      "Tensor batch_offsets, float radius, int num_samples) -> (Tensor, Tensor)"));
}

} // namespace sota_ops::ball_query
