#include <torch/library.h>
#include "sota_ops/ccl.h"

namespace sota_ops::ccl {

std::tuple<at::Tensor, at::Tensor> connected_components_labeling(at::Tensor indices, at::Tensor edges) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::connected_components_labeling", "")
                       .typed<decltype(connected_components_labeling)>();
  return op.call(indices, edges);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
    "sota_ops::connected_components_labeling(Tensor indices, Tensor edges) -> (Tensor, Tensor)"));
}

} // namespace sota_ops::ccl
