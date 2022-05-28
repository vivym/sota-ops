#include <torch/library.h>
#include "sota_ops/ccl.h"

namespace sota_ops::ccl {

at::Tensor connected_components_labeling(
    const at::Tensor& indices,
    const at::Tensor& edges,
    bool compacted) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::connected_components_labeling", "")
                       .typed<decltype(connected_components_labeling)>();
  return op.call(indices, edges, compacted);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sota_ops::connected_components_labeling(Tensor indices, Tensor edges, "
      "bool compacted) -> Tensor"));
}

} // namespace sota_ops::ccl
