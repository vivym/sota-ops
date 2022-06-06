#include <torch/library.h>
#include "sota_ops/expand.h"

namespace sota_ops::expand {

std::tuple<at::Tensor, at::Tensor> expand_csr(const at::Tensor& offsets, int64_t output_size) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::expand_csr", "")
                       .typed<decltype(expand_csr)>();
  return op.call(offsets, output_size);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sota_ops::expand_csr(Tensor offsets, int output_size) -> (Tensor, Tensor)"));
}

} // namespace sota_ops::expand
