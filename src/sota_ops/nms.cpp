#include <torch/library.h>
#include "sota_ops/nms.h"

namespace sota_ops::nms {

at::Tensor nms(const at::Tensor& ious, const at::Tensor& scores, double threshold) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::nms", "")
                       .typed<decltype(nms)>();
  return op.call(ious, scores, threshold);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sota_ops::nms(Tensor ious, Tensor scores, float threshold) -> Tensor"));
}

} // namespace sota_ops::nms
