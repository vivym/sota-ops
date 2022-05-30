#include <torch/library.h>
#include "sota_ops/reduce.h"

namespace sota_ops::reduce {

at::Tensor segmented_reduce(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::segmented_reduce", "")
                       .typed<decltype(segmented_reduce)>();
  return op.call(values, segment_offsets_begin, segment_offsets_end, mode);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sota_ops::segmented_reduce(Tensor values, Tensor segment_offsets_begin, "
      "Tensor segment_offsets_end, int mode) -> Tensor"));
}

} // namespace sota_ops::reduce
