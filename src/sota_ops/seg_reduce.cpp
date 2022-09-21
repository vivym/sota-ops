#include <torch/library.h>
#include "sota_ops/seg_reduce.h"

namespace sota_ops::seg_reduce {

at::Tensor segmented_reduce_test1(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sota_ops::segmented_reduce_test1", "")
                       .typed<decltype(segmented_reduce_test1)>();
  return op.call(values, segment_offsets_begin, segment_offsets_end, mode);
}

TORCH_LIBRARY_FRAGMENT(sota_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sota_ops::segmented_reduce_test1(Tensor values, Tensor segment_offsets_begin, "
      "Tensor segment_offsets_end, int mode) -> Tensor"));
}

} // namespace sota_ops::seg_reduce
