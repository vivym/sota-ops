#include <cusparse.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <moderngpu/kernel_segreduce.hxx>

#include "sota_ops/seg_reduce.h"
#include "sota_ops/utils/thrust_allocator.h"

namespace sota_ops::seg_reduce::cuda {

template <typename scalar_t, typename index_t>
void segmented_reduce_test1_impl(
    at::Tensor& output,
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode) {
  auto num_segments = segment_offsets_begin.size(0);
  auto total_values = values.size(0);

  auto output_ptr = output.data_ptr<scalar_t>();
  auto values_ptr = values.data_ptr<scalar_t>();
  auto segment_offsets_begin_ptr = segment_offsets_begin.data_ptr<index_t>();
  auto segment_offsets_end_ptr = segment_offsets_end.data_ptr<index_t>();

  mgpu::standard_context_t context(false);
  using launch_t = mgpu::launch_params_t<32 * 6, 11>;

  mgpu::segreduce<launch_t>(
      values_ptr, total_values,
      segment_offsets_begin_ptr, num_segments,
      output_ptr,
      mgpu::plus_t<scalar_t>(),
      static_cast<scalar_t>(0),
      context);
}

at::Tensor segmented_reduce_test1(
    const at::Tensor& values,
    const at::Tensor& segment_offsets_begin,
    const at::Tensor& segment_offsets_end,
    int64_t mode) {
  auto num_segments = segment_offsets_begin.size(0);

  auto output = at::empty({num_segments}, values.options());

  AT_DISPATCH_FLOATING_TYPES(values.type(), "segmented_reduce_cuda_test1", [&] {
    if (segment_offsets_begin.scalar_type() == at::kInt) {
      segmented_reduce_test1_impl<scalar_t, int32_t>(
          output, values, segment_offsets_begin, segment_offsets_end, mode);
    // } else if (segment_offsets_begin.scalar_type() == at::kLong) {
    //   segmented_reduce_test1_impl<scalar_t, int64_t>(
    //       output, values, segment_offsets_begin, segment_offsets_end, mode);
    } else {
      AT_ERROR("Unsupported index type (segmented_reduce_cuda_test1)");
    }
  });

  return output;
}

TORCH_LIBRARY_IMPL(sota_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sota_ops::segmented_reduce_test1"),
         TORCH_FN(segmented_reduce_test1));
}

} // namespace sota_ops::seg_reduce::cuda
