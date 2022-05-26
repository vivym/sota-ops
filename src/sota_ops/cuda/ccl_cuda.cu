#include <torch/library.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "sota_ops/ccl.h"

namespace sota_ops::ccl {

template <typename Scalar>
void connected_components_labeling_cuda(
    at::Tensor &labels,
    at::Tensor indices,
    at::Tensor edges) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
}

std::tuple<at::Tensor, at::Tensor> connected_components_labeling_cuda(at::Tensor indices, at::Tensor edges) {
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");

  TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
  TORCH_CHECK(edges.dim() == 1, "edges must be a 1D tensor");

  auto labels = at::empty({indices.size(0) - 1}, indices.options());

  return {labels, labels};
}

TORCH_LIBRARY_IMPL(sota_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sota_ops::connected_components_labeling"),
         TORCH_FN(connected_components_labeling));
}

} // namespace sota_ops::ccl
