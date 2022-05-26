#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "sota_ops/ball_query.h"
#include "sota_ops/utils/thrust_allocator.h"

namespace sota_ops::ball_query {

template <typename scalar_t, typename index_t, typename policy_t>
void ball_query_cuda_impl_thrust(
    const policy_t& policy,
    index_t* __restrict__ indices_ptr,
    index_t* __restrict__ num_points_per_query_ptr,
    scalar_t* __restrict__ points_ptr,
    scalar_t* __restrict__ query_ptr,
    index_t* __restrict__ batch_indices_ptr,
    index_t* __restrict__ batch_offsets_ptr,
    scalar_t radius2,
    index_t num_samples,
    index_t num_queries) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_queries),
      [=] __host__ __device__ (index_t i) {
        auto batch_idx = batch_indices_ptr[i];
        auto from = batch_offsets_ptr[batch_idx];
        auto to = batch_offsets_ptr[batch_idx + 1];

        auto q_x = query_ptr[i * 3 + 0];
        auto q_y = query_ptr[i * 3 + 1];
        auto q_z = query_ptr[i * 3 + 2];

        index_t cnt = 0;
        for (auto k = from; k < to && cnt < num_samples; k++) {
          auto x = points_ptr[k * 3 + 0];
          auto y = points_ptr[k * 3 + 1];
          auto z = points_ptr[k * 3 + 2];
          auto d2 = (q_x - x) * (q_x - x) + (q_y - y) * (q_y - y) +
                    (q_z - z) * (q_z - z);
          if (d2 < radius2) {
            indices_ptr[i * num_samples + cnt] = k;
            cnt++;
          }
        }
        num_points_per_query_ptr[i] = cnt;
      });
}

template <typename scalar_t, typename index_t>
void ball_query_cuda_impl(
    at::Tensor& indices,
    at::Tensor& num_points_per_query,
    at::Tensor points,
    at::Tensor query,
    at::Tensor batch_indices,
    at::Tensor batch_offsets,
    double radius,
    int64_t num_samples) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto num_queries = points.size(0);

  auto indices_ptr = indices.data_ptr<index_t>();
  auto num_points_per_query_ptr = num_points_per_query.data_ptr<index_t>();
  auto points_ptr = points.data_ptr<scalar_t>();
  auto query_ptr = query.data_ptr<scalar_t>();
  auto batch_indices_ptr = batch_indices.data_ptr<index_t>();
  auto batch_offsets_ptr = batch_offsets.data_ptr<index_t>();

  ball_query_cuda_impl_thrust<scalar_t, index_t>(
      policy,
      indices_ptr,
      num_points_per_query_ptr,
      points_ptr,
      query_ptr,
      batch_indices_ptr,
      batch_offsets_ptr,
      static_cast<scalar_t>(radius * radius),
      static_cast<index_t>(num_samples),
      static_cast<index_t>(num_queries));
}

std::tuple<at::Tensor, at::Tensor> ball_query_cuda(
    at::Tensor points,
    at::Tensor query,
    at::Tensor batch_indices,
    at::Tensor batch_offsets,
    double radius,
    int64_t num_samples) {
  TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(batch_indices.is_cuda(), "batch_indices must be a CUDA tensor");
  TORCH_CHECK(batch_offsets.is_cuda(), "batch_offsets must be a CUDA tensor");

  TORCH_CHECK(points.dim() == 2, "points must be a 2D tensor");
  TORCH_CHECK(query.dim() == 2, "edges must be a 2D tensor");
  TORCH_CHECK(batch_indices.dim() == 1, "batch_indices must be a 1D tensor");
  TORCH_CHECK(batch_offsets.dim() == 1, "batch_offsets must be a 1D tensor");

  TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(batch_indices.is_contiguous(), "batch_indices must be contiguous");
  TORCH_CHECK(batch_offsets.is_contiguous(), "batch_offsets must be contiguous");

  auto indices = at::empty({query.size(0), num_samples}, batch_indices.options());
  auto num_points_per_query = at::empty({query.size(0)}, batch_indices.options());

  AT_DISPATCH_FLOATING_TYPES(points.type(), "ball_query_cuda", [&] {
    if (batch_indices.scalar_type() == at::kInt) {
      ball_query_cuda_impl<scalar_t, int32_t>(
          indices,
          num_points_per_query,
          points,
          query,
          batch_indices,
          batch_offsets,
          radius,
          num_samples);
    } else if (batch_indices.scalar_type() == at::kLong) {
      ball_query_cuda_impl<scalar_t, int64_t>(
          indices,
          num_points_per_query,
          points,
          query,
          batch_indices,
          batch_offsets,
          radius,
          num_samples);
    } else {
      AT_ERROR("Unsupported type");
    }
  });

  return {indices, num_points_per_query};
}

TORCH_LIBRARY_IMPL(sota_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sota_ops::ball_query"),
         TORCH_FN(ball_query_cuda));
}

} // namespace sota_ops::ball_query