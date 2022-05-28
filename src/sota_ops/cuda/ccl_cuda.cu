#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "sota_ops/ccl.h"
#include "sota_ops/disjoint_set.h"
#include "sota_ops/utils/thrust_allocator.h"

namespace sota_ops::ccl {

namespace {
  template <typename index_t, bool compacted>
  __host__ __device__
  inline std::tuple<index_t, index_t> get_edge_range(
      const index_t u, const index_t* const __restrict__ indices_ptr) {
    if constexpr (compacted) {
      return {indices_ptr[u], indices_ptr[u + 1]};
    } else {
      return {indices_ptr[u * 2], indices_ptr[u * 2 + 1]};
    }
  }

  template <typename index_t, bool compacted, typename policy_t>
  void initialize(
      const policy_t& policy,
      int num_nodes,
      index_t* __restrict__ indices_ptr,
      index_t* __restrict__ edges_ptr,
      index_t* __restrict__ labels_ptr) {
    // initialize with first smaller neighbor ID
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes),
        [=] __host__ __device__ (index_t u) {
          const auto [from, to] = get_edge_range<index_t, compacted>(u, indices_ptr);
          auto parent = u;
          for (auto i = from; i < to; i++) {
            const auto v = edges_ptr[i];
            if (v < parent) {
              parent = v;
              break;
            }
          }
          labels_ptr[u] = parent;
        });
  }

  template <typename index_t, bool compacted, typename policy_t>
  void process_1(
      const policy_t& policy,
      int num_nodes,
      index_t* __restrict__ indices_ptr,
      index_t* __restrict__ edges_ptr,
      index_t* __restrict__ labels_ptr) {
    // process low-degree nodes at thread granularity and fill work queue
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes),
        [=] __host__ __device__ (index_t u) {
          if (labels_ptr[u] == u) {
            // 0 degree
            return;
          }

          const auto [from, to] = get_edge_range<index_t, compacted>(u, indices_ptr);
          const auto degree = to - from;
          if (degree > 1600000) {
            // index_t idx;
            // if (degree <= 352) {
            //   idx = atomicAdd(&top_l, 1);
            // } else {
            //   idx = atomicAdd(&top_h, -1);
            // }
            // working_list_ptr[idx] = u;
          } else {
            auto u_label = disjoint_set::find(u, labels_ptr);
            for (auto i = from; i < to; i++) {
              const auto v = edges_ptr[i];
              if (u <= v) {
                continue;
              }

              const auto v_label = disjoint_set::find(v, labels_ptr);
              disjoint_set::merge(u_label, v_label, labels_ptr);
            }
          }
        });
  }

  template <typename index_t, typename policy_t>
  void flatten(
      const policy_t& policy,
      int num_nodes,
      index_t* __restrict__ labels_ptr) {
    // flatten
    thrust::for_each(
        policy,
        thrust::make_counting_iterator<index_t>(0),
        thrust::make_counting_iterator<index_t>(num_nodes),
        [=] __host__ __device__ (index_t u) {
          const auto parent = disjoint_set::find<index_t, false>(u, labels_ptr);
          if (parent != u) {
            labels_ptr[u] = parent;
          }
        });
  }
}

template <typename index_t, bool compacted>
void connected_components_labeling_cuda_impl(
    // outputs
    at::Tensor &labels,
    // inputs
    const at::Tensor& indices,
    const at::Tensor& edges) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  index_t num_nodes = labels.size(0);
  index_t num_edges = edges.size(0);

  auto labels_ptr = labels.data_ptr<index_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto edges_ptr = edges.data_ptr<index_t>();

  auto indices_cpu = indices.cpu();
  auto indices_cpu_ptr = indices_cpu.data_ptr<index_t>();

  auto edges_cpu = edges.cpu();
  auto edges_cpu_ptr = edges_cpu.data_ptr<index_t>();

  initialize<index_t, compacted>(
      policy, num_nodes, indices_ptr, edges_ptr, labels_ptr);
  process_1<index_t, compacted>(
      policy, num_nodes, indices_ptr, edges_ptr, labels_ptr);
  flatten<index_t>(policy, num_nodes, labels_ptr);
}

at::Tensor connected_components_labeling_cuda(
    const at::Tensor& indices,
    const at::Tensor& edges,
    bool compacted) {
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");

  TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
  TORCH_CHECK(edges.dim() == 1, "edges must be a 1D tensor");

  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(edges.is_contiguous(), "edges must be contiguous");

  auto num_nodes = compacted ? indices.size(0) - 1 : indices.size(0) / 2;
  auto labels = at::empty({num_nodes}, indices.options());

  if (indices.scalar_type() == at::kInt) {
    if (compacted) {
      connected_components_labeling_cuda_impl<int32_t, true>(labels, indices, edges);
    } else {
      connected_components_labeling_cuda_impl<int32_t, false>(labels, indices, edges);
    }
  } else if (indices.scalar_type() == at::kLong) {
    if (compacted) {
      connected_components_labeling_cuda_impl<int64_t, true>(labels, indices, edges);
    } else {
      connected_components_labeling_cuda_impl<int64_t, false>(labels, indices, edges);
    }
  } else {
    AT_ERROR("Unsupported type");
  }

  return labels;
}

TORCH_LIBRARY_IMPL(sota_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sota_ops::connected_components_labeling"),
         TORCH_FN(connected_components_labeling_cuda));
}

} // namespace sota_ops::ccl
