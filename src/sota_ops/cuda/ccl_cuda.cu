#include <torch/library.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include "sota_ops/ccl.h"
#include "sota_ops/utils/thrust_allocator.h"

namespace sota_ops::ccl {

namespace {
  template <typename dst_t, typename src_t>
  __forceinline__ __host__ __device__
  dst_t type_reinterpret(src_t value)
  {
      return *(reinterpret_cast<dst_t*>(&value));
  }

  template <typename index_t>
  __device__ index_t representative(const index_t idx, index_t* const __restrict__ labels_ptr) {
    index_t cur = labels_ptr[idx];
    if (cur != idx) {
      index_t next = labels_ptr[cur], prev = idx;
      while (cur > next) {
        labels_ptr[prev] = next;
        prev = cur;
        cur = next;
        next = labels_ptr[cur];
      }
    }
    return cur;
  }

  template <typename index_t, typename policy_t>
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
          const auto from = indices_ptr[u], to = indices_ptr[u + 1];
          auto v = u;
          for (auto i = from; i < to; i++) {
            if (edges_ptr[i] < v) {
              v = edges_ptr[i];
              break;
            }
          }
          labels_ptr[u] = v;

          // if (u == 0) {
          //   top_l = 0;
          //   pos_l = 0;
          //   top_h = num_nodes - 1;
          //   pos_h = num_nodes - 1;
          // }
        });
  }

  __forceinline__ __device__
  int64_t _atomicCAS(int64_t* address, int64_t compare, int64_t val) {
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    auto ret = atomicCAS(
        reinterpret_cast<unsigned long long *>(address),
        type_reinterpret<unsigned long long>(compare),
        type_reinterpret<unsigned long long>(val));
    return type_reinterpret<int64_t>(ret);
  }

  __forceinline__ __device__
  int64_t _atomicCAS(int32_t* address, int32_t compare, int32_t val) {
    return atomicCAS(address, compare, val);
  }

  template <typename index_t, typename policy_t>
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
          const auto label = labels_ptr[u];
          if (label == u) {
            // 0 degree
            return;
          }

          const auto from = indices_ptr[u], to = indices_ptr[u + 1];
          const auto degree = to - from;
          if (degree > 16) {
            // index_t idx;
            // if (degree <= 352) {
            //   idx = atomicAdd(&top_l, 1);
            // } else {
            //   idx = atomicAdd(&top_h, -1);
            // }
            // working_list_ptr[idx] = u;
          } else {
            auto u_label = representative(u, labels_ptr);
            for (auto i = from; i < to; i++) {
              const auto v = edges_ptr[i];
              if (u <= v) {
                continue;
              }

              auto v_label = representative(v, labels_ptr);

              while (u_label != v_label) {
                if (u_label < v_label) {
                  auto v_next = _atomicCAS(labels_ptr + v_label, v_label, u_label);
                  if (v_next == v_label) {
                    break;
                  }
                  v_label = v_next;
                } else {
                  auto u_next = _atomicCAS(labels_ptr + u_label, u_label, v_label);
                  if (u_next == u_label) {
                    break;
                  }
                  u_label = u_next;
                }
              }
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
          auto cur = labels_ptr[u];
          if (cur != u) {
            auto next = labels_ptr[cur];
            while (cur > next) {
              cur = next;
              next = labels_ptr[cur];
            }
            labels_ptr[u] = cur;
          }
        });
  }
}

template <typename index_t>
void connected_components_labeling_cuda_impl(
    // outputs
    at::Tensor &labels,
    // inputs
    at::Tensor indices,
    at::Tensor edges) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(utils::ThrustAllocator()).on(stream);

  auto num_nodes = indices.size(0) - 1;
  auto num_edges = edges.size(0);

  auto labels_ptr = labels.data_ptr<index_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto edges_ptr = edges.data_ptr<index_t>();

  auto work_queue = at::empty_like(labels);
  auto work_queue_ptr = work_queue.data_ptr<index_t>();

  // __device__ index_t top_l, pos_l, top_h, pos_h;
  initialize<index_t>(policy, num_nodes, indices_ptr, edges_ptr, labels_ptr);
  process_1<index_t>(policy, num_nodes, indices_ptr, edges_ptr, labels_ptr);
  flatten<index_t>(policy, num_nodes, labels_ptr);
}

at::Tensor connected_components_labeling_cuda(at::Tensor indices, at::Tensor edges) {
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");

  TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
  TORCH_CHECK(edges.dim() == 1, "edges must be a 1D tensor");

  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(edges.is_contiguous(), "edges must be contiguous");

  auto labels = at::empty({indices.size(0) - 1}, indices.options());

  if (indices.scalar_type() == at::kInt) {
    connected_components_labeling_cuda_impl<int32_t>(labels, indices, edges);
  } else if (indices.scalar_type() == at::kLong) {
    connected_components_labeling_cuda_impl<int64_t>(labels, indices, edges);
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
