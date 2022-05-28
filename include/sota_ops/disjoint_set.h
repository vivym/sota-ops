#pragma once

#include <cuda_runtime.h>

namespace sota_ops::disjoint_set {

namespace {
  template <typename dst_t, typename src_t>
  __forceinline__ __host__ __device__
  dst_t type_reinterpret(src_t value)
  {
      return *(reinterpret_cast<dst_t*>(&value));
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
  int32_t _atomicCAS(int32_t* address, int32_t compare, int32_t val) {
    return atomicCAS(address, compare, val);
  }
} // namespace

template <typename index_t, bool compress_path = true>
__host__ __device__
index_t find(index_t u, index_t* parent_ptr) {
  auto cur = parent_ptr[u];
  if (cur != u) {
    auto next = parent_ptr[cur], prev = u;
    while (cur > next) {
      if constexpr (compress_path) {
        parent_ptr[prev] = next;
        prev = cur;
      }
      cur = next;
      next = parent_ptr[cur];
    }
  }
  return cur;
}

template <typename index_t>
__host__ __device__
void merge(index_t u, index_t v, index_t* parent_ptr) {
  while (u != v) {
    if (u > v) {
      // swap
      auto tmp = u;
      u = v;
      v = tmp;
    }
    auto v_next = _atomicCAS(parent_ptr + v, v, u);
    if (v_next == v) {
      break;
    }
    v = v_next;
  }
}

} // namespace sota_ops::disjoint_set
