#pragma once
#include <c10/metal/common.h>

#define THREADGROUP_MEMORY_BYTES (32 * 1024)
#define THREADGROUP_MEMORY_FLOATS (THREADGROUP_MEMORY_BYTES / 4)
#define MAX_THREADS_PER_THREADGROUP (1024)

template <typename idx_type_t = uint32_t>
struct SDPAParams {
  ::c10::metal::array<idx_type_t, 4> q_strides;
  ::c10::metal::array<idx_type_t, 4> k_strides;
  ::c10::metal::array<idx_type_t, 4> v_strides;
  ::c10::metal::array<idx_type_t, 4> mask_strides;
  ::c10::metal::array<idx_type_t, 4> out_strides;
  ::c10::metal::array<idx_type_t, 3> logsumexp_strides;
  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t L;
  uint32_t E;
  uint32_t S;
  uint32_t Ev;
  uint32_t d;
  uint32_t N;
  uint32_t B_r;
  uint32_t B_c;
  float scale;
};
