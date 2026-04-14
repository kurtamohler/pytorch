#pragma once
#include <c10/metal/common.h>

// Tile size for shared memory blocking. 32×32 = 1024 threads per threadgroup,
// which the limit.
#define TILE_SIZE 32

template <unsigned N = 4, typename idx_type_t = int32_t>
struct SDPAParams {
  ::c10::metal::array<idx_type_t, N> q_strides;
  ::c10::metal::array<idx_type_t, N> k_strides;
  ::c10::metal::array<idx_type_t, N> v_strides;
  ::c10::metal::array<idx_type_t, N> out_strides;
  ::c10::metal::array<idx_type_t, N> attn_strides;

  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t L;
  uint32_t E;
  uint32_t S;
  uint32_t Ev;

  float scale;
};

struct SDPANewParams {
  uint gqa_factor;
  uint N;
  ::c10::metal::array<uint, 3> qkv_head_strides;
  ::c10::metal::array<uint, 3> qkv_seq_strides;
  float scale;
  ::c10::metal::array<uint, 3> mask_strides;
  bool has_mask;
  ::c10::metal::array<uint, 3> qkv_batch_strides;
  uint num_q_heads;
  bool is_causal;
};
