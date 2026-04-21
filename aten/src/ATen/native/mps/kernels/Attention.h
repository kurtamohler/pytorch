#pragma once
#include <c10/metal/common.h>

#define TILE_ROWS 32
#define TILE_COLS 64
#define NUM_THREADS (((TILE_ROWS * TILE_COLS) > 1024) ? 1024 : (TILE_ROWS * TILE_COLS))

template <unsigned N = 4, typename idx_type_t = uint32_t>
struct SDPAParams {
  ::c10::metal::array<idx_type_t, N> q_strides;
  ::c10::metal::array<idx_type_t, N> k_strides;
  ::c10::metal::array<idx_type_t, N> v_strides;
  ::c10::metal::array<idx_type_t, N> mask_strides;
  ::c10::metal::array<idx_type_t, N> out_strides;
  ::c10::metal::array<idx_type_t, N> attn_strides;
  uint32_t batch_size;
  uint32_t num_heads;
  uint32_t L;
  uint32_t E;
  uint32_t S;
  uint32_t Ev;
  uint32_t gqa_factor;
  float scale;
};
