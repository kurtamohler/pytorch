// SDPA decode kernels (short Q sequence). Three variants:
//   sdpa_vector            - one-pass, for short qL with moderate kL
//   sdpa_vector_2pass_1    - two-pass pass 1, splits the K loop across blocks
//   sdpa_vector_2pass_2    - two-pass pass 2, aggregates per-block partials
//
// Adapted from MLX:
//   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#pragma once

#include <ATen/native/mps/kernels/Attention.h>
#include <ATen/native/mps/kernels/PrefillAttention.h>

template <typename T, int D, int V = D, bool is_causal = false>
[[kernel]] void sdpa_vector(const device T* queries [[buffer(0)]],
                            const device T* keys [[buffer(1)]],
                            const device T* values [[buffer(2)]],
                            device T* out [[buffer(3)]],
                            const constant uint& gqa_factor [[buffer(4)]],
                            const constant uint& N [[buffer(5)]],
                            const constant uint3& qkv_head_strides [[buffer(6)]],
                            const constant uint3& qkv_seq_strides [[buffer(7)]],
                            const constant float& scale [[buffer(8)]],
                            const device bool* mask [[buffer(9)]],
                            const constant uint3& mask_strides [[buffer(10)]],
                            const constant bool& has_mask [[buffer(11)]],
                            const constant uint4& qkv_batch_strides_heads [[buffer(12)]],
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint3 tpg [[threadgroups_per_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint qk_per_thread = D / BD;
  constexpr uint v_per_thread = V / BD;
  const uint q_head_stride = qkv_head_strides.x;
  const uint q_seq_stride = qkv_seq_strides.x;
  const uint q_batch_stride = qkv_batch_strides_heads.x;
  const uint k_head_stride = qkv_head_strides.y;
  const uint k_seq_stride = qkv_seq_strides.y;
  const uint k_batch_stride = qkv_batch_strides_heads.y;
  const uint v_head_stride = qkv_head_strides.z;
  const uint v_seq_stride = qkv_seq_strides.z;
  const uint v_batch_stride = qkv_batch_strides_heads.z;
  const uint num_heads = qkv_batch_strides_heads.w;
  const uint mask_kv_seq_stride = mask_strides.x;
  const uint mask_q_seq_stride = mask_strides.y;
  const uint mask_head_stride = mask_strides.z;
  uint inner_k_stride = BN * int(k_seq_stride);
  uint inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int bh_idx = tid.x;
  const int batch_idx = bh_idx / int(num_heads);
  const int head_idx = bh_idx - batch_idx * int(num_heads);
  const int q_seq_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  const int Q = tpg.y;
  const int group_offset = bh_idx * Q + q_seq_idx;
  const int o_offset = group_offset;
  queries +=
      batch_idx * q_batch_stride + head_idx * q_head_stride + q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  values +=
      batch_idx * v_batch_stride + kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * v_per_thread;
  if (has_mask) {
    mask += bh_idx * mask_head_stride + simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (is_causal) {
      use_key = int(i) <= q_seq_idx;
    } else if (has_mask) {
      use_key = mask[0];
    }
    if (use_key) {
      // Read the key
      for (uint j = 0; j < qk_per_thread; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = metal::fast::exp(max_score - new_max);
      U exp_score = metal::fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (has_mask) {
      mask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = metal::fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D, bool is_causal = false>
[[kernel]] void sdpa_vector_2pass_1(const device T* queries [[buffer(0)]],
                                    const device T* keys [[buffer(1)]],
                                    const device T* values [[buffer(2)]],
                                    device T* out [[buffer(3)]],
                                    device float* sums [[buffer(4)]],
                                    device float* maxs [[buffer(5)]],
                                    const constant uint& gqa_factor [[buffer(6)]],
                                    const constant uint& N [[buffer(7)]],
                                    const constant uint3& qkv_head_strides [[buffer(8)]],
                                    const constant uint3& qkv_seq_strides [[buffer(9)]],
                                    const constant float& scale [[buffer(10)]],
                                    const device bool* mask [[buffer(11)]],
                                    const constant uint3& mask_strides [[buffer(12)]],
                                    const constant bool& has_mask [[buffer(13)]],
                                    const constant uint4& qkv_batch_strides_heads [[buffer(14)]],
                                    uint3 tid [[threadgroup_position_in_grid]],
                                    uint3 tpg [[threadgroups_per_grid]],
                                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                                    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  const int q_head_stride = qkv_head_strides.x;
  const int q_seq_stride = qkv_seq_strides.x;
  const int q_batch_stride = qkv_batch_strides_heads.x;
  const int k_head_stride = qkv_head_strides.y;
  const int k_seq_stride = qkv_seq_strides.y;
  const int k_batch_stride = qkv_batch_strides_heads.y;
  const int v_head_stride = qkv_head_strides.z;
  const int v_seq_stride = qkv_seq_strides.z;
  const int v_batch_stride = qkv_batch_strides_heads.z;
  const int num_heads = qkv_batch_strides_heads.w;
  const int mask_kv_seq_stride = mask_strides.x;
  const int mask_q_seq_stride = mask_strides.y;
  const int mask_head_stride = mask_strides.z;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);
  constexpr int blocks = 32;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int block_idx = tid.z;
  const int bh_idx = tid.x;
  const int batch_idx = bh_idx / int(num_heads);
  const int head_idx = bh_idx - batch_idx * int(num_heads);
  const int q_seq_idx = tid.y;
  const int o_offset = bh_idx * tpg.y + q_seq_idx;
  const int kv_head_idx = head_idx / gqa_factor;

  queries +=
      batch_idx * q_batch_stride + head_idx * q_head_stride + q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride + (block_idx * BN + simd_gid) * k_seq_stride +
      simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride + (block_idx * BN + simd_gid) * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask +=
        bh_idx * mask_head_stride + (block_idx * BN + simd_gid) * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    bool use_key = true;
    if (is_causal) {
      use_key = int(i) <= q_seq_idx;
    } else if (has_mask) {
      use_key = mask[0];
    }
    if (use_key) {
      // Read the key
      for (uint j = 0; j < qk_per_thread; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += blocks * inner_k_stride;
    values += blocks * inner_v_stride;
    if (has_mask) {
      mask += BN * blocks * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] = o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (uint j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(const device T* partials [[buffer(0)]],
                                    const device float* sums [[buffer(1)]],
                                    const device float* maxs [[buffer(2)]],
                                    device T* out [[buffer(3)]],
                                    uint3 tid [[threadgroup_position_in_grid]],
                                    uint3 tpg [[threadgroups_per_grid]],
                                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                                    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int hq_offset = head_idx * tpg.y + q_seq_idx;
  partials += hq_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += hq_offset * blocks;
  maxs += hq_offset * blocks;
  out += hq_offset * D + simd_gid * elem_per_thread;

  // First every thread reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (uint i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (uint i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// simdgroup_multiply_accumulate operates on 8x8 sub-tiles
#define SUBTILE_SIZE 8
#define SUBTILE_GRID_M (TILE_ROWS / SUBTILE_SIZE)     // 4
#define SUBTILE_GRID_N (TILE_ROWS / SUBTILE_SIZE)     // 4 (inner-sum subtiles)
#define SUBTILE_GRID_K (TILE_COLS    / SUBTILE_SIZE)     // 8
#define SMEM_FLOATS (TILE_ROWS * TILE_ROWS + TILE_ROWS * TILE_COLS)  // 3072 floats = 12 KB
#define THREADS_PER_ROW (NUM_THREADS / TILE_ROWS);
#define NUMEL_PER_THREAD ((TILE_ROWS * TILE_COLS) / NUM_THREADS)


// Scaled matrix multiplication `r = scale * a @ b`.
// `a` is size (M, N)
// `b` is size (N, K)
// `r` is size (M, K)
//
// For performance, this function uses a tiled matmul algorithm where each
// threadgroup operates on one pair of TILE_SIZExTILE_SIZE tiles of the inputs
// at a time, so that work can be done in `threadgroup` memory, which is faster
// than `device` or `constant` memory.
//
// Each pair of threadgroup tiles is further broken up into 8x8 sub-tiles, whose
// partial results are calculated with `simdgroup_multiply_accumulate`, which is
// a performant way to calculate a matmul between two 8x8 matrices and
// accumulate with previous results.
//
// Note: the pointer type for `a` is templated because both `constant` and
// `device` pointer types need to be supported for this argument.
template <typename T, bool b_transpose>
static void mm_simdgroup(
    device T* r,
    uint32_t r_stride0,
    uint32_t r_stride1,
    device T* a,
    uint32_t a_stride0,
    uint32_t a_stride1,
    device T* b,
    uint32_t b_stride0,
    uint32_t b_stride1,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    float scale,
    uint32_t tile_m,
    uint32_t threadgroup_row,
    uint32_t threadgroup_col,
    uint simdgroup_idx,
    threadgroup float* smem) {
  threadgroup float* tile_a = smem;
  threadgroup float* tile_b = smem + TILE_ROWS * TILE_ROWS;
  // tile_b's region is reused below as the output staging buffer (same shape).

  const uint32_t num_K = (K + TILE_COLS    - 1) / TILE_COLS;
  const uint32_t num_N = (N + TILE_ROWS - 1) / TILE_ROWS;

  // All 32 simdgroups are active, in a SUBTILE_GRID_M x SUBTILE_GRID_K = 4x8
  // grid of 8x8 output sub-tiles.
  const uint32_t subtile_m = simdgroup_idx / SUBTILE_GRID_K;  // 0..3
  const uint32_t subtile_k = simdgroup_idx % SUBTILE_GRID_K;  // 0..7

  for (uint32_t tile_k = 0; tile_k < num_K; tile_k++) {
    simdgroup_float8x8 subtile_r =
        make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    for (uint32_t tile_n = 0; tile_n < num_N; tile_n++) {
      uint32_t a_row = tile_m * TILE_ROWS + threadgroup_row;
      uint32_t a_col = tile_n * TILE_ROWS + threadgroup_col;
      tile_a[threadgroup_row * TILE_ROWS + threadgroup_col] =
          (a_row < M && a_col < N)
          ? float(a[a_row * a_stride0 + a_col * a_stride1])
          : 0.f;

      // Cooperatively fill tile_b (TILE_SIZE x TILE_K -- 2 elements per thread,
      // covering columns threadgroup_col and threadgroup_col + TILE_SIZE).
      uint32_t b_row = tile_n * TILE_ROWS + threadgroup_row;
      for (uint32_t i = 0; i < NUMEL_PER_THREAD; i++) {
        uint32_t b_col_local = threadgroup_col + i * THREADS_PER_ROW;
        uint32_t b_col = tile_k * TILE_COLS + b_col_local;
        uint32_t b_idx = b_transpose
            ? (b_row * b_stride1 + b_col * b_stride0)
            : (b_row * b_stride0 + b_col * b_stride1);
        tile_b[threadgroup_row * TILE_COLS + b_col_local] =
            (b_row < N && b_col < K) ? float(b[b_idx]) : 0.f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // All 32 simdgroups compute their 8x8 output sub-tile.
      for (uint32_t subtile_n = 0; subtile_n < (uint32_t)SUBTILE_GRID_N;
           subtile_n++) {
        simdgroup_float8x8 subtile_a, subtile_b;
        // subtile_a <-- tile_a[subtile_m*8 .. +7][subtile_n*8 .. +7]
        simdgroup_load(
            subtile_a,
            tile_a,
            TILE_ROWS,
            ulong2(subtile_n * SUBTILE_SIZE, subtile_m * SUBTILE_SIZE));
        // subtile_b <-- tile_b[subtile_n*8 .. +7][subtile_k*8 .. +7]
        simdgroup_load(
            subtile_b,
            tile_b,
            TILE_COLS,
            ulong2(subtile_k * SUBTILE_SIZE, subtile_n * SUBTILE_SIZE));
        simdgroup_multiply_accumulate(
            subtile_r, subtile_a, subtile_b, subtile_r);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale, then stage the 8x8 result through tile_b's smem region
    // (which is exactly TILE_SIZE x TILE_K -- the output tile shape).
    if (scale != 1.0f) {
      subtile_r.thread_elements() *= scale;
    }
    simdgroup_store(
        subtile_r,
        tile_b,
        TILE_COLS,
        ulong2(subtile_k * SUBTILE_SIZE, subtile_m * SUBTILE_SIZE));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative write of the TILE_SIZE x TILE_K output tile to r
    // (2 elements per thread).
    uint32_t r_row = tile_m * TILE_ROWS + threadgroup_row;
    for (uint32_t i = 0; i < NUMEL_PER_THREAD; i++) {
      uint32_t r_col_local = threadgroup_col + i * THREADS_PER_ROW;
      uint32_t r_col = tile_k * TILE_COLS + r_col_local;
      if (r_row < M && r_col < K) {
        r[r_row * r_stride0 + r_col * r_stride1] =
            T(tile_b[threadgroup_row * TILE_COLS + r_col_local]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Find the batch and head offset of `mask` only if the mask is enabled for the
// kernel.
struct MaskBatchOffset {
  template <typename T_MASK>
  inline static constant T_MASK* apply(
      constant T_MASK* mask,
      uint32_t stride0,
      uint32_t stride1,
      uint32_t batch_idx,
      uint32_t head_idx) {
    return mask + stride0 * batch_idx + stride1 * head_idx;
  }

  template <>
  inline constant void* apply(
      constant void* mask,
      uint32_t stride0,
      uint32_t stride1,
      uint32_t batch_idx,
      uint32_t head_idx) {
    return mask;
  }
};

// Apply the attention mask if enabled for the kernel. The mask can either be a
// float or a bool. void indicates that the mask is disabled.
struct AttnMask {
  template <typename T_MASK>
  inline static float apply(float value, constant T_MASK* mask, uint32_t idx) {
    auto masked_value = value + mask[idx];
    return ::metal::isnan(masked_value) ? -INFINITY : masked_value;
  }

  template <>
  inline float apply(float value, constant bool* mask, uint32_t idx) {
    return mask[idx] ? value : -INFINITY;
  }

  template <>
  inline float apply(float value, constant void* mask, uint32_t idx) {
    return value;
  }
};

// Apply the causal mask if enabled for the kernel. If enabled, the upper right
// elements are masked out.
struct CausalMask {
  template <bool is_causal, enable_if_t<is_causal, bool> = true>
  inline static float apply(float value, uint32_t row, uint32_t col) {
    return (col <= row) ? value : -INFINITY;
  }

  template <bool is_causal, enable_if_t<!is_causal, bool> = true>
  inline static float apply(float value, uint32_t row, uint32_t col) {
    return value;
  }
};

// Load a value from `attn` and apply masks
template <typename T, typename T_MASK, bool is_causal>
inline float load_attn_value(
    device T* attn,
    uint32_t attn_stride0,
    uint32_t attn_stride1,
    constant T_MASK* mask,
    uint32_t mask_stride0,
    uint32_t mask_stride1,
    uint32_t row,
    uint32_t col) {
  auto attn_idx = row * attn_stride0 + col * attn_stride1;
  auto mask_idx = row * mask_stride0 + col * mask_stride1;
  return CausalMask::apply<is_causal>(
      AttnMask::apply(static_cast<float>(attn[attn_idx]), mask, mask_idx),
      row,
      col);
}

// In-place softmax `attn = softmax(attn, dim=-1)`.
// `attn` is size (L, S)
//
// Within each row of the input, the following steps are performed:
//  1) Find `row_max`, the maximum value in the row
//  2) Find `row_sum`, sum of `exp(value - row_max)` for each value.
//  3) Write normalized `exp(value - row_max) / row_sum` to each value in place.
//
// The `mask` is applied when values are read from `attn`.
//
// For performance, the input is broken up into TILE_SIZE x TILE_SIZE tiles.
// During step 1 and 2, each threadgroup operates on one tile of the input at a
// time, so that the reduction work can be performed in threadgroup memory.
// First, each thread in the threadgroup accumulates the max/sum of the values
// in its assigned position in the tile and writes it into its spot in
// threadgroup memory. Then, a binary reduction is performed on the rows of the
// tile.
template <typename T, typename T_MASK, bool is_causal>
static void softmax_rows(
    device T* attn,
    uint32_t attn_stride0,
    uint32_t attn_stride1,
    constant T_MASK* mask,
    uint32_t mask_stride0,
    uint32_t mask_stride1,
    uint32_t L,
    uint32_t S,
    uint32_t tile_row_idx,
    uint32_t threadgroup_row,
    uint32_t threadgroup_col,
    threadgroup float* smem) {
  const uint32_t row = tile_row_idx * TILE_ROWS + threadgroup_row;
  const bool valid = row < L;

  // Step 1- Find the max value in each row
  float local_max = -INFINITY;
  if (valid) {
    for (uint32_t col = threadgroup_col; col < S; col += TILE_ROWS) {
      float value = load_attn_value<T, T_MASK, is_causal>(
          attn,
          attn_stride0,
          attn_stride1,
          mask,
          mask_stride0,
          mask_stride1,
          row,
          col);
      local_max = max(local_max, value);
    }
  }
  smem[threadgroup_row * TILE_ROWS + threadgroup_col] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduce the partial max values in threadgroup memory
  for (uint32_t stride = TILE_ROWS / 2; stride > 0; stride >>= 1) {
    if (threadgroup_col < stride)
      smem[threadgroup_row * TILE_ROWS + threadgroup_col] =
          max(smem[threadgroup_row * TILE_ROWS + threadgroup_col],
              smem[threadgroup_row * TILE_ROWS + threadgroup_col + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float row_max = smem[threadgroup_row * TILE_ROWS];

  // Step 2 - Find the exp sum in each row
  float local_sum = 0.f;
  if (valid) {
    for (uint32_t col = threadgroup_col; col < S; col += TILE_ROWS) {
      float value = load_attn_value<T, T_MASK, is_causal>(
          attn,
          attn_stride0,
          attn_stride1,
          mask,
          mask_stride0,
          mask_stride1,
          row,
          col);
      float e = precise::exp(value - row_max);
      local_sum += ::metal::isnan(e) ? 0 : e;
    }
  }
  smem[threadgroup_row * TILE_ROWS + threadgroup_col] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reduce the partial sum values in threadgroup memory
  for (uint32_t stride = TILE_ROWS / 2; stride > 0; stride >>= 1) {
    if (threadgroup_col < stride)
      smem[threadgroup_row * TILE_ROWS + threadgroup_col] +=
          smem[threadgroup_row * TILE_ROWS + threadgroup_col + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float row_sum = smem[threadgroup_row * TILE_ROWS];

  // Step 3 - Normalize in place
  if (valid) {
    for (uint32_t col = threadgroup_col; col < S; col += TILE_ROWS) {
      float value = load_attn_value<T, T_MASK, is_causal>(
          attn,
          attn_stride0,
          attn_stride1,
          mask,
          mask_stride0,
          mask_stride1,
          row,
          col);
      float e = precise::exp(value - row_max);
      auto attn_idx = row * attn_stride0 + col * attn_stride1;
      attn[attn_idx] = row_sum == 0 ? 0 : T(e / row_sum);
    }
  }
}

template <typename T, typename T_MASK, bool is_causal>
kernel void sdpa_general(
    device T* out [[buffer(0)]],
    device T* attn [[buffer(1)]],
    device T* q [[buffer(2)]],
    device T* k [[buffer(3)]],
    device T* v [[buffer(4)]],
    constant T_MASK* mask [[buffer(5)]],
    constant SDPAParams<>& params [[buffer(6)]],
    uint3 lid [[thread_position_in_threadgroup]], // (x=col, y=row)
    uint3 tgid [[threadgroup_position_in_grid]], // one group per (batch, head)
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]] // 0..31
) {
  threadgroup float smem[SMEM_FLOATS];

  // Find the batch and head offsets of each of the inputs and outputs
  const uint32_t head_idx = tgid.y;
  const uint32_t kv_head_idx = head_idx / params.gqa_factor;
  const uint32_t batch_idx = tgid.x;
  const uint32_t tile_m = tgid.z;

  out += params.out_strides[0] * batch_idx + params.out_strides[1] * head_idx;
  attn +=
      params.attn_strides[0] * batch_idx + params.attn_strides[1] * head_idx;
  q += params.q_strides[0] * batch_idx + params.q_strides[1] * head_idx;
  k += params.k_strides[0] * batch_idx + params.k_strides[1] * kv_head_idx;
  v += params.v_strides[0] * batch_idx + params.v_strides[1] * kv_head_idx;
  mask = MaskBatchOffset::apply(
      mask,
      params.mask_strides[0],
      params.mask_strides[1],
      batch_idx,
      head_idx);

  const uint32_t threadgroup_row = lid.x / THREADS_PER_ROW;
  const uint32_t threadgroup_col = lid.x % THREADS_PER_ROW;

  // Matmul `q`, size (L, E), and `k^T`, size (E, S), then multiply by `scale`,
  // and write output to `attn`.
  mm_simdgroup<T, true>(
      attn,
      params.attn_strides[2],
      params.attn_strides[3],
      q,
      params.q_strides[2],
      params.q_strides[3],
      k,
      params.k_strides[2],
      params.k_strides[3],
      params.L,
      params.E,
      params.S,
      params.scale,
      tile_m,
      threadgroup_row,
      threadgroup_col,
      simdgroup_idx,
      smem);

  threadgroup_barrier(mem_flags::mem_device);

  // Perform softmax to `attn` in-place.
  softmax_rows<T, T_MASK, is_causal>(
      attn,
      params.attn_strides[2],
      params.attn_strides[3],
      mask,
      params.mask_strides[2],
      params.mask_strides[3],
      params.L,
      params.S,
      tile_m,
      threadgroup_row,
      threadgroup_col,
      smem);

  threadgroup_barrier(mem_flags::mem_device);

  // Matmul `attn`, size (L, S), and `v`, size (S, Ev), and write output to
  // `out`.
  mm_simdgroup<T, false>(
      out,
      params.out_strides[2],
      params.out_strides[3],
      attn,
      params.attn_strides[2],
      params.attn_strides[3],
      v,
      params.v_strides[2],
      params.v_strides[3],
      params.L,
      params.S,
      params.Ev,
      /*scale=*/1.f,
      tile_m,
      threadgroup_row,
      threadgroup_col,
      simdgroup_idx,
      smem);
}

#define INSTANTIATE_SDPA_VECTOR_ONE(DTYPE, QK_DIM, VALUE_DIM, CAUSAL, NAME_SUFFIX)                            \
  template[[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM "_" #VALUE_DIM NAME_SUFFIX)]] kernel void             \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM, CAUSAL>(const device DTYPE* queries [[buffer(0)]],                    \
                                                const device DTYPE* keys [[buffer(1)]],                       \
                                                const device DTYPE* values [[buffer(2)]],                     \
                                                device DTYPE* out [[buffer(3)]],                              \
                                                const constant uint& gqa_factor [[buffer(4)]],                \
                                                const constant uint& N [[buffer(5)]],                         \
                                                const constant uint3& qkv_head_strides [[buffer(6)]],         \
                                                const constant uint3& qkv_seq_strides [[buffer(7)]],          \
                                                const constant float& scale [[buffer(8)]],                    \
                                                const device bool* mask [[buffer(9)]],                        \
                                                const constant uint3& mask_strides [[buffer(10)]],            \
                                                const constant bool& has_mask [[buffer(11)]],                 \
                                                const constant uint4& qkv_batch_strides_heads [[buffer(12)]], \
                                                uint3 tid [[threadgroup_position_in_grid]],                   \
                                                uint3 tpg [[threadgroups_per_grid]],                          \
                                                uint simd_gid [[simdgroup_index_in_threadgroup]],             \
                                                uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)           \
  INSTANTIATE_SDPA_VECTOR_ONE(DTYPE, QK_DIM, VALUE_DIM, false, ""); \
  INSTANTIATE_SDPA_VECTOR_ONE(DTYPE, QK_DIM, VALUE_DIM, true, "_causal");

#define INSTANTIATE_SDPA_VECTOR_2PASS_1_ONE(DTYPE, QK_DIM, VALUE_DIM, CAUSAL, NAME_SUFFIX)                            \
  template[[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM "_" #VALUE_DIM NAME_SUFFIX)]] kernel void             \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM, CAUSAL>(const device DTYPE* queries [[buffer(0)]],                    \
                                                        const device DTYPE* keys [[buffer(1)]],                       \
                                                        const device DTYPE* values [[buffer(2)]],                     \
                                                        device DTYPE* out [[buffer(3)]],                              \
                                                        device float* sums [[buffer(4)]],                             \
                                                        device float* maxs [[buffer(5)]],                             \
                                                        const constant uint& gqa_factor [[buffer(6)]],                \
                                                        const constant uint& N [[buffer(7)]],                         \
                                                        const constant uint3& qkv_head_strides [[buffer(8)]],         \
                                                        const constant uint3& qkv_seq_strides [[buffer(9)]],          \
                                                        const constant float& scale [[buffer(10)]],                   \
                                                        const device bool* mask [[buffer(11)]],                       \
                                                        const constant uint3& mask_strides [[buffer(12)]],            \
                                                        const constant bool& has_mask [[buffer(13)]],                 \
                                                        const constant uint4& qkv_batch_strides_heads [[buffer(14)]], \
                                                        uint3 tid [[threadgroup_position_in_grid]],                   \
                                                        uint3 tpg [[threadgroups_per_grid]],                          \
                                                        uint simd_gid [[simdgroup_index_in_threadgroup]],             \
                                                        uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM)           \
  INSTANTIATE_SDPA_VECTOR_2PASS_1_ONE(DTYPE, QK_DIM, VALUE_DIM, false, ""); \
  INSTANTIATE_SDPA_VECTOR_2PASS_1_ONE(DTYPE, QK_DIM, VALUE_DIM, true, "_causal");

#define INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, VALUE_DIM)                             \
  template [[host_name("sdpa_vector_2pass_2_" #DTYPE "_" #VALUE_DIM)]] kernel void        \
  sdpa_vector_2pass_2<DTYPE, VALUE_DIM>(const device DTYPE* partials [[buffer(0)]],       \
                                        const device float* sums [[buffer(1)]],           \
                                        const device float* maxs [[buffer(2)]],           \
                                        device DTYPE* out [[buffer(3)]],                  \
                                        uint3 tid [[threadgroup_position_in_grid]],       \
                                        uint3 tpg [[threadgroups_per_grid]],              \
                                        uint simd_gid [[simdgroup_index_in_threadgroup]], \
                                        uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 128, 128);         \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 256, 256);         \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 64, 64);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 96, 96);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 128, 128); \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 256, 256); \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 64);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 96);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 128);  \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 256);

INSTANTIATE_SDPA_VECTOR_HEADS(float);
INSTANTIATE_SDPA_VECTOR_HEADS(half);
INSTANTIATE_SDPA_VECTOR_HEADS(bfloat);

#define CAUSAL_SUFFIX_true "_causal"
#define CAUSAL_SUFFIX_false ""

#define REGISTER_SDPA_GENERAL(T, T_MASK, IS_CAUSAL)                        \
  template[[host_name("sdpa_general_" #T                                   \
                      "_" #T_MASK CAUSAL_SUFFIX_##IS_CAUSAL)]] kernel void \
  sdpa_general<T, T_MASK, IS_CAUSAL>(                                      \
      device T * out [[buffer(0)]],                                        \
      device T * attn [[buffer(1)]],                                       \
      device T * q [[buffer(2)]],                                          \
      device T * k [[buffer(3)]],                                          \
      device T * v [[buffer(4)]],                                          \
      constant T_MASK * mask [[buffer(5)]],                                \
      constant SDPAParams<> & params [[buffer(6)]],                        \
      uint3 lid [[thread_position_in_threadgroup]],                        \
      uint3 tgid [[threadgroup_position_in_grid]],                         \
      uint simdgroup_idx [[simdgroup_index_in_threadgroup]]);

#define REGISTER_SDPA_GENERAL_MASK_TYPES(T) \
  REGISTER_SDPA_GENERAL(T, void, false);    \
  REGISTER_SDPA_GENERAL(T, void, true);     \
  REGISTER_SDPA_GENERAL(T, bool, false);    \
  REGISTER_SDPA_GENERAL(T, T, false);

REGISTER_SDPA_GENERAL_MASK_TYPES(float);
REGISTER_SDPA_GENERAL_MASK_TYPES(half);
REGISTER_SDPA_GENERAL_MASK_TYPES(bfloat);
