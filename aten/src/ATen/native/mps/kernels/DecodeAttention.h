// SDPA decode kernels (short Q sequence). Three variants:
//   sdpa_vector            - one-pass, for short qL with moderate kL
//   sdpa_vector_2pass_1    - two-pass pass 1, splits the K loop across blocks
//   sdpa_vector_2pass_2    - two-pass pass 2, aggregates per-block partials
//
// Adapted from MLX:
//   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#pragma once

#include <ATen/native/mps/kernels/Attention.h>

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
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
  const uint mask_head_stride = mask_strides.x;
  const uint mask_kv_seq_stride = mask_strides.y;
  const uint mask_q_seq_stride = mask_strides.z;
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
  queries += batch_idx * q_batch_stride + head_idx * q_head_stride +
      q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride +
      simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride +
      simd_gid * v_seq_stride + simd_lid * v_per_thread;
  if (has_mask) {
    mask += bh_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
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
    if (!has_mask || mask[0]) {
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

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
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

  queries += batch_idx * q_batch_stride + head_idx * q_head_stride +
      q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride +
      (block_idx * BN + simd_gid) * k_seq_stride + simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride +
      (block_idx * BN + simd_gid) * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask += bh_idx * mask_head_stride +
        (block_idx * BN + simd_gid) * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
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
    if (!has_mask || mask[0]) {
      // Read the key
      for (uint i = 0; i < qk_per_thread; i++) {
        k[i] = static_cast<U>(keys[i]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint i = 0; i < qk_per_thread; i++) {
        score += q[i] * k[i];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * static_cast<U>(values[i]);
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
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores[simd_gid] - new_max);
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
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
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
  partials +=
      hq_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
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

// Matrix `mat` of size (N x d) is broken up into blocks of size (B x d). This
// function loads the `i`-th block into threadgroup memory.
template <typename ptr_t>
static inline void load_matrix_block(threadgroup float* block, ptr_t mat, uint32_t mat_stride0, uint32_t mat_stride1, uint32_t N, uint32_t B, uint32_t i, uint32_t d, uint lid, uint tptg) {
  for (uint32_t block_offset = lid; block_offset < B * d; block_offset += tptg) {
    uint32_t block_row = block_offset / d;
    uint32_t col = block_offset % d;
    auto mat_row = i * B + block_row;
    auto mat_offset = mat_row * mat_stride0 + col * mat_stride1;
    float val = (mat_row < N) ?  static_cast<float>(mat[mat_offset]) : 0;
    block[block_offset] = val;
  }
}

template <typename T>
static inline void write_matrix_block(threadgroup float* block, device T* mat, uint32_t mat_stride0, uint32_t mat_stride1, uint32_t N, uint32_t B, uint32_t i, uint32_t d, uint lid, uint tptg) {
  for (uint32_t block_offset = lid; block_offset < B * d; block_offset += tptg) {
    uint32_t block_row = block_offset / d;
    uint32_t col = block_offset % d;
    auto mat_row = i * B + block_row;
    auto mat_offset = mat_row * mat_stride0 + col * mat_stride1;
    if (mat_row < N) {
      mat[mat_offset] = static_cast<T>(block[block_offset]);
    }
  }
}

static inline void load_vector_block(threadgroup float* vec_i, device float* vec, uint32_t i, uint32_t B, uint lid, uint tptg) {
  for (uint idx = lid; idx < B; idx += tptg) {
    vec_i[idx] = vec[i * B + idx];
  }
}

static inline void write_vector_block(threadgroup float* vec_i, device float* vec, uint32_t i, uint32_t B, uint lid, uint tptg) {
  for (uint idx = lid; idx < B; idx += tptg) {
    vec[i * B + idx] = vec_i[idx];
  }
}


// Compute `R = scale * A @ B`, where A is shape (M x N), B is shape (N x K),
// and R is shape (M x K). Matrices are assumed to be contiguous row-major. If
// `B_transpose` is true, then B is actually shape (K x N), still contiguous
// row-major, and `R = scale * A @ B^T` is computed.
//
// Fast path requires M, N, K all multiples of 8.
template <bool B_transpose>
static void matmul_simd2(threadgroup float* R,
                         threadgroup const float* A,
                         threadgroup const float* B,
                         uint32_t M, uint32_t N, uint32_t K,
                         uint simd_lane_id,
                         uint simd_group_id,
                         uint simd_groups,
                         float scale = 1.0f) {
  const uint32_t M_tiles = M / 8;
  const uint32_t K_tiles = K / 8;
  const uint32_t N_tiles = N / 8;
  const uint32_t total_tiles = M_tiles * K_tiles;

  // Each SIMD group owns one 8x8 output tile, sweeping N in steps of 8.
  for (uint32_t tile_idx = simd_group_id;
       tile_idx < total_tiles;
       tile_idx += simd_groups) {
    const uint32_t tile_row = tile_idx / K_tiles;  // along M
    const uint32_t tile_col = tile_idx % K_tiles;  // along K

    // Register-resident accumulator. make_filled_simdgroup_matrix zeros it.
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // Walk the K (inner) dimension in 8-wide chunks.
    for (uint32_t n_tile = 0; n_tile < N_tiles; ++n_tile) {
      simdgroup_float8x8 a_frag, b_frag;

      // A tile at (tile_row, n_tile): top-left at A[tile_row*8, n_tile*8].
      // A is row-major with stride N.
      simdgroup_load(a_frag,
                     A,
                     /* stride */ N,
                     /* origin */ ulong2(n_tile * 8, tile_row * 8),
                     /* transpose */ false);

      // B tile at (n_tile, tile_col).
      // If !B_transpose: B is (N x K) row-major, stride K, top-left at
      //   B[n_tile*8, tile_col*8]. No transpose on load.
      // If  B_transpose: B is (K x N) row-major, stride N, top-left at
      //   B[tile_col*8, n_tile*8]. Load with transpose=true so the fragment
      //   represents the (n_tile, tile_col) tile of B^T.
      if (B_transpose) {
        simdgroup_load(b_frag,
                       B,
                       /* stride */ N,
                       /* origin */ ulong2(n_tile * 8, tile_col * 8),
                       /* transpose */ true);
      } else {
        simdgroup_load(b_frag,
                       B,
                       /* stride */ K,
                       /* origin */ ulong2(tile_col * 8, n_tile * 8),
                       /* transpose */ false);
      }

      // acc += a_frag @ b_frag, in hardware.
      simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
    }

    // Fold scale into the accumulator before storing. Each lane scales the
    // two elements it owns; no barrier or coordination needed.
    if (scale != 1.0f) {
      acc.thread_elements() *= scale;
    }

    // Store the tile to R (M x K, stride K), top-left at (tile_row*8, tile_col*8).
    simdgroup_store(acc,
                    R,
                    /* stride */ K,
                    /* origin */ ulong2(tile_col * 8, tile_row * 8),
                    /* transpose */ false);
  }
}

// Compute `R = scale * A @ B`, where A is shape (M x N), B is shape (N x K),
// and R is shape (M x K). Matrices are assumed to be contiguous row-major. If
// `B_transpose` is true, then B is actually shape (K x N), still contiguous
// row-major, and `R = scale * A @ B^T` is computed.
static void matmul(threadgroup float* R, threadgroup float* A, threadgroup float* B, uint32_t M, uint32_t N, uint32_t K, bool B_transpose, bool accumulate, uint lid, uint tptg, float scale=1) {
  // Divide the output elements up between each thread.
  for (uint32_t R_idx = lid; R_idx < (M * K); R_idx += tptg) {
    uint32_t RB_col = R_idx % K;
    uint32_t RA_row = R_idx / K;

    float dot_prod = 0;

    // Compute dot product of `dot(A[RA_row, :], B[:, RB_col]`
    for (uint32_t dot_idx = 0; dot_idx < N; dot_idx++) {
      float a = A[RA_row * N + dot_idx];
      float b = B[B_transpose ? (RB_col * N + dot_idx) : (dot_idx * K + RB_col)];
      dot_prod += a * b;
    }

    auto res = scale * dot_prod;
    auto R_offset = RA_row * K + RB_col;

    if (accumulate) {
      R[R_offset] += res;

    } else {
      R[R_offset] = res;
    }
  }
}

// Compute `R = scale * A @ B` (or `R += ...` if accumulate), where A is
// logically (M x N), B is (N x K), R is (M x K). All buffers must be
// allocated with dimensions rounded up to multiples of 8:
//   M_pad = ceil(M/8)*8, N_pad = ceil(N/8)*8, K_pad = ceil(K/8)*8
// with the slack rows/cols of A and B zero-filled (R's slack is don't-care
// going in; on output its slack region is set to 0 if !accumulate, or left
// equal to its prior value if accumulate).
//
// Row strides are the *padded* dims: A stride N_pad, B stride K_pad
// (or N_pad when B_transpose), R stride K_pad. If B_transpose, B is stored
// as (K_pad x N_pad) row-major.
template <bool B_transpose, bool accumulate>
static void matmul_simd(threadgroup float* R, threadgroup float* A, threadgroup float* B,
                   uint32_t M, uint32_t N, uint32_t K,
                   uint simd_lid, uint sg_id, uint nsg,
                   float scale = 1) {
  uint32_t M_pad = (M + 7) & ~7u;
  uint32_t N_pad = (N + 7) & ~7u;
  uint32_t K_pad = (K + 7) & ~7u;

  uint32_t M_tiles = M_pad / 8;
  uint32_t K_tiles = K_pad / 8;
  uint32_t N_tiles = N_pad / 8;
  uint32_t total_tiles = M_tiles * K_tiles;

  for (uint32_t tile_idx = sg_id; tile_idx < total_tiles; tile_idx += nsg) {
    uint32_t r_off = (tile_idx / K_tiles) * 8;
    uint32_t c_off = (tile_idx % K_tiles) * 8;
    threadgroup float* R_tile = R + r_off * K_pad + c_off;
    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    for (uint32_t k_tile = 0; k_tile < N_tiles; k_tile++) {
      uint32_t k_off = k_tile * 8;
      simdgroup_float8x8 a_mat, b_mat;
      simdgroup_load(a_mat, A + r_off * N_pad + k_off, N_pad);
      if (B_transpose) {
        simdgroup_load(b_mat, B + c_off * N_pad + k_off, N_pad, ulong2(0, 0), true);
      } else {
        simdgroup_load(b_mat, B + k_off * K_pad + c_off, K_pad);
      }
      simdgroup_multiply_accumulate(acc, a_mat, b_mat, acc);
    }

    if (scale != 1.0f) {
      acc.thread_elements() *= scale;
    }

    if (accumulate) {
      simdgroup_float8x8 r_old;
      simdgroup_load(r_old, R_tile, K_pad);
      acc.thread_elements() += r_old.thread_elements();
    }

    simdgroup_store(acc, R_tile, K_pad);
  }
}

// This kernel implements the forward pass of the FlashAttention-2 algorithm
// (https://arxiv.org/abs/2307.08691). Each of the output matrices, size N x d,
// is broken up into blocks of size B_r x d, and each threadgroup is responsible
// for calculating one block of one output matrix.
template <typename T, typename T_MASK, bool is_causal>
kernel void sdpa_flash(
    device T* out [[buffer(0)]],
    device float* logsumexp [[buffer(1)]],
    constant T* Q [[buffer(2)]],
    constant T* K [[buffer(3)]],
    constant T* V [[buffer(4)]],
    constant T_MASK* mask [[buffer(5)]],
    constant SDPAParams<>& params [[buffer(6)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tptg [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]
) {
  auto B_r = params.B_r;
  auto B_c = params.B_c;
  auto N = params.N;
  auto d = params.d;

  // Find the batch and head offsets of each of the inputs and outputs
  const uint32_t head_idx = tgid.x % params.num_heads;
  const uint32_t batch_idx = tgid.x / params.num_heads;
  out += params.out_strides[0] * batch_idx + params.out_strides[1] * head_idx;
  logsumexp += params.logsumexp_strides[0] * batch_idx + params.logsumexp_strides[1] * head_idx;
  Q += params.q_strides[0] * batch_idx + params.q_strides[1] * head_idx;
  K += params.k_strides[0] * batch_idx + params.k_strides[1] * head_idx;
  V += params.v_strides[0] * batch_idx + params.v_strides[1] * head_idx;

  auto KV_j_size = B_c * d;
  auto Q_i_size = B_r * d;
  auto O_i_size = B_r * d;
  auto S_ij_size = B_r * B_c;
  auto l_i_size = B_r;
  auto m_i_size = B_r;

  threadgroup float smem[THREADGROUP_MEMORY_FLOATS];
  threadgroup float* KV_j = smem;
  threadgroup float* Q_i = KV_j + KV_j_size;
  threadgroup float* O_i = Q_i + Q_i_size;
  threadgroup float* S_ij = O_i + O_i_size;
  threadgroup float* l_i = S_ij + S_ij_size;
  threadgroup float* m_i = l_i + l_i_size;
  threadgroup float* m_i_prev = m_i + m_i_size;

  uint32_t i = tgid.y;
  uint32_t T_c = (N + B_c - 1) / B_c;

  load_matrix_block(Q_i, Q, params.q_strides[2], params.q_strides[3], N, B_r, i, d, lid.x, tptg.x);

  for (uint32_t j = 0; j < T_c; j++) {
    load_matrix_block(KV_j, K, params.k_strides[2], params.k_strides[3], N, B_c, j, d, lid.x, tptg.x);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    //matmul(S_ij, Q_i, KV_j, B_r, d, B_c, true, false, lid.x, tptg.x, params.scale);
    matmul_simd</*B_transpose=*/true, /*accumulate=*/false>(
      S_ij, Q_i, KV_j, B_r, d, B_c,
      simd_lane_id, simd_group_id, simd_groups,
      params.scale);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    load_matrix_block(KV_j, V, params.v_strides[2], params.v_strides[3], N, B_c, j, d, lid.x, tptg.x);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    //matmul(O_i, S_ij, KV_j, B_r, B_c, d, false, true, lid.x, tptg.x, 1);
    matmul_simd</*B_transpose=*/false, /*accumulate=*/true>(
      O_i, S_ij, KV_j,
      B_r, B_c, d,
      simd_lane_id, simd_group_id, simd_groups,
      /*scale=*/1);

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  write_matrix_block(O_i, out, params.out_strides[2], params.out_strides[3], N, B_r, i, d, lid.x, tptg.x);
  write_vector_block(l_i, logsumexp, i, B_r, lid.x, tptg.x);
}

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)           \
  template [[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM            \
                       "_" #VALUE_DIM)]] kernel void                \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM>(                            \
      const device DTYPE* queries [[buffer(0)]],                    \
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

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM)   \
  template [[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM    \
                       "_" #VALUE_DIM)]] kernel void                \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM>(                    \
      const device DTYPE* queries [[buffer(0)]],                    \
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

#define INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, VALUE_DIM)                 \
  template                                                                    \
      [[host_name("sdpa_vector_2pass_2_" #DTYPE "_" #VALUE_DIM)]] kernel void \
      sdpa_vector_2pass_2<DTYPE, VALUE_DIM>(                                  \
          const device DTYPE* partials [[buffer(0)]],                         \
          const device float* sums [[buffer(1)]],                             \
          const device float* maxs [[buffer(2)]],                             \
          device DTYPE* out [[buffer(3)]],                                    \
          uint3 tid [[threadgroup_position_in_grid]],                         \
          uint3 tpg [[threadgroups_per_grid]],                                \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
          uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 128, 128);         \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 64, 64);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 96, 96);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 128, 128); \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 64);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 96);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 128);

INSTANTIATE_SDPA_VECTOR_HEADS(float);
INSTANTIATE_SDPA_VECTOR_HEADS(half);
INSTANTIATE_SDPA_VECTOR_HEADS(bfloat);

#define CAUSAL_SUFFIX_true "_causal"
#define CAUSAL_SUFFIX_false ""

#define REGISTER_SDPA_FLASH(T, T_MASK, IS_CAUSAL)                               \
  template[[host_name("sdpa_flash_" #T "_" #T_MASK CAUSAL_SUFFIX_##IS_CAUSAL)]] \
  kernel void sdpa_flash<T, T_MASK, IS_CAUSAL>(                                 \
      device T * out [[buffer(0)]],                                             \
      device float * logsumexp [[buffer(1)]],                                   \
      constant T * Q [[buffer(2)]],                                             \
      constant T * K [[buffer(3)]],                                             \
      constant T * V [[buffer(4)]],                                             \
      constant T_MASK * mask [[buffer(5)]],                                     \
      constant SDPAParams<> & params [[buffer(6)]],                             \
      uint2 lid [[thread_position_in_threadgroup]],                             \
      uint2 tgid [[threadgroup_position_in_grid]],                              \
      uint2 tptg [[threads_per_threadgroup]],                                   \
      uint simd_lane_id [[thread_index_in_simdgroup]],                          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                    \
      uint simd_groups [[simdgroups_per_threadgroup]]);

#define REGISTER_SDPA_FLASH_MASK_TYPES(T) \
  REGISTER_SDPA_FLASH(T, void, false);    \
  REGISTER_SDPA_FLASH(T, void, true);     \
  REGISTER_SDPA_FLASH(T, bool, false);    \
  REGISTER_SDPA_FLASH(T, T, false);

REGISTER_SDPA_FLASH_MASK_TYPES(float);
REGISTER_SDPA_FLASH_MASK_TYPES(half);
REGISTER_SDPA_FLASH_MASK_TYPES(bfloat);
