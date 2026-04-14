// Largely influeneced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#include <ATen/native/mps/kernels/Attention.h>
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

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
  const uint k_head_stride = qkv_head_strides.y;
  const uint k_seq_stride = qkv_seq_strides.y;
  const uint v_head_stride = qkv_head_strides.z;
  const uint v_seq_stride = qkv_seq_strides.z;
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

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  const int Q = tpg.y;
  const int group_offset = head_idx * Q + q_seq_idx;
  const int o_offset = group_offset;
  queries += head_idx * q_head_stride + q_seq_idx * q_seq_stride +
      simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
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
  const int k_head_stride = qkv_head_strides.y;
  const int k_seq_stride = qkv_seq_strides.y;
  const int v_head_stride = qkv_head_strides.z;
  const int v_seq_stride = qkv_seq_strides.z;
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

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int o_offset = head_idx * tpg.y + q_seq_idx;
  const int kv_head_idx = head_idx / gqa_factor;

  queries += head_idx * q_head_stride + q_seq_idx * q_seq_stride +
      simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride +
      (block_idx * BN + simd_gid) * k_seq_stride + simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride +
      (block_idx * BN + simd_gid) * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride +
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

template <typename T, int BQ, int BK, int BD, int WM, int WN>
kernel void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant uint& qL [[buffer(4)]],
    const constant uint& kL [[buffer(5)]],
    const constant uint& gqa_factor [[buffer(6)]],
    const constant float& scale [[buffer(7)]],
    const constant uint& NK [[buffer(8)]],
    const constant uint3& Q_strides [[buffer(9)]],
    const constant uint3& K_strides [[buffer(10)]],
    const constant uint3& V_strides [[buffer(11)]],
    const constant uint3& O_strides [[buffer(12)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]]) {
  // 1. Compute a full linear thread id from the 3D local id.
  constexpr int THREADGROUP_DIM_X = 32;
  constexpr int THREADGROUP_DIM_Y = WM;
  constexpr int THREADGROUP_DIM_Z = WN;
  const int threads_in_group =
      THREADGROUP_DIM_X * THREADGROUP_DIM_Y * THREADGROUP_DIM_Z;
  int tid = local_pos.x + local_pos.y * THREADGROUP_DIM_X +
      local_pos.z * (THREADGROUP_DIM_X * THREADGROUP_DIM_Y);

  // 2. Compute the effective number of Q (query) rows for this tile.
  const int query_seq_length = qL;
  int start_q = group_pos.x * BQ;
  uint tile_rows =
      (start_q + BQ <= query_seq_length) ? BQ : (query_seq_length - start_q);

  // 3. Compute Global Pointers Offsets for Q and O.
  uint batch = group_pos.z;
  uint head = group_pos.y;
  uint seq_tile = group_pos.x;

  const device T* Q_tile_ptr = Q + batch * Q_strides.x + head * Q_strides.y +
      seq_tile * BQ * Q_strides.z;
  device T* O_tile_ptr = O + batch * O_strides.x + head * O_strides.y +
      seq_tile * BQ * O_strides.z;

  // Adjust head index for K and V using gqa_factor.
  uint kv_head = head / gqa_factor;
  const device T* K_ptr = K + batch * K_strides.x + kv_head * K_strides.y;
  const device T* V_ptr = V + batch * V_strides.x + kv_head * V_strides.y;

  // 4. Declare Threadgroup (Shared) Memory for tiles.
  // qTile covers BQ rows (each of length BD), kTile and vTile cover BK rows.
  threadgroup T qTile[BQ * BD];
  threadgroup T kTile[BK * BD];
  threadgroup T vTile[BK * BD];

  // 5. Load Q from global memory into threadgroup memory & apply scaling.
  uint tile_q_elements = tile_rows * BD;
  for (uint i = tid; i < tile_q_elements; i += threads_in_group) {
    int row = i / BD;
    int col = i % BD;
    qTile[i] = Q_tile_ptr[row * Q_strides.z + col] * (T)scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 6. Initialize accumulation buffers for output and softmax reduction.
  float oAcc[BQ * BD]; // Only first tile_q_elements are used
  float row_max[BQ]; // For each valid query row
  float row_sum[BQ]; // For each valid query row
  for (uint i = 0; i < tile_rows; i++) {
    row_max[i] = -FLT_MAX;
    row_sum[i] = 0.0f;
  }
  for (uint i = 0; i < tile_q_elements; i++) {
    oAcc[i] = 0.0f;
  }

  // 7. Loop over the Key/Value (KV) sequence tiles.
  for (uint kb_tile = 0; kb_tile < NK; ++kb_tile) {
    uint kv_base = kb_tile * BK; // first KV row in this tile
    uint total_kv_elements = BK * BD;

    // --- Load K and V tiles into threadgroup memory.
    // For positions that are out-of-bound (padded) set K to -INFINITY.
    for (uint i = tid; i < total_kv_elements; i += threads_in_group) {
      int row = i / BD;
      int col = i % BD;
      if ((kv_base + row) < kL) {
        kTile[i] = K_ptr[(kv_base + row) * K_strides.z + col];
        vTile[i] = V_ptr[(kv_base + row) * V_strides.z + col];
      } else {
        // For invalid keys, assign a very negative value so that exp(-inf)=0
        kTile[i] = static_cast<T>(-INFINITY);
        vTile[i] = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8. Compute the score matrix S = Q x (K)^T for this KV tile.
    float S[BQ * BK];
    for (uint i = 0; i < tile_rows; i++) {
      for (int j = 0; j < BK; j++) {
        float dot = 0.0f;
        // Only compute dot product if this tile row corresponds to a valid key.
        if ((kv_base + j) < kL) {
          for (int d = 0; d < BD; d++) {
            dot += qTile[i * BD + d] * kTile[j * BD + d];
          }
        } else {
          dot = -INFINITY;
        }
        S[i * BK + j] = dot;
      }
    }

    // 9. Update softmax statistics (row-wise) using an online reduction.
    for (uint i = 0; i < tile_rows; i++) {
      float old_max = row_max[i];
      float new_max = old_max;
      for (int j = 0; j < BK; j++) {
        float val = S[i * BK + j];
        if (val > new_max) {
          new_max = val;
        }
      }
      float factor = exp(old_max - new_max);
      row_max[i] = new_max;
      // Scale the accumulated numerator for this row.
      for (int d = 0; d < BD; d++) {
        oAcc[i * BD + d] *= factor;
      }
      // Exponentiate the scores and accumulate the sums.
      float exp_sum = 0.0f;
      for (int j = 0; j < BK; j++) {
        float s_val = exp(S[i * BK + j] - new_max);
        S[i * BK + j] = s_val;
        exp_sum += s_val;
      }
      row_sum[i] = row_sum[i] * factor + exp_sum;
    }

    // 10. Use the softmax weights to compute the weighted sum of V.
    for (uint i = 0; i < tile_rows; i++) {
      for (int d = 0; d < BD; d++) {
        float acc = 0.0f;
        for (int j = 0; j < BK; j++) {
          acc += S[i * BK + j] * vTile[j * BD + d];
        }
        oAcc[i * BD + d] += acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } // End of KV tile loop

  // 11. Normalize the accumulated output and store the results to global
  // memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_pos.x == 0 && local_pos.y == 0 && local_pos.z == 0) {
    for (uint i = 0; i < tile_rows; i++) {
      for (int d = 0; d < BD; d++) {
        O_tile_ptr[i * O_strides.z + d] =
            static_cast<T>(oAcc[i * BD + d] / row_sum[i]);
      }
    }
  }
}

// Multiply two matrices `a` (shape M x N) and `b` (shape N x K). Multiply result
// by scalar `alpha` and write into matrix `r` (shape M x K).
template <typename T, typename A_ptr, typename op_T = c10::metal::opmath_t<T>>
static inline void mm(
    device T* r,
    uint32_t r_stride0,
    uint32_t r_stride1,
    A_ptr a,
    uint32_t a_stride0,
    uint32_t a_stride1,
    constant T* b,
    uint32_t b_stride0,
    uint32_t b_stride1,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    op_T alpha,
    uint tid,
    uint tptg
) {
  uint32_t MK = M * K;

  for (uint32_t elem_idx = tid; elem_idx < MK; elem_idx += tptg) {
    uint32_t row = elem_idx / K;
    uint32_t col = elem_idx % K;

    op_T acc = 0;

    uint32_t a_base = row * a_stride0;
    uint32_t b_base = col * b_stride1;

    uint32_t a_idx = a_base;
    uint32_t b_idx = b_base;

    uint32_t n = 0;

    for (; n + 3 < N; n += 4) {
      op_T a0 = static_cast<op_T>(a[a_idx]);
      op_T b0 = static_cast<op_T>(b[b_idx]);

      op_T a1 = static_cast<op_T>(a[a_idx + a_stride1]);
      op_T b1 = static_cast<op_T>(b[b_idx + b_stride0]);

      op_T a2 = static_cast<op_T>(a[a_idx + 2 * a_stride1]);
      op_T b2 = static_cast<op_T>(b[b_idx + 2 * b_stride0]);

      op_T a3 = static_cast<op_T>(a[a_idx + 3 * a_stride1]);
      op_T b3 = static_cast<op_T>(b[b_idx + 3 * b_stride0]);

      acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;

      a_idx += 4 * a_stride1;
      b_idx += 4 * b_stride0;
    }

    // tail
    for (; n < N; n++) {
      acc += static_cast<op_T>(a[a_idx]) *
             static_cast<op_T>(b[b_idx]);
      a_idx += a_stride1;
      b_idx += b_stride0;
    }

    r[row * r_stride0 + col * r_stride1] =
        static_cast<T>(acc * alpha);
  }
}

template <typename T>
kernel void sdpa(
    device T* out [[buffer(0)]],
    device T* attn [[buffer(1)]],
    constant T* q [[buffer(2)]],
    constant T* k [[buffer(3)]],
    constant T* v [[buffer(4)]],
    constant SDPAParams<>& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {

  using op_T = c10::metal::opmath_t<T>;

  // Each threadgroup operates on one head of one batch.
  uint32_t head_idx = tgid % params.num_heads;
  uint32_t batch_idx = tgid / params.num_heads;
  out += params.out_strides[0] * batch_idx + params.out_strides[1] * head_idx;
  attn += params.attn_strides[0] * batch_idx + params.attn_strides[1] * head_idx;
  q += params.q_strides[0] * batch_idx + params.q_strides[1] * head_idx;
  k += params.k_strides[0] * batch_idx + params.k_strides[1] * head_idx;
  v += params.v_strides[0] * batch_idx + params.v_strides[1] * head_idx;

  // Now we are operating on single matrices. q is size L x E, k is S x E, v is S x Ev.
  // We have to do a matmul between q and the transpose of k. Then run softmax on that. Then
  // matmul the result by v.

  auto L = params.L;        // query sequence length
  auto S = params.S;        // key/value sequence length
  auto E = params.E;        // embedding dimension for q, k
  auto Ev = params.Ev;      // embedding dimension for v, output

  // Compute attn = q @ k^T
  // Note that the strides of `k` are swapped, in order to transpose it
  mm(
    attn, params.attn_strides[2], params.attn_strides[3],
    q, params.q_strides[2], params.q_strides[3],
    k, params.k_strides[3], params.k_strides[2],
    L, E, S,
    params.scale,
    tid, tptg);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // In-place softmax on attn
  for (uint32_t row = tid; row < L; row += tptg) {
    op_T exp_sum = 0;
    for (uint32_t col = 0; col < S; col++) {
      auto elem = static_cast<op_T>(attn[row * params.attn_strides[2] + col * params.attn_strides[3]]);
      exp_sum += precise::exp(elem);
    }
    for (uint32_t col = 0; col < S; col++) {
      auto elem = static_cast<op_T>(attn[row * params.attn_strides[2] + col * params.attn_strides[3]]);
      attn[row * params.attn_strides[2] + col * params.attn_strides[3]] = static_cast<T>(precise::exp(elem) / exp_sum);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute out = attn @ v
  mm(
    out, params.out_strides[2], params.out_strides[3],
    attn, params.attn_strides[2], params.attn_strides[3],
    v, params.v_strides[2], params.v_strides[3],
    L, S, Ev,
    static_cast<op_T>(1),
    tid, tptg);
}

// Tiled matmul: r[M,K] = alpha * a[M,N] @ b[N,K]
//
// Key idea: instead of each thread independently reading all N elements from
// global memory (very bandwidth-heavy), threads cooperate to load
// TILE_SIZE×TILE_SIZE sub-blocks of a and b into fast threadgroup memory, then
// each thread accumulates partial dot-products from the cached tiles.
//
// The +1 column padding (TILE_SIZE+1 instead of TILE_SIZE) avoids shared memory
// bank conflicts when threads in the same column read from tile_a, and when
// threads in the same row read from tile_b.
//
// smem must point to at least 2 * TILE_SIZE * (TILE_SIZE+1) floats of
// threadgroup memory.
template <typename T, typename A_ptr>
static void mm_tiled(
    device T* r,         uint r_s0, uint r_s1,
    A_ptr a,             uint a_s0, uint a_s1,
    constant T* b,       uint b_s0, uint b_s1,
    uint M, uint N, uint K,
    float alpha,
    uint lr, uint lc,    // 2-D local thread position (row, col) within TILE_SIZE
    threadgroup float* smem
) {
  constexpr uint PAD = TILE_SIZE;
  threadgroup float* tile_a = smem;              // [TILE_SIZE][PAD]
  threadgroup float* tile_b = smem + TILE_SIZE * PAD; // [TILE_SIZE][PAD]
  const uint num_M = (M + TILE_SIZE - 1) / TILE_SIZE;
  const uint num_K = (K + TILE_SIZE - 1) / TILE_SIZE;
  const uint num_N = (N + TILE_SIZE - 1) / TILE_SIZE;

  for (uint tm = 0; tm < num_M; tm++) {
    const uint out_row = tm * TILE_SIZE + lr;

    for (uint tk = 0; tk < num_K; tk++) {
      const uint out_col = tk * TILE_SIZE + lc;
      float acc = 0.f;

      for (uint tn = 0; tn < num_N; tn++) {
        // --- cooperatively load one TILE_SIZE of A and one TILE_SIZE of B ---
        uint ar = tm * TILE_SIZE + lr;
        uint ac = tn * TILE_SIZE + lc;
        tile_a[lr * PAD + lc] = (ar < M && ac < N)
            ? float(a[ar * a_s0 + ac * a_s1]) : 0.f;
        uint br = tn * TILE_SIZE + lr, bc = tk * TILE_SIZE + lc;
        tile_b[lr * PAD + lc] = (br < N && bc < K)
            ? float(b[br * b_s0 + bc * b_s1]) : 0.f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- accumulate dot-product from shared memory ---
        for (uint n = 0; n < TILE_SIZE; n++) {
          acc = fma(tile_a[lr * PAD + n], tile_b[n * PAD + lc], acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      if (out_row < M && out_col < K) {
        r[out_row * r_s0 + out_col * r_s1] = T(acc * alpha);
      }
    }
  }
}

// Numerically stable in-place row-wise softmax on attn[L, S].
//
// Uses the lc dimension (TILE_SIZE threads per lr-row) to parallelize the two
// reductions (max and sum) via a tree reduction in shared memory.  This fixes
// the overflow risk in the original (which omitted max subtraction).
//
// smem must be at least TILE_SIZE*TILE_SIZE floats (fits inside the mm_tiled
// allocation).
template <typename T>
static void softmax_rows(
    device T* attn, uint s0, uint s1,
    uint L, uint S,
    uint lr, uint lc,
    threadgroup float* smem
) {
    // All threads iterate the same number of outer steps so every thread hits
    // every barrier — required by Metal's threadgroup_barrier rules.
    const uint num_row_tiles = (L + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tm = 0; tm < num_row_tiles; tm++) {
        const uint m = tm * TILE_SIZE + lr;
        const bool valid = m < L;

        // ---- step 1: find row max (lc threads collaborate) ----
        float local_max = -INFINITY;
        if (valid) {
            for (uint col = lc; col < S; col += TILE_SIZE)
                local_max = max(local_max, float(attn[m * s0 + col * s1]));
        }
        smem[lr * TILE_SIZE + lc] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
            if (lc < stride)
                smem[lr * TILE_SIZE + lc] = max(smem[lr * TILE_SIZE + lc],
                                           smem[lr * TILE_SIZE + lc + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float row_max = smem[lr * TILE_SIZE]; // broadcast: all threads read same slot

        // ---- step 2: sum of exp(x - max) ----
        float local_sum = 0.f;
        if (valid) {
            for (uint col = lc; col < S; col += TILE_SIZE)
                local_sum += precise::exp(float(attn[m * s0 + col * s1]) - row_max);
        }
        smem[lr * TILE_SIZE + lc] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
            if (lc < stride)
                smem[lr * TILE_SIZE + lc] += smem[lr * TILE_SIZE + lc + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float row_sum = smem[lr * TILE_SIZE];

        // ---- step 3: normalize ----
        if (valid) {
            for (uint col = lc; col < S; col += TILE_SIZE) {
                float e = precise::exp(float(attn[m * s0 + col * s1]) - row_max);
                attn[m * s0 + col * s1] = T(e / row_sum);
            }
        }
        // mem_device fence: writes to attn (device memory) must be visible to
        // all threads before the next mm_tiled reads from it.
        threadgroup_barrier(mem_flags::mem_device);
    }
}

template <typename T>
kernel void sdpa_tiled(
    device T*               out    [[buffer(0)]],
    device T*               attn   [[buffer(1)]],
    constant T*             q      [[buffer(2)]],
    constant T*             k      [[buffer(3)]],
    constant T*             v      [[buffer(4)]],
    constant SDPAParams<>&  params [[buffer(5)]],
    uint2 lid   [[thread_position_in_threadgroup]], // (col=x, row=y)
    uint2  tgid  [[threadgroup_position_in_grid]]   // one group per (batch, head)
) {
    threadgroup float smem[2 * TILE_SIZE * (TILE_SIZE + 1)];
    const uint head_idx  = tgid.x % params.num_heads;
    const uint batch_idx = tgid.x / params.num_heads;
    out  += params.out_strides[0]  * batch_idx + params.out_strides[1]  * head_idx;
    attn += params.attn_strides[0] * batch_idx + params.attn_strides[1] * head_idx;
    q    += params.q_strides[0]    * batch_idx + params.q_strides[1]    * head_idx;
    k    += params.k_strides[0]    * batch_idx + params.k_strides[1]    * head_idx;
    v    += params.v_strides[0]    * batch_idx + params.v_strides[1]    * head_idx;

    const auto L = params.L, S = params.S, E = params.E, Ev = params.Ev;
    const uint lr = lid.y, lc = lid.x;

    // 1) attn[L,S] = scale * q[L,E] @ k^T[E,S]   (swap k's strides to transpose)
    mm_tiled<T, constant T*>(
        attn, params.attn_strides[2], params.attn_strides[3],
        q,    params.q_strides[2],    params.q_strides[3],
        k,    params.k_strides[3],    params.k_strides[2],  // <-- transposed
        L, E, S, params.scale, lr, lc, smem);

    // Fence: attn writes (device memory) must reach softmax reads.
    threadgroup_barrier(mem_flags::mem_device);

    // 2) softmax(attn) in-place, row-wise
    softmax_rows<T>(attn, params.attn_strides[2], params.attn_strides[3],
                    L, S, lr, lc, smem);

    // softmax_rows already issues a mem_device barrier at its end, so it's not
    // needed here.

    // 3) out[L,Ev] = attn[L,S] @ v[S,Ev]
    mm_tiled<T, device T*>(
        out,  params.out_strides[2],  params.out_strides[3],
        attn, params.attn_strides[2], params.attn_strides[3],
        v,    params.v_strides[2],    params.v_strides[3],
        L, S, Ev, 1.f, lr, lc, smem);
}



// simdgroup_multiply_accumulate operates on 8×8 blocks
#define SG_TILE   8
// number of 8-wide slices across one TILE_SIZE dimension
#define SG_GRID   (TILE_SIZE / SG_TILE)   // 4
// +1 eliminates bank conflicts: tile_a[lr * PAD + n] → bank (lr + n) % 32
#define PAD       (TILE_SIZE)
// smem needed: two TILE_SIZE×PAD planes (also covers softmax reduction)
#define SMEM_FLOATS (2 * TILE_SIZE * PAD)  // 2 × 32 × 33 = 2112 floats ≈ 8.25 KB

// ---------------------------------------------------------------------------
// Tiled matmul: r[M,K] = alpha * a[M,N] @ b[N,K]
//
// Uses simdgroup_multiply_accumulate to hit Apple Silicon's hardware matrix
// units, the same path taken by MPSGraph.
//
// Thread layout: 32×32 = 32 simdgroups. The 16 active ones (sg_idx 0..15)
// are arranged in a 4×4 grid, each owning one 8×8 output subblock of the
// 32×32 macro-tile. The remaining 16 contribute to the cooperative loads but
// skip the matrix multiply.
//
// smem: 2 × TILE_SIZE × PAD floats (tile_a then tile_b, no overlap)
// ---------------------------------------------------------------------------
template <typename T, typename A_ptr>
static void mm_simdgroup(
    device T*    r,  uint r_s0, uint r_s1,
    A_ptr        a,  uint a_s0, uint a_s1,
    constant T*  b,  uint b_s0, uint b_s1,
    uint M, uint N, uint K,
    float alpha,
    uint lr, uint lc, uint sg_idx,
    threadgroup float* smem
) {
    threadgroup float* tile_a = smem;
    threadgroup float* tile_b = smem + TILE_SIZE * PAD;

    const uint num_M = (M + TILE_SIZE - 1) / TILE_SIZE;
    const uint num_K = (K + TILE_SIZE - 1) / TILE_SIZE;
    const uint num_N = (N + TILE_SIZE - 1) / TILE_SIZE;

    // sg_idx 0..15: active (4×4 grid of 8×8 output blocks).
    // sg_idx 16..31: load data but do no multiply-accumulate.
    const bool active  = (sg_idx < (uint)(SG_GRID * SG_GRID));
    const uint sg_row  = sg_idx / SG_GRID;  // 0..3 for active
    const uint sg_col  = sg_idx % SG_GRID;  // 0..3 for active

    for (uint tm = 0; tm < num_M; tm++) {
        for (uint tk = 0; tk < num_K; tk++) {

            // Initialise for every simdgroup (avoids UB; inactive ones never
            // reach simdgroup_store so the value is never observed).
            simdgroup_float8x8 c_sg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

            for (uint tn = 0; tn < num_N; tn++) {
                // --- all 1024 threads cooperatively fill tile_a and tile_b ---
                {
                    uint ar = tm * TILE_SIZE + lr, ac = tn * TILE_SIZE + lc;
                    tile_a[lr * PAD + lc] = (ar < M && ac < N)
                        ? float(a[ar * a_s0 + ac * a_s1]) : 0.f;
                    uint br = tn * TILE_SIZE + lr, bc = tk * TILE_SIZE + lc;
                    tile_b[lr * PAD + lc] = (br < N && bc < K)
                        ? float(b[br * b_s0 + bc * b_s1]) : 0.f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // --- active simdgroups compute their 8×8 subblock ---
                // Iterates SG_GRID=4 times to cover the full 32-wide N dimension
                // of the tile. Each iteration multiplies two 8×8 fragments and
                // accumulates into c_sg.
                if (active) {
                    for (uint k8 = 0; k8 < (uint)SG_GRID; k8++) {
                        simdgroup_float8x8 a_sg, b_sg;
                        // a_sg ← tile_a[sg_row*8 .. sg_row*8+7][k8*8 .. k8*8+7]
                        simdgroup_load(a_sg, tile_a, PAD,
                                       ulong2(k8 * SG_TILE, sg_row * SG_TILE));
                        // b_sg ← tile_b[k8*8 .. k8*8+7][sg_col*8 .. sg_col*8+7]
                        simdgroup_load(b_sg, tile_b, PAD,
                                       ulong2(sg_col * SG_TILE, k8 * SG_TILE));
                        simdgroup_multiply_accumulate(c_sg, a_sg, b_sg, c_sg);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } // tn

            // --- scatter result back through smem, then write to r ---
            // The 16 active simdgroups cover all 4×4 = 16 blocks of the
            // 32×32 output tile, so every position in tile_a is written.
            if (active) {
                simdgroup_store(c_sg, tile_a, PAD,
                                ulong2(sg_col * SG_TILE, sg_row * SG_TILE));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            {
                uint out_row = tm * TILE_SIZE + lr;
                uint out_col = tk * TILE_SIZE + lc;
                if (out_row < M && out_col < K)
                    r[out_row * r_s0 + out_col * r_s1] =
                        T(tile_a[lr * PAD + lc] * alpha);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } // tk
    } // tm
}

// ---------------------------------------------------------------------------
// In-place numerically-stable row-wise softmax on attn[L, S].
//
// Uses an online (single-pass) algorithm to compute max and sum jointly,
// cutting global reads of attn from 3 → 2 versus the standard two-scan
// approach. The TILE_SIZE threads sharing an lr value collaborate via a
// log₂(TILE_SIZE) tree reduction.
//
// smem: two TILE_SIZE×TILE_SIZE planes (fits in SMEM_FLOATS).
// ---------------------------------------------------------------------------
template <typename T>
static void softmax_rows_new(
    device T* attn, uint s0, uint s1,
    uint L, uint S,
    uint lr, uint lc,
    threadgroup float* smem
) {
    // Use the first half of smem for max, second for sum.
    // Both planes fit inside the 2*TILE_SIZE*PAD allocation (2048 < 2112).
    threadgroup float* smem_max = smem;
    threadgroup float* smem_sum = smem + TILE_SIZE * TILE_SIZE;

    const uint num_row_tiles = (L + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tm = 0; tm < num_row_tiles; tm++) {
        const uint m   = tm * TILE_SIZE + lr;
        const bool valid = m < L;

        // ---- Pass 1: online joint max + sum (single scan) ----------------
        // When a new value x exceeds the running max, rescale the running sum:
        //   sum_new = sum_old * exp(max_old - x) + 1
        // This is equivalent to the two-pass approach but reads attn once.
        float local_max = -INFINITY;
        float local_sum = 0.f;
        if (valid) {
            for (uint col = lc; col < S; col += TILE_SIZE) {
                float x = float(attn[m * s0 + col * s1]);
                if (x > local_max) {
                    local_sum = local_sum * precise::exp(local_max - x) + 1.f;
                    local_max = x;
                } else {
                    local_sum += precise::exp(x - local_max);
                }
            }
        }

        // Tree-reduce across lc: merge (max, sum) pairs with the identity
        //   m_new = max(m_a, m_b)
        //   s_new = s_a * exp(m_a - m_new) + s_b * exp(m_b - m_new)
        smem_max[lr * TILE_SIZE + lc] = local_max;
        smem_sum[lr * TILE_SIZE + lc] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
            if (lc < stride) {
                float ma = smem_max[lr * TILE_SIZE + lc];
                float mb = smem_max[lr * TILE_SIZE + lc + stride];
                float sa = smem_sum[lr * TILE_SIZE + lc];
                float sb = smem_sum[lr * TILE_SIZE + lc + stride];
                float mn = max(ma, mb);
                smem_max[lr * TILE_SIZE + lc] = mn;
                smem_sum[lr * TILE_SIZE + lc] =
                    sa * precise::exp(ma - mn) + sb * precise::exp(mb - mn);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float row_max = smem_max[lr * TILE_SIZE];  // broadcast
        const float row_sum = smem_sum[lr * TILE_SIZE];

        // ---- Pass 2: normalise and write ---------------------------------
        if (valid) {
            for (uint col = lc; col < S; col += TILE_SIZE) {
                float x = float(attn[m * s0 + col * s1]);
                attn[m * s0 + col * s1] =
                    T(precise::exp(x - row_max) / row_sum);
            }
        }
        // mem_device: softmax writes to attn (device memory) must be visible
        // to all threads before the next mm_simdgroup reads from it.
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ---------------------------------------------------------------------------
// SDPA kernel
// ---------------------------------------------------------------------------
template <typename T>
kernel void sdpa_tiled_new(
    device T*               out    [[buffer(0)]],
    device T*               attn   [[buffer(1)]],
    constant T*             q      [[buffer(2)]],
    constant T*             k      [[buffer(3)]],
    constant T*             v      [[buffer(4)]],
    constant SDPAParams<>&  params [[buffer(5)]],
    uint2 lid     [[thread_position_in_threadgroup]],  // (x=col, y=row)
    uint2 tgid    [[threadgroup_position_in_grid]],    // one group per (batch, head)
    uint  sg_idx  [[simdgroup_index_in_threadgroup]]   // 0..31
) {
    threadgroup float smem[SMEM_FLOATS];

    const uint head_idx  = tgid.x % params.num_heads;
    const uint batch_idx = tgid.x / params.num_heads;
    out  += params.out_strides[0]  * batch_idx + params.out_strides[1]  * head_idx;
    attn += params.attn_strides[0] * batch_idx + params.attn_strides[1] * head_idx;
    q    += params.q_strides[0]    * batch_idx + params.q_strides[1]    * head_idx;
    k    += params.k_strides[0]    * batch_idx + params.k_strides[1]    * head_idx;
    v    += params.v_strides[0]    * batch_idx + params.v_strides[1]    * head_idx;

    const auto L = params.L, S = params.S, E = params.E, Ev = params.Ev;
    const uint lr = lid.y, lc = lid.x;

    // 1) attn[L,S] = scale * q[L,E] @ k^T[E,S]  (k's strides swapped = transpose)
    mm_simdgroup<T, constant T*>(
        attn, params.attn_strides[2], params.attn_strides[3],
        q,    params.q_strides[2],    params.q_strides[3],
        k,    params.k_strides[3],    params.k_strides[2],   // <-- transposed
        L, E, S, params.scale,
        lr, lc, sg_idx, smem);

    threadgroup_barrier(mem_flags::mem_device);

    // 2) softmax(attn) in-place; ends with its own mem_device barrier
    softmax_rows_new<T>(attn, params.attn_strides[2], params.attn_strides[3],
                    L, S, lr, lc, smem);

    // 3) out[L,Ev] = attn[L,S] @ v[S,Ev]
    mm_simdgroup<T, device T*>(
        out,  params.out_strides[2],  params.out_strides[3],
        attn, params.attn_strides[2], params.attn_strides[3],
        v,    params.v_strides[2],    params.v_strides[3],
        L, S, Ev, 1.f,
        lr, lc, sg_idx, smem);
}








// PyTorch's sdpa_vector kernel (one-pass variant)
template <typename T, int D, int V = D>
kernel void sdpa_vector_new(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const device T* mask [[buffer(4)]],
    constant SDPANewParams& params [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  auto gqa_factor = params.gqa_factor;
  auto N = params.N;
  auto qkv_head_strides = params.qkv_head_strides;
  auto qkv_seq_strides = params.qkv_seq_strides;
  auto scale = params.scale;
  auto mask_strides = params.mask_strides;
  auto has_mask = params.has_mask;
  auto qkv_batch_strides = params.qkv_batch_strides;
  auto num_q_heads = params.num_q_heads;
  auto is_causal = params.is_causal;

  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint qk_per_thread = D / BD;
  constexpr uint v_per_thread = V / BD;
  const uint q_head_stride = qkv_head_strides[0];
  const uint q_seq_stride = qkv_seq_strides[0];
  const uint q_batch_stride = qkv_batch_strides[0];
  const uint k_head_stride = qkv_head_strides[1];
  const uint k_seq_stride = qkv_seq_strides[1];
  const uint k_batch_stride = qkv_batch_strides[1];
  const uint v_head_stride = qkv_head_strides[2];
  const uint v_seq_stride = qkv_seq_strides[2];
  const uint v_batch_stride = qkv_batch_strides[2];
  const uint mask_head_stride = mask_strides[0];
  const uint mask_kv_seq_stride = mask_strides[1];
  const uint mask_q_seq_stride = mask_strides[2];
  uint inner_k_stride = BN * int(k_seq_stride);
  uint inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.x;  // Flattened batch*heads index
  const int q_seq_idx = tid.y;

  // Decompose flattened head_idx into batch and head indices
  const int batch_idx = head_idx / num_q_heads;
  const int head_in_batch = head_idx % num_q_heads;
  const int kv_head_idx = head_in_batch / gqa_factor;

  const int Q = tpg.y;
  const int group_offset = head_idx * Q + q_seq_idx;
  const int o_offset = group_offset;

  // Use decomposed indices with separate batch and head strides
  queries += batch_idx * q_batch_stride + head_in_batch * q_head_stride + q_seq_idx * q_seq_stride +
      simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
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
    // Check mask: for floating point masks, values > -1e9 are considered valid (not masked)
    // Masked positions typically have -inf or very negative values
    bool is_valid = !has_mask || (static_cast<U>(mask[0]) > -1e9f);

    // Apply causal masking: compute absolute query position and mask future keys
    // Absolute query position = (N - Q) + q_seq_idx where Q = tpg.y
    if (is_causal) {
      is_valid = is_valid && (i <= (N - int(tpg.y) + int(q_seq_idx)));
    }

    if (is_valid) {
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

      // Add mask value to score if mask is present
      if (has_mask) {
        score += static_cast<U>(mask[0]);
      }

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

#define INSTANTIATE_SDPA_VECTOR_NEW(T, QK_DIM, VALUE_DIM)   \
  template [[host_name("sdpa_vector_new_" #T "_" #QK_DIM    \
                       "_" #VALUE_DIM)]] kernel void        \
  sdpa_vector_new<T, QK_DIM, VALUE_DIM>(                    \
    const device T* queries [[buffer(0)]], \
    const device T* keys [[buffer(1)]], \
    const device T* values [[buffer(2)]], \
    device T* out [[buffer(3)]], \
    const device T* mask [[buffer(4)]], \
    constant SDPANewParams& params [[buffer(5)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint3 tpg [[threadgroups_per_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_NEW_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR_NEW(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR_NEW(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR_NEW(DTYPE, 128, 128);

INSTANTIATE_SDPA_VECTOR_NEW_HEADS(float);
INSTANTIATE_SDPA_VECTOR_NEW_HEADS(bfloat);

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)   \
  template [[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM    \
                       "_" #VALUE_DIM)]] kernel void        \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM>(                    \
      const device DTYPE* queries [[buffer(0)]],            \
      const device DTYPE* keys [[buffer(1)]],               \
      const device DTYPE* values [[buffer(2)]],             \
      device DTYPE* out [[buffer(3)]],                      \
      const constant uint& gqa_factor [[buffer(4)]],        \
      const constant uint& N [[buffer(5)]],                 \
      const constant uint3& qkv_head_strides [[buffer(6)]], \
      const constant uint3& qkv_seq_strides [[buffer(7)]],  \
      const constant float& scale [[buffer(8)]],            \
      const device bool* mask [[buffer(9)]],                \
      const constant uint3& mask_strides [[buffer(10)]],    \
      const constant bool& has_mask [[buffer(11)]],         \
      uint3 tid [[threadgroup_position_in_grid]],           \
      uint3 tpg [[threadgroups_per_grid]],                  \
      uint simd_gid [[simdgroup_index_in_threadgroup]],     \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM) \
  template [[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM  \
                       "_" #VALUE_DIM)]] kernel void              \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM>(                  \
      const device DTYPE* queries [[buffer(0)]],                  \
      const device DTYPE* keys [[buffer(1)]],                     \
      const device DTYPE* values [[buffer(2)]],                   \
      device DTYPE* out [[buffer(3)]],                            \
      device float* sums [[buffer(4)]],                           \
      device float* maxs [[buffer(5)]],                           \
      const constant uint& gqa_factor [[buffer(6)]],              \
      const constant uint& N [[buffer(7)]],                       \
      const constant uint3& qkv_head_strides [[buffer(8)]],       \
      const constant uint3& qkv_seq_strides [[buffer(9)]],        \
      const constant float& scale [[buffer(10)]],                 \
      const device bool* mask [[buffer(11)]],                     \
      const constant uint3& mask_strides [[buffer(12)]],          \
      const constant bool& has_mask [[buffer(13)]],               \
      uint3 tid [[threadgroup_position_in_grid]],                 \
      uint3 tpg [[threadgroups_per_grid]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],           \
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

#define INSTANTIATE_ATTN(DTYPE, bq, bk, bd, wm, wn)                      \
  template [[host_name("attention_" #DTYPE "_bq" #bq "_bk" #bk "_bd" #bd \
                       "_wm" #wm "_wn" #wn)]] [[kernel]] void            \
  attention<DTYPE, bq, bk, bd, wm, wn>(                                  \
      const device DTYPE* Q [[buffer(0)]],                               \
      const device DTYPE* K [[buffer(1)]],                               \
      const device DTYPE* V [[buffer(2)]],                               \
      device DTYPE* O [[buffer(3)]],                                     \
      const constant uint& qL [[buffer(4)]],                             \
      const constant uint& kL [[buffer(5)]],                             \
      const constant uint& gqa_factor [[buffer(6)]],                     \
      const constant float& scale [[buffer(7)]],                         \
      const constant uint& NK [[buffer(8)]],                             \
      const constant uint3& Q_strides [[buffer(9)]],                     \
      const constant uint3& K_strides [[buffer(10)]],                    \
      const constant uint3& V_strides [[buffer(11)]],                    \
      const constant uint3& O_strides [[buffer(12)]],                    \
      uint3 group_pos [[threadgroup_position_in_grid]],                  \
      uint3 local_pos [[thread_position_in_threadgroup]]);

#define INSTANTIATE_ATTN_SHAPES_HELPER(dtype) \
  INSTANTIATE_ATTN(dtype, 32, 16, 128, 4, 1)  \
  INSTANTIATE_ATTN(dtype, 32, 32, 80, 4, 1)   \
  INSTANTIATE_ATTN(dtype, 32, 32, 64, 4, 1)

INSTANTIATE_ATTN_SHAPES_HELPER(float);
INSTANTIATE_ATTN_SHAPES_HELPER(half);
INSTANTIATE_ATTN_SHAPES_HELPER(bfloat);

#define REGISTER_SDPA(T)                         \
template [[host_name("sdpa_" #T)]]               \
kernel void sdpa<T>(                             \
    device T* out [[buffer(0)]],                 \
    device T* attn [[buffer(1)]],                \
    constant T* q [[buffer(2)]],                 \
    constant T* k [[buffer(3)]],                 \
    constant T* v [[buffer(4)]],                 \
    constant SDPAParams<>& params [[buffer(5)]], \
    uint tid [[thread_position_in_threadgroup]], \
    uint tptg [[threads_per_threadgroup]],       \
    uint tgid [[threadgroup_position_in_grid]]);

REGISTER_SDPA(float);
REGISTER_SDPA(half);
REGISTER_SDPA(bfloat);

#define REGISTER_SDPA_TILED(T)                         \
template [[host_name("sdpa_tiled_" #T)]]               \
kernel void sdpa_tiled<T>(                             \
    device T*               out    [[buffer(0)]],      \
    device T*               attn   [[buffer(1)]],      \
    constant T*             q      [[buffer(2)]],      \
    constant T*             k      [[buffer(3)]],      \
    constant T*             v      [[buffer(4)]],      \
    constant SDPAParams<>&  params [[buffer(5)]],      \
    uint2 lid   [[thread_position_in_threadgroup]],    \
    uint2  tgid  [[threadgroup_position_in_grid]]);    \
template [[host_name("sdpa_tiled_new_" #T)]]               \
kernel void sdpa_tiled_new<T>(                             \
    device T*               out    [[buffer(0)]],      \
    device T*               attn   [[buffer(1)]],      \
    constant T*             q      [[buffer(2)]],      \
    constant T*             k      [[buffer(3)]],      \
    constant T*             v      [[buffer(4)]],      \
    constant SDPAParams<>&  params [[buffer(5)]],      \
    uint2 lid   [[thread_position_in_threadgroup]],    \
    uint2  tgid  [[threadgroup_position_in_grid]],     \
    uint  sg_idx  [[simdgroup_index_in_threadgroup]]);

REGISTER_SDPA_TILED(float);
REGISTER_SDPA_TILED(half);
REGISTER_SDPA_TILED(bfloat);
