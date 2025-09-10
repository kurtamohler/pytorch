#include <ATen/native/mps/kernels/EmbeddingBag.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;


template <typename T, typename I>
void embedding_bag_impl(
    constant T* weight,
    constant I* indices,
    constant I* offsets,
    device T* output,
    device I* offset2bag,
    device I* bag_size,
    device I* max_indices,
    constant EmbeddingBagParams<uint32_t>& params,
    uint tid) {
  auto num_indices = params.num_indices;
  auto num_bags = params.num_bags;
  auto feature_size = params.feature_size;
  constant auto& output_strides = params.output_strides;
  constant auto& weight_strides = params.weight_strides;
  constant auto& max_indices_strides = params.max_indices_strides;

  auto bag_idx = tid / feature_size;
  auto feature_idx = tid % feature_size;

  output += bag_idx * output_strides[0] + feature_idx * output_strides[1];

  uint32_t offsets_end = min(bag_idx + 1, num_bags - 1);
  bool is_last_bag = bag_idx + 1 == num_bags;
  uint32_t indices_start = static_cast<uint32_t>(offsets[bag_idx]);
  uint32_t indices_end = is_last_bag * (num_indices) + (!is_last_bag) * (static_cast<uint32_t>(offsets[offsets_end]));
  uint32_t first_index = static_cast<uint32_t>(indices[indices_start]);

  T out_val;

  if (params.mode == EmbeddingBagMode::MAX) {
    out_val = weight[first_index * weight_strides[0] + feature_idx * weight_strides[1]];

    for (uint32_t indices_idx = indices_start + 1; indices_idx < indices_end; indices_idx++) {
      uint32_t index = static_cast<uint32_t>(indices[indices_idx]);
      T val = weight[index * weight_strides[0] + feature_idx * weight_strides[1]];
      out_val = max(val, out_val);
    }

  } else {
    out_val = 0;

    for (uint32_t indices_idx = indices_start; indices_idx < indices_end; indices_idx++) {
      uint32_t index = static_cast<uint32_t>(indices[indices_idx]);
      T val = weight[index * weight_strides[0] + feature_idx * weight_strides[1]];
      out_val += val;
    }
    if (params.mode == EmbeddingBagMode::MEAN) {
      out_val /= indices_end - indices_start;
    }
  }

  *output = out_val;
}

template <typename T, typename I>
kernel void embedding_bag(
    constant T* weight [[buffer(0)]],
    constant I* indices [[buffer(1)]],
    constant I* offsets [[buffer(2)]],
    device T* output [[buffer(3)]],
    device I* offset2bag [[buffer(4)]],
    device I* bag_size [[buffer(5)]],
    device I* max_indices [[buffer(6)]],
    constant EmbeddingBagParams<uint32_t>& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  embedding_bag_impl(
    weight,
    indices,
    offsets,
    output,
    offset2bag,
    bag_size,
    max_indices,
    params,
    tid
  );
}

#define REGISTER_EMBEDDING_BAG_OP(T, I)                            \
  template [[host_name("embedding_bag_" #T "_" #I)]]               \
  kernel void embedding_bag<T, I>(                                 \
      constant T * weight [[buffer(0)]],                           \
      constant I * indices [[buffer(1)]],                          \
      constant I * offsets [[buffer(2)]],                          \
      device T * output [[buffer(3)]],                             \
      device I * offset2bag [[buffer(4)]],                         \
      device I * bag_size [[buffer(5)]],                           \
      device I * max_indices [[buffer(6)]],                        \
      constant EmbeddingBagParams<uint32_t>& params [[buffer(7)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_EMBEDDING_BAG_OP(float, int);
REGISTER_EMBEDDING_BAG_OP(float, long);
REGISTER_EMBEDDING_BAG_OP(half, int);
REGISTER_EMBEDDING_BAG_OP(half, long);
REGISTER_EMBEDDING_BAG_OP(bfloat, int);
REGISTER_EMBEDDING_BAG_OP(bfloat, long);
