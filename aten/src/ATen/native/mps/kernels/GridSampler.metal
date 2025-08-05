#include <ATen/native/mps/kernels/GridSampler.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct GridSamplerOffsets {
  int32_t output;
  int32_t input;
  int32_t grid;

  GridSamplerOffsets() : output(0), input(0), grid(0) {}
};

// Find offsets into the tensors that this thread will operate on
GridSamplerOffsets find_grid_sampler_offsets(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    constant int32_t* grid_sizes,
    constant int32_t* grid_strides,
    int32_t sampler_dims,
    uint tid) {
  int32_t dims = sampler_dims + 2;
  GridSamplerOffsets offsets;

  auto output_idx = static_cast<int32_t>(tid);
  int32_t sampler_dim_indices[3];

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % output_sizes[dim];
    output_idx = output_idx / output_sizes[dim];

    // Select the output element that this thread will calculate.
    // output shape:
    //   2 sampler dims: (N, C, Hout, Wout)
    //   3 sampler dims: (N, C, Dout, Hout, Wout)
    offsets.output += output_strides[dim] * dim_idx;

    // Select the batch and channel for the input.
    // input shape:
    //   2 sampler dims: (N, C, Hin, Win)
    //   3 sampler dims: (N, C, Din, Hin, Win)
    if (dim <= 1) {
      offsets.input += input_strides[dim] * dim_idx;
    }

    // Select the grid coordinates for the output element.
    // grid shape:
    //   2 sampler dims: (N, Hout, Wout, 2)
    //   3 sampler dims: (N, Dout, Hout, Wout, 3)
    if (dim == 0) {
      offsets.grid += grid_strides[dim] * dim_idx;
    }
    if (dim >= 2) {
      int32_t grid_dim = dim - 1;
      offsets.grid += grid_strides[grid_dim] * dim_idx;
    }
  }

  return offsets;
}

// Calculates a single output element.
// input shape:
//    2 sampler dims: (Hin, Win)
//    3 sampler dims: (Din, Hin, Win)
// grid shape:
//    2 sampler dims: (2)
//    3 sampler dims: (3)
template <typename T>
void grid_sampler_single_element(
    device T* output,
    constant T* input,
    constant T* grid,
    int32_t sampler_dims,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    GridSamplerInterpolation interpolation_mode,
    GridSamplerPadding padding_mode,
    bool align_corners) {}

template <typename T>
kernel void grid_sampler(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto grid_sizes = params.grid_sizes.data();
  auto grid_strides = params.grid_strides.data();
  auto sampler_dims = params.sampler_dims;

  auto offsets = find_grid_sampler_offsets(
      output_sizes,
      output_strides,
      input_sizes,
      input_strides,
      grid_sizes,
      grid_strides,
      sampler_dims,
      tid);

  output += offsets.output;
  input += offsets.input;
  grid += offsets.grid;

  input_sizes += 2;
  input_strides += 2;

  auto interpolation_mode = params.interpolation_mode;
  auto padding_mode = params.padding_mode;
  auto align_corners = params.align_corners;

  grid_sampler_single_element(
      output,
      input,
      grid,
      sampler_dims,
      input_sizes,
      input_strides,
      interpolation_mode,
      padding_mode,
      align_corners);
}

#define REGISTER_GRID_SAMPLER_OP(DTYPE)                     \
  template [[host_name("grid_sampler_" #DTYPE)]]            \
  kernel void grid_sampler<DTYPE>(                          \
      device DTYPE * output [[buffer(0)]],                  \
      constant DTYPE * input [[buffer(1)]],                 \
      constant DTYPE * grid [[buffer(2)]],                  \
      constant GridSamplerParams<5> & params [[buffer(3)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_GRID_SAMPLER_OP(float);
REGISTER_GRID_SAMPLER_OP(half);
REGISTER_GRID_SAMPLER_OP(bfloat);
