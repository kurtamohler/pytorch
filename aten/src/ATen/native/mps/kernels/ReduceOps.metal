#include <ATen/native/mps/kernels/ReduceOps.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct norm_abs_functor {
  template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(::precise::abs(x));
  }

  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline float operator()(const T x) {
    const auto abs_2 = ::precise::abs(float2(x));
    return c10::metal::hypot(abs_2.x, abs_2.y);
  }
};

struct norm_cast_functor {
  template <
      typename T_from,
      typename T_to,
      enable_if_t<is_complex_v<T_from> && !is_complex_v<T_to>, bool> = true>
  inline T_to call(const T_from x) {
    return static_cast<T_to>(x.x);
  }

  template <
      typename T_from,
      typename T_to,
      enable_if_t<!is_complex_v<T_from> && is_complex_v<T_to>, bool> = true>
  inline T_to call(const T_from x) {
    return T_to{static_cast<decltype(T_to{}.x)>(x), 0};
  }

  template <
      typename T_from,
      typename T_to,
      enable_if_t<is_complex_v<T_from> == is_complex_v<T_to>, bool> = true>
  inline T_to call(const T_from x) {
    return static_cast<T_to>(x);
  }
};

// `reduction_idx` is the index of a particular batch of input elements that all
// get reduced to one output element. `reduction_element_idx` is the index of
// just one input element within its batch.
static uint32_t get_input_offset(
    uint32_t reduction_element_idx,
    uint32_t reduction_idx,
    constant NormParams<>& params) {
  uint32_t input_offset = 0;

  for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
    auto input_dim_size = params.input_sizes[dim];
    auto output_dim_size = params.output_sizes[dim];

    // If the the input and output have the same size for this dim, then this
    // dim is not being reduced, so we index by `reduction_idx`
    if (input_dim_size == output_dim_size) {
      auto index_in_dim = reduction_idx % input_dim_size;
      reduction_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];

      // Otherwise, this dim is being reduced, so we index by
      // `reduction_element_idx`
    } else {
      auto index_in_dim = reduction_element_idx % input_dim_size;
      reduction_element_idx /= input_dim_size;
      input_offset += index_in_dim * params.input_strides[dim];
    }
  }
  return input_offset;
}

template <typename TI, typename TC, typename TO>
kernel void norm(
    constant TI* input [[buffer(0)]],
    device TO* output [[buffer(1)]],
    constant NormParams<>& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  using TA = opmath_t<TO>;
  threadgroup TA shared_output[MAX_THREADGROUP_SIZE];
  TA thread_output = 0;

  if (params.p == INFINITY) {
    thread_output = -INFINITY;
  } else if (params.p == -INFINITY) {
    thread_output = INFINITY;
  }

  // Each thread in the threadgroup reduces one or more element of the input and
  // places the result in its corresponding element of `shared_output`.
  for (uint32_t reduction_element_idx = tid;
       reduction_element_idx < params.reduction_size;
       reduction_element_idx += tptg) {
    TC input_elem = norm_cast_functor().call<TI, TC>(
        input[get_input_offset(reduction_element_idx, tgid, params)]);
    TA input_abs = static_cast<TA>(norm_abs_functor()(input_elem));

    if (params.p == INFINITY) {
      thread_output = max(static_cast<TA>(input_abs), thread_output);

    } else if (params.p == -INFINITY) {
      thread_output = min(static_cast<TA>(input_abs), thread_output);

    } else if (params.p == 0) {
      thread_output = (input_abs == 0) ? 0 : 1;

    } else {
      thread_output += static_cast<TA>(::precise::pow(
          static_cast<TA>(norm_abs_functor()(input_abs)),
          static_cast<TA>(params.p)));
    }
  }

  shared_output[tid] = thread_output;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Binary reduce each element of `shared_output` across the threadgroup
  for (uint32_t stride = 1; stride < tptg; stride *= 2) {
    if ((tid % (stride * 2) == 0) && (tid + stride) < tptg) {
      if (params.p == INFINITY) {
        shared_output[tid] =
            max(shared_output[tid], shared_output[tid + stride]);
      } else if (params.p == -INFINITY) {
        shared_output[tid] =
            min(shared_output[tid], shared_output[tid + stride]);
      } else {
        shared_output[tid] += shared_output[tid + stride];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // One thread in the threadgroup writes the final reduced value to the output
  if (tid == 0) {
    uint32_t output_offset = 0;
    uint32_t reduction_idx = tgid;

    for (int32_t dim = params.ndim - 1; dim >= 0; dim--) {
      auto output_dim_size = params.output_sizes[dim];

      if (output_dim_size > 1) {
        auto index_in_dim = reduction_idx % output_dim_size;
        reduction_idx /= output_dim_size;
        output_offset += index_in_dim * params.output_strides[dim];
      }
    }

    auto output_elem = shared_output[0];
    if (params.p != 0 && params.p != 1 && params.p != INFINITY &&
        params.p != -INFINITY) {
      output_elem = static_cast<TA>(
          ::precise::pow(output_elem, static_cast<TA>(1 / params.p)));
    }
    output[output_offset] = static_cast<TO>(output_elem);
  }
}

#define REGISTER_NORM(TI, TC, TO)                     \
  template [[host_name("norm_" #TI "_" #TC "_" #TO)]] \
  kernel void norm<TI, TC, TO>(                       \
      constant TI * input [[buffer(0)]],              \
      device TO * output [[buffer(1)]],               \
      constant NormParams<> & params [[buffer(2)]],   \
      uint tid [[thread_position_in_threadgroup]],    \
      uint tptg [[threads_per_threadgroup]],          \
      uint tgid [[threadgroup_position_in_grid]]);

#define REGISTER_NORM_INPUT_TYPE(TI) \
  REGISTER_NORM(TI, float, float);   \
  REGISTER_NORM(TI, half, half);     \
  REGISTER_NORM(TI, bfloat, bfloat); \
  REGISTER_NORM(TI, float2, float);  \
  REGISTER_NORM(TI, half2, half);

REGISTER_NORM_INPUT_TYPE(float)
REGISTER_NORM_INPUT_TYPE(half)
REGISTER_NORM_INPUT_TYPE(bfloat)
REGISTER_NORM_INPUT_TYPE(float2)
REGISTER_NORM_INPUT_TYPE(half2)
