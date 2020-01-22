#pragma once

// ${generated_comment}

#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/cuda_lazy_init.h>

#include <ATen/ATen.h>
#include <c10/core/ReductionDim.h>
#include <pybind11/pybind11.h>

// Contains inline wrappers around ATen functions that release the GIL and
// switch to the correct CUDA device.

namespace torch { namespace autograd {

using at::Tensor;
using at::Scalar;
using at::TensorList;
using at::IntArrayRef;
using at::Generator;
using at::Storage;
using c10::ReductionDim;

${py_method_dispatch}

}} // namespace torch::autograd
