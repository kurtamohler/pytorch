#pragma once
#include <c10/core/DefaultDtype.h>
// #include <c10/core/Backend.h>
// #include <c10/core/Layout.h>
// #include <c10/core/ScalarType.h>
// #include <c10/core/Device.h>
// #include <c10/core/TensorTypeSet.h>

// #include <c10/util/Optional.h>
// #include <c10/util/C++17.h>
// #include <c10/macros/Macros.h>

#include <c10/util/Exception.h>
#include <c10/util/C++17.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
// #include <ATen/native/TensorIterator.h>

#include <iostream>
#include <vector>
#include <bitset>

namespace c10 {

using DimMask = std::bitset<64>;

class C10_API ReductionDim {
public:
  ReductionDim()
    : has_value_(false)
  {
    std::cout << "Constructor, no args" << std::endl;
  }

  ReductionDim(IntArrayRef dims) {

  }

  ReductionDim(std::vector<int64_t> dims)
    : has_value_(true), dims_(dims)
  {
    std::cout << "Constructor, vector arg" << std::endl;
  }



  // ReductionDim(IntArrayRef dims)
  //   :has_value_(true)
  // {
  //   // TODO: need code here
  //   std::cout << "Constructor, IntArrayRef arg" << std::endl;

  // }
  // c10::ArrayRef

  // ReductionDim(c10::optional<IntArrayRef> opt_dims) {
  //   has_value_ = opt_dims.has_value();

  //   if (has_value_) {
  //     dims_ = opt_dims.value().vec();
  //   }
  // }

  // ReductionDim(c10::optional<ScalarType> opt_dim) {
  //   has_value_ = opt_dim.has_value();

  //   if (has_value_) {
  //     // ScalarType scalar = opt_dim.value();
  //     // dims_.push_back(scalar::Long);
  //     dims_.push_back(static_cast<uint64_t>(opt_dim.value()));
  //   }
  // }

  // ReductionDim(const Tensor& tensor, DimnameList dim_names) {
  //   dims_.reserve(dim_names.size());
  //   for (const auto& name : dim_names) {
  //     dims_.push_back(dimname_to_position(tensor, name));
  //   }
  // }

  ReductionDim(int64_t dim)
    : has_value_(true)
  {
    dims_.push_back(dim);
    std::cout << "Constructor, int arg" << std::endl;
  }

  std::vector<int64_t> vec() {
    return dims_;
  }

  bool has_value() const { return has_value_; }

  DimMask make_dim_mask(int64_t dims_in_tensor);

  // c10::optional<IntArrayRef> get_optional();

private:
  bool has_value_;
  std::vector<int64_t> dims_;


};

}