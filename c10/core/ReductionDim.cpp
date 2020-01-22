// #include <ATen/core/ReductionDim.h>
#include <c10/core/ReductionDim.h>
// #include <ATen/WrapDimUtils.h>

namespace c10 {

DimMask ReductionDim::make_dim_mask(int64_t dims_in_tensor) {
  auto mask = DimMask();
  // if (has_value_) {
  //   for (int64_t dim : dims_) {
  //     mask.set(at::maybe_wrap_dim(dim, dims_in_tensor));
  //   }
  // } else {
  //   mask.flip();
  // }
  return mask;
}

// c10::optional<IntArrayRef> ReductionDim::get_optional() {
//   if (has_value_) {
//     return dims_;
//   } else {
//     return c10::nullopt;
//   }
// }

}