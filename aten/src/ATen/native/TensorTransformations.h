#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/roll.h>
#endif

#include <c10/util/Exception.h>

namespace at {
namespace native {

static inline Tensor roll_common(const Tensor& self, IntArrayRef shifts, OptionalIntArrayRef opt_dims) {
  TORCH_CHECK(shifts.size() > 0, "`shifts` required");
  if ((!opt_dims.has_value() || opt_dims.value().size() == 0) && shifts.size() == 1) {
    auto flattened = self.contiguous().view(self.numel());
    return roll(flattened, shifts[0], 0).view(self.sizes());
  }
  TORCH_CHECK(
    opt_dims.has_value(),
    "dimensions cannot be None when shifts.size() > 1");
  IntArrayRef dims = opt_dims.value();
  TORCH_CHECK(
    shifts.size() == dims.size(),
    "shifts and dimensions must align. shifts: ", shifts.size(), ", dims:", dims.size()
  );
  TORCH_CHECK(
    dims.size() > 1,
    "dimensions must have more than one entry when shifts.size() > 1");
  auto tail_shifts = shifts.slice(1);
  auto tail_dims = dims.slice(1);
  auto first_dim_rolled = roll(self, shifts[0], dims[0]);
  return at::roll(first_dim_rolled, tail_shifts, tail_dims);
}

}}  // namespace at::native
