#pragma onces
#include <c10/metal/common.h>

struct OrgqrParams {
  uint32_t num_batches;
  uint32_t m;
  uint32_t n;
  uint32_t k;

  ::c10::metal::array<uint32_t, 3> A_strides;
  ::c10::metal::array<uint32_t, 2> tau_strides;
  ::c10::metal::array<uint32_t, 3> H_strides;
};
