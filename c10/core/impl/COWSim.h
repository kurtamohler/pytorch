#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>

namespace c10::impl::cowsim {

using COWSimAliasGroupID  = std::uintptr_t;

C10_API void test_func();

class C10_API COWSimChecker : public c10::intrusive_ptr_target {
 public:
  COWSimChecker()
      : first_writer_(0) {}

  void check_on_read(COWSimAliasGroupID group_id);

  void check_on_write(COWSimAliasGroupID group_id);

 private:
  COWSimAliasGroupID first_writer_;
};

} // namespace c10::impl::cowsim