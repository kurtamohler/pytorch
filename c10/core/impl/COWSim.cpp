#include <c10/core/impl/COWSim.h>
#include <iostream>

namespace c10::impl::cowsim {

void test_func() {
}

void raise_warning() {
  // TODO: Improve this message
  TORCH_WARN(
    "Detected read or write to a different alias group",
    " of the same alias that was previously written to after",
    " a conditional view was taken."
  );
}

void COWSimChecker::check_on_read(COWSimAliasGroupID group_id) {
  // TODO: Currently using `group_id == 0` for nosim, but need to
  // just add new arg for that

  if (first_writer_ == 0 || group_id == 0 || group_id == first_writer_) {
    return;
  } else {
    raise_warning();
  }

}

void COWSimChecker::check_on_write(COWSimAliasGroupID group_id) {
  // TODO: Currently using `group_id == 0` for nosim, but need to
  // just add new arg for that
  std::cout << "group_id: " << group_id << std::endl;

  if (group_id == 0) {
    return;
  } else if (first_writer_ == 0) {
    first_writer_ = group_id;
  } else if (group_id == first_writer_) {
    return;
  } else {
    raise_warning();
  }

}


} // namespace c10::impl::cowsim