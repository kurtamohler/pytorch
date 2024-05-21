#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl::cow {

void cow_deleter(void* ctx) {
  static_cast<COWDeleterContext*>(ctx)->decrement_refcount();
}

COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data)
    : data_(std::move(data)) {
  // We never wrap a COWDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow_deleter);
}

auto COWDeleterContext::increment_refcount() -> void {
  auto refcount = ++refcount_;
  TORCH_INTERNAL_ASSERT(refcount > 1);
}

auto COWDeleterContext::decrement_refcount()
    -> std::variant<NotLastReference, LastReference> {
  auto refcount = --refcount_;
  TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
  if (refcount == 0) {
    std::unique_lock lock(mutex_);
    auto result = std::move(data_);
    lock.unlock();
    delete this;
    return {std::move(result)};
  }

  return std::shared_lock(mutex_);
}

bool COWDeleterContext::is_cow() const {
  return is_cow_;
}

void COWDeleterContext::enable_cow() {
  TORCH_INTERNAL_ASSERT(!is_cow_);
  is_cow_ = true;
}

COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

bool COWSimDeleterContext::is_cowsim() const {
  return is_cowsim_;
}

void COWSimDeleterContext::enable_cowsim() {
  TORCH_INTERNAL_ASSERT(!is_cowsim_);
  is_cowsim_ = true;
}

enum class AccessType { READ, WRITE };

void COWSimDeleterContext::raise_warning(AccessType access_type) {
  if (!has_raised_) {
    // TODO: Improve this message
    TORCH_WARN(
        "Detected divergent behavior on ",
        (access_type == AccessType::READ) ? "read" : "write");
    has_raised_ = true;
  }
}

void COWSimDeleterContext::check_write(COWSimAccessorID writer) {
  if (!has_first_writer_) {
    if (get_extra_conditional_view_warnings()) {
      TORCH_WARN("Detected first write to a deprecated conditional view")
    }
    has_first_writer_ = true;
    first_writer_ = writer;
  } else if (writer != first_writer_) {
    raise_warning(AccessType::WRITE);
  }
}

void COWSimDeleterContext::check_read(COWSimAccessorID reader) {
  if (has_first_writer_ && reader != first_writer_) {
    raise_warning(AccessType::READ);
  }
}

static bool _extra_conditional_view_warnings = false;

C10_API void set_extra_conditional_view_warnings(bool mode) {
  _extra_conditional_view_warnings = mode;
}

C10_API bool get_extra_conditional_view_warnings() {
  return _extra_conditional_view_warnings;
}

} // namespace c10::impl::cow
