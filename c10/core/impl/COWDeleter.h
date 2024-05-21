#pragma once

#include <c10/macros/Export.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <variant>

namespace c10::impl::cow {

using COWSimAccessorID = std::uintptr_t;
enum class AccessType;

class COWSimDeleterContext {
 public:
  bool is_cowsim() const;
  void enable_cowsim();
  void check_write(COWSimAccessorID writer);
  void check_read(COWSimAccessorID reader);

 protected:
  COWSimDeleterContext() = default;
  ~COWSimDeleterContext() = default;

 private:
  void raise_warning(AccessType access_type);

  bool is_cowsim_ = false;
  bool has_first_writer_ = false;
  bool has_raised_ = false;
  COWSimAccessorID first_writer_ = 0;
};

// A COWDeleterContext object is used as the `ctx` argument for DataPtr
// to implement a Copy-on-write (COW) DataPtr.
class C10_API COWDeleterContext : public COWSimDeleterContext {
 public:
  // Creates an instance, holding the pair of data and original
  // deleter.
  //
  // Note that the deleter will only be called in our destructor if
  // the last reference to this goes away without getting
  // materialized.
  explicit COWDeleterContext(std::unique_ptr<void, DeleterFnPtr> data);

  // Increments the current refcount.
  void increment_refcount();

  // See README.md in this directory to understand the locking
  // strategy.

  // Represents a reference to the context.
  //
  // This is returned by decrement_refcount to allow the caller to
  // copy the data under the shared lock.
  using NotLastReference = std::shared_lock<std::shared_mutex>;

  // Represents the last reference to the context.
  //
  // This will be returned by decrement_refcount when it is the last
  // reference remaining and after any pending copies have completed.
  using LastReference = std::unique_ptr<void, DeleterFnPtr>;

  // Decrements the refcount, returning a handle indicating what to
  // do with it.
  std::variant<NotLastReference, LastReference> decrement_refcount();

  bool is_cow() const;
  void enable_cow();

 protected:
  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using cow::delete_context as the deleter.
  ~COWDeleterContext();

 private:
  std::shared_mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;
  bool is_cow_ = false;
};

// `cow_deleter` is used as the `ctx_deleter` for DataPtr to implement a COW
// DataPtr.
//
// Warning: This should only be called on a pointer to a COWDeleterContext that
// was allocated on the heap with `new`, because when the refcount reaches 0,
// the context is deleted with `delete`.
C10_API void cow_deleter(void* ctx);

// Enables future behavior to make operators always return a copy in cases where
// they currently conditionally return a view or a copy
C10_API void set_extra_conditional_view_warnings(bool mode);
C10_API bool get_extra_conditional_view_warnings();

} // namespace c10::impl::cow
