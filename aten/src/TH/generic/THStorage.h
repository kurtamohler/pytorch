#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THStorage.h"
#else

#include <c10/core/StorageImpl.h>

/* on pourrait avoir un liste chainee
   qui initialise math, lab structures (or more).
   mouais -- complique.

   Pb: THMapStorage is kind of a class
   THLab_()... comment je m'en sors?

   en template, faudrait que je les instancie toutes!!! oh boy!
   Et comment je sais que c'est pour Cuda? Le type float est le meme dans les <>

   au bout du compte, ca serait sur des pointeurs float/double... etc... = facile.
   primitives??
 */

// Struct definition is moved to THStorage.hpp (so this file stays C compatible)

#define THStorage at::StorageImpl

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THStorage type.
#define THByteStorage THStorage

/* should not differ with API */
TH_API void THStorage_(setFlag)(THStorage *storage, const char flag);
TH_API void THStorage_(clearFlag)(THStorage *storage, const char flag);

#endif
