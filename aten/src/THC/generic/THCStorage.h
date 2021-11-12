#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorage.h"
#else

#define THCStorage THStorage

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THCStorage type.
#define THCudaStorage                       THCStorage
#define THCudaByteStorage                   THCStorage

TORCH_CUDA_CU_API void THCStorage_(
    setFlag)(THCState* state, THCStorage* storage, const char flag);
TORCH_CUDA_CU_API void THCStorage_(
    clearFlag)(THCState* state, THCStorage* storage, const char flag);

TORCH_CUDA_CU_API int THCStorage_(
    getDevice)(THCState* state, const THCStorage* storage);

#endif
