#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <THC/THCStorage.h>
// Should work with THStorageClass
#include <TH/THStorageFunctions.hpp>

#include <c10/core/ScalarType.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
