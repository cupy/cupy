#ifndef INCLUDE_GUARD_CUPY_CUGRAPH_H
#define INCLUDE_GUARD_CUPY_CUGRAPH_H

#ifdef CUPY_USE_HIP

// Since ROCm/HIP does not have cuGraph, we simply include the stubs here
// to avoid code dup.
#include "stub/cupy_cugraph.h"

#elif !defined(CUPY_NO_CUDA)

// #include <library_types.h>
// #include <cugraph/algorithms.hpp>
#include <algorithms.hpp>

#else

#include "stub/cupy_cugraph.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUGRAPH_H
