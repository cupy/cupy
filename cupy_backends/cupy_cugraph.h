#ifndef INCLUDE_GUARD_CUPY_CUGRAPH_H
#define INCLUDE_GUARD_CUPY_CUGRAPH_H

#ifdef CUPY_USE_HIP

// Since ROCm/HIP does not have cuGraph, we simply include the stubs here
// to avoid code dup.
#include "stub/cupy_cugraph.h"

#elif !defined(CUPY_NO_CUDA)

#include <algorithms.hpp>

// The following header file is available since version 0.19.0
#if __has_include(<version_config.hpp>)
#include <version_config.hpp>
#else
#warning <version_config.hpp> is not found
#define CUGRAPH_VERSION_MAJOR 0
#define CUGRAPH_VERSION_MINOR 0
#define CUGRAPH_VERSION_PATCH 0
#endif

#else

#include "stub/cupy_cugraph.h"

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUGRAPH_H
