#pragma once

#ifdef __CUDACC_RTC__

// for using CCCL
#include <cuda/std/type_traits>
#include <cuda/std/limits>
namespace std {
    using cuda::std::conditional;
    using cuda::std::is_same;
    using cuda::std::is_pointer;
    using cuda::std::is_volatile;
    using cuda::std::remove_cv;
    using cuda::std::enable_if;
    using cuda::std::is_signed;
    using cuda::std::numeric_limits;
}

#endif
