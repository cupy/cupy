#pragma once

#ifdef __CUDACC_RTC__

// for using CCCL
#include <cuda/std/type_traits>
#include <cuda/std/limits>
namespace std {
    // TODO(leofang): expose all APIs patched by Jitify for parity
    using cuda::std::conditional;
    using cuda::std::enable_if;
    using cuda::std::false_type;
    using cuda::std::is_array;
    using cuda::std::is_floating_point;
    using cuda::std::is_function;
    using cuda::std::is_integral;
    using cuda::std::is_same;
    using cuda::std::is_signed;
    using cuda::std::is_pointer;
    using cuda::std::is_unsigned;
    using cuda::std::is_volatile;
    using cuda::std::make_signed;
    using cuda::std::make_unsigned;
    using cuda::std::remove_cv;
    using cuda::std::remove_reference;
    using cuda::std::remove_pointer;
    using cuda::std::result_of;
    using cuda::std::true_type;

    using cuda::std::numeric_limits;
}

#endif
