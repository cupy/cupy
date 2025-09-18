#pragma once

#ifdef __CUDACC_RTC__

/*
 * Note: Every time this file is updated, build_num in jitify.pyx should be bumped.
 */

// for using CCCL with Jitify
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/std/tuple>

namespace std {
    // <type_traits>
    // TODO(leofang): expose all APIs patched by Jitify for parity
    using cuda::std::add_const;
    using cuda::std::add_cv;
    using cuda::std::add_lvalue_reference;
    using cuda::std::add_volatile;
    using cuda::std::alignment_of;
    using cuda::std::conditional;
    using cuda::std::enable_if;
    using cuda::std::false_type;
    using cuda::std::integral_constant;
    using cuda::std::is_array;
    using cuda::std::is_base_of;
    using cuda::std::is_convertible;
    using cuda::std::is_enum;
    using cuda::std::is_floating_point;
    using cuda::std::is_function;
    using cuda::std::is_integral;
    using cuda::std::is_lvalue_reference;
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

    // <limits>
    using cuda::std::numeric_limits;

    // <utility>
    using cuda::std::declval;
#if __cplusplus >= 201402L
    using cuda::std::index_sequence;
    using cuda::std::integer_sequence;
    using cuda::std::make_index_sequence;
    using cuda::std::make_integer_sequence;
#endif
    using cuda::std::make_pair;
    using cuda::std::pair;

    // <tuple>
    using cuda::std::get;
    using cuda::std::tuple;
    using cuda::std::make_tuple;
}

#endif
