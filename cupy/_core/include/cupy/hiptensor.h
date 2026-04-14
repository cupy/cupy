#ifndef INCLUDE_GUARD_CUPY_HIPTENSOR_H
#define INCLUDE_GUARD_CUPY_HIPTENSOR_H

#if __has_include(<hiptensor/hiptensor.h>)
#include <hiptensor/hiptensor.h>
#elif __has_include(<hiptensor/hiptensor.hpp>)
#include <hiptensor/hiptensor.hpp>
#elif __has_include(<hiptensor.h>)
#include <hiptensor.h>
#elif __has_include(<hiptensor.hpp>)
#include <hiptensor.hpp>
#else
#error "hipTensor header not found"
#endif

#endif  // INCLUDE_GUARD_CUPY_HIPTENSOR_H
