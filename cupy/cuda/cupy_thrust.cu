#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "cupy_thrust.h"

using namespace thrust;

void cupy::thrust::stable_sort(float *first, float *last) {
    device_ptr<float> dp_first = device_pointer_cast(first);
    device_ptr<float> dp_last  = device_pointer_cast(last);
    stable_sort(dp_first, dp_last);
}
