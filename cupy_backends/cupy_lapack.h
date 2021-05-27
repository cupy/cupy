#ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUPY_CUSOLVER_H

#include <type_traits>

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "cuda/cupy_cusolver.h"

#elif defined(CUPY_USE_HIP) // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "hip/cupy_rocsolver.h"

#else // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "stub/cupy_cusolver.h"

#endif // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)


/*
 * loop-based batched gesvd (only used on CUDA)
 */
template<typename T1, typename T2>
using gesvd = cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, T1*, int, T2*, T1*, int, T1*, int, T1*, int, T2*, int*);

template<typename T1, typename T2> struct gesvd_func { gesvd<T1, T2> ptr; };
template<> struct gesvd_func<float, float> { gesvd<float, float> ptr = cusolverDnSgesvd; };
template<> struct gesvd_func<double, double> { gesvd<double, double> ptr = cusolverDnDgesvd; };
template<> struct gesvd_func<cuComplex, float> { gesvd<cuComplex, float> ptr = cusolverDnCgesvd; };
template<> struct gesvd_func<cuDoubleComplex, double> { gesvd<cuDoubleComplex, double> ptr = cusolverDnZgesvd; };

template<typename T>
int gesvd_loop(
        intptr_t handle, char jobu, char jobvt, int m, int n, intptr_t a_ptr,
        intptr_t s_ptr, intptr_t u_ptr, intptr_t vt_ptr,
        intptr_t w_ptr, int buffersize, intptr_t info_ptr,
        int batch_size) {
    /*
     * Assumptions:
     * 1. the stream is set prior to calling this function
     * 2. the workspace is reused in the loop
     */

    cusolverStatus_t status;
    int k = (m<n?m:n);
    typedef typename std::conditional<(std::is_same<T, float>::value) || (std::is_same<T, cuComplex>::value), float,
                                      /* double or cuDoubleComplex */ double>::type real_type;
    T* A = reinterpret_cast<T*>(a_ptr);
    real_type* S = reinterpret_cast<real_type*>(s_ptr);
    T* U = reinterpret_cast<T*>(u_ptr);
    T* VT = reinterpret_cast<T*>(vt_ptr);
    T* Work = reinterpret_cast<T*>(w_ptr);
    int* devInfo = reinterpret_cast<int*>(info_ptr);

    // we can't use "if constexpr" to do a compile-time branch selection as it's C++17 only,
    // so we use custom traits instead
    gesvd<T, real_type> func = gesvd_func<T, real_type>().ptr;

    for (int i=0; i<batch_size; i++) {
        // setting rwork to NULL as we don't need it
        status = func(
            reinterpret_cast<cusolverDnHandle_t>(handle), jobu, jobvt, m, n, A, m,
            S, U, m, VT, n, Work, buffersize, NULL, devInfo);
        if (status != 0) break;
        A += m * n;
        S += k;
        U += (jobu=='A' ? m*m : (jobu=='S' ? m*k : /* jobu=='O' or 'N' */ 0));
        VT += (jobvt=='A' ? n*n : (jobvt=='S' ? n*k : /* jobvt=='O' or 'N' */ 0));
        devInfo += 1;
    }
    return status;
}


#endif // #ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
