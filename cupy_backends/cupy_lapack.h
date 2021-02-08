#ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUPY_CUSOLVER_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "cuda/cupy_cusolver.h"

#elif defined(CUPY_USE_HIP) // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "hip/cupy_rocsolver.h"

#else // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "stub/cupy_cusolver.h"

#endif // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)



template<typename T1, typename T2>
using gesvd = cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, T1*, int, T2*, T1*, int, T1*, int, T1*, int, T2*, int*);

template<typename T1, typename T2>
struct gesvd_func {
    gesvd<T1, T2> ptr;
};

template<>
struct gesvd_func<float, float> {
    gesvd<float, float> ptr = cusolverDnSgesvd;
};

template<>
struct gesvd_func<double, double> {
    gesvd<double, double> ptr = cusolverDnDgesvd;
};

template<>
struct gesvd_func<cuComplex, float> {
    gesvd<cuComplex, float> ptr = cusolverDnCgesvd;
};

template<>
struct gesvd_func<cuDoubleComplex, double> {
    gesvd<cuDoubleComplex, double> ptr = cusolverDnZgesvd;
};

template<typename T>
int gesvd_loop(
        intptr_t handle, char jobu, char jobvt, int m, int n, T* A,
        int lda, intptr_t s_ptr, intptr_t u_ptr, int ldu, intptr_t vt_ptr,
        int ldvt, intptr_t w_ptr, int buffersize, intptr_t info_ptr,
        int batch_size) {
    cusolverStatus_t status;
    int k = (m<n?m:n);
    typedef typename std::conditional<(std::is_same<T, float>::value) || (std::is_same<T, cuComplex>::value), float,
                                      /* double or cuDoubleComplex */ double>::type real_type;
    real_type* S = reinterpret_cast<real_type*>(s_ptr);
    T* U = reinterpret_cast<T*>(u_ptr);
    T* VT = reinterpret_cast<T*>(vt_ptr);
    T* Work = reinterpret_cast<T*>(w_ptr);
    int* devInfo = reinterpret_cast<int*>(info_ptr);

    // it's too bad that we can't use if constexpr from C++17 to do a compile-time if,
    // so we use custom traits instead
    // cusolverStatus_t (*gesvd)(cusolverDnHandle_t, signed char, signed char, int, int, T*, int, real_type*, T*, int, T*, int, T*, int, real_type*, int*);
    gesvd<T, real_type> func = gesvd_func<T, real_type>().ptr;

    for (int i=0; i<batch_size; i++) {
        // setting rwork to NULL as we don't need it
        status = func(
            reinterpret_cast<cusolverDnHandle_t>(handle), jobu, jobvt, m, n, A, lda,
            S, U, ldu, VT, ldvt, Work, buffersize, NULL, devInfo);
        if (status != 0) break;
        A += m * n;
        S += k;
        U += m * m;
        VT += n * n;
        devInfo += 1;
    }
    return status;
}


#endif // #ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
