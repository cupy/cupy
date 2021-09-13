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


#if !defined(CUPY_USE_HIP)
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


/*
 * loop-based batched geqrf (only used on CUDA)
 */
template<typename T>
using geqrf = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, T*, int, T*, T*, int, int*);

template<typename T> struct geqrf_func { geqrf<T> ptr; };
template<> struct geqrf_func<float> { geqrf<float> ptr = cusolverDnSgeqrf; };
template<> struct geqrf_func<double> { geqrf<double> ptr = cusolverDnDgeqrf; };
template<> struct geqrf_func<cuComplex> { geqrf<cuComplex> ptr = cusolverDnCgeqrf; };
template<> struct geqrf_func<cuDoubleComplex> { geqrf<cuDoubleComplex> ptr = cusolverDnZgeqrf; };

template<typename T>
int geqrf_loop(
        intptr_t handle, int m, int n, intptr_t a_ptr, int lda,
        intptr_t tau_ptr, intptr_t w_ptr,
        int buffersize, intptr_t info_ptr,
        int batch_size) {
    /*
     * Assumptions:
     * 1. the stream is set prior to calling this function
     * 2. the workspace is reused in the loop
     */

    cusolverStatus_t status;
    int k = (m<n?m:n);
    T* A = reinterpret_cast<T*>(a_ptr);
    T* Tau = reinterpret_cast<T*>(tau_ptr);
    T* Work = reinterpret_cast<T*>(w_ptr);
    int* devInfo = reinterpret_cast<int*>(info_ptr);

    // we can't use "if constexpr" to do a compile-time branch selection as it's C++17 only,
    // so we use custom traits instead
    geqrf<T> func = geqrf_func<T>().ptr;

    for (int i=0; i<batch_size; i++) {
        status = func(reinterpret_cast<cusolverDnHandle_t>(handle),
                      m, n, A, lda, Tau, Work, buffersize, devInfo);
        if (status != 0) break;
        A += m * n;
        Tau += k;
        devInfo += 1;
    }
    return status;
}

#else

template<typename T>
int gesvd_loop(
        intptr_t handle, char jobu, char jobvt, int m, int n, intptr_t a_ptr,
        intptr_t s_ptr, intptr_t u_ptr, intptr_t vt_ptr,
        intptr_t w_ptr, int buffersize, intptr_t info_ptr,
        int batch_size) {
    // we need a dummy stub for HIP as it's not used
    return 0;
}


/*
 * batched geqrf (only used on HIP)
 */
template<typename T>
using geqrf = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, T* const[], int, T*, long int, int);

template<typename T> struct geqrf_func { geqrf<T> ptr; };
template<> struct geqrf_func<float> { geqrf<float> ptr = rocsolver_sgeqrf_batched; };
template<> struct geqrf_func<double> { geqrf<double> ptr = rocsolver_dgeqrf_batched; };
// we need the correct func pointer here, so can't cast!
template<> struct geqrf_func<rocblas_float_complex> { geqrf<rocblas_float_complex> ptr = rocsolver_cgeqrf_batched; };
template<> struct geqrf_func<rocblas_double_complex> { geqrf<rocblas_double_complex> ptr = rocsolver_zgeqrf_batched; };

template<typename T>
int geqrf_loop(
        intptr_t handle, int m, int n, intptr_t a_ptr, int lda,
        intptr_t tau_ptr, intptr_t w_ptr,
        int buffersize, intptr_t info_ptr,
        int batch_size) {
    /*
     * Assumptions:
     * 1. the stream is set prior to calling this function
     * 2. ignore w_ptr, buffersize, and info_ptr as rocSOLVER does not need them
     */

    cusolverStatus_t status;

    // we can't use "if constexpr" to do a compile-time branch selection as it's C++17 only,
    // so we use custom traits instead
    typedef typename std::conditional<
        std::is_floating_point<T>::value,
        T,
        typename std::conditional<std::is_same<T, cuComplex>::value,
                                  rocblas_float_complex,
                                  rocblas_double_complex>::type
        >::type data_type;
    geqrf<data_type> func = geqrf_func<data_type>().ptr;
    data_type* const* A = reinterpret_cast<data_type* const*>(a_ptr);
    data_type* Tau = reinterpret_cast<data_type*>(tau_ptr);
    int k = (m<n)?m:n;

    // use rocSOLVER's batched geqrf
    status = func((cusolverDnHandle_t)handle, m, n, A, lda, Tau, k, batch_size);

    return status;
}
#endif // #if !defined(CUPY_USE_HIP)


/*
 * loop-based batched orgqr (used on both CUDA & HIP)
 */
template<typename T>
using orgqr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, T*, int, const T*, T*, int, int*);

template<typename T> struct orgqr_func { orgqr<T> ptr; };
template<> struct orgqr_func<float> { orgqr<float> ptr = cusolverDnSorgqr; };
template<> struct orgqr_func<double> { orgqr<double> ptr = cusolverDnDorgqr; };
template<> struct orgqr_func<cuComplex> { orgqr<cuComplex> ptr = cusolverDnCungqr; };
template<> struct orgqr_func<cuDoubleComplex> { orgqr<cuDoubleComplex> ptr = cusolverDnZungqr; };

template<typename T>
int orgqr_loop(
        intptr_t handle, int m, int n, int k, intptr_t a_ptr, int lda,
        intptr_t tau_ptr, intptr_t w_ptr,
        int buffersize, intptr_t info_ptr,
        int batch_size, int origin_n) {
    /*
     * Assumptions:
     * 1. the stream is set prior to calling this function
     * 2. the workspace is reused in the loop
     */

    cusolverStatus_t status;
    T* A = reinterpret_cast<T*>(a_ptr);
    const T* Tau = reinterpret_cast<const T*>(tau_ptr);
    T* Work = reinterpret_cast<T*>(w_ptr);
    int* devInfo = reinterpret_cast<int*>(info_ptr);

    // we can't use "if constexpr" to do a compile-time branch selection as it's C++17 only,
    // so we use custom traits instead
    orgqr<T> func = orgqr_func<T>().ptr;

    for (int i=0; i<batch_size; i++) {
        status = func(reinterpret_cast<cusolverDnHandle_t>(handle),
                      m, n, k, A, lda, Tau, Work, buffersize, devInfo);
        if (status != 0) break;
        A += m * origin_n;
        Tau += k;
        devInfo += 1;
    }

    return status;
}
#endif // #ifndef INCLUDE_GUARD_CUPY_CUSOLVER_H
