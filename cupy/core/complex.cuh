template<typename T> __device__ bool isnan(thrust::complex<T> x) {
    return isnan(x.real()) || isnan(x.imag());
}
template<typename T> __device__ bool isinf(thrust::complex<T> x) {
    return isinf(x.real()) || isinf(x.imag());
}
template<typename T> __device__ bool isfinite(thrust::complex<T> x) {
    return isfinite(x.real()) && isfinite(x.imag());
}

// ToDo: assignment operator for thrust::complex<T> = T2 for T2 all types
