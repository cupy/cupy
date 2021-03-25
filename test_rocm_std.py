#include <hip/hip_runtime.h>
#include <thrust/complex.h>
//#include <limits>


namespace std {
template <>
class numeric_limits<thrust::complex<float>> {
  public:
    static __host__ __device__ thrust::complex<float> max() noexcept {
        return thrust::complex<float>(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }

    static __host__ __device__ thrust::complex<float> lowest() noexcept {
        return thrust::complex<float>(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    }
};
}  // namespace std


int main() {
  thrust::complex<float> a = std::numeric_limits<thrust::complex<float>>::max();
  return 0;
}
