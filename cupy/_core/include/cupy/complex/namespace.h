#if defined(__CUDACC_RTC__) || defined(__HIPCC_RTC__) || ( defined(__HIPCC__) && HIP_VERSION < 50000000 )
#define THRUST_NAMESPACE_BEGIN namespace thrust {
#define THRUST_NAMESPACE_END }
#else
#include <thrust/detail/config.h>
#endif
