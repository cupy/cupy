# TODO

## short term todo (Nov 2025)
1. fill_kernel() -> aclnnop_FillScalar  GeneralOp 类型

2. backends -> cupy.backends,  git mv + _features.py
disable _preflight, so ignore Cutensor submodule

3. enable most of math api  (done), aclnnop register (almost done)
matmul (done), dot 

4. BitwiseAddScalar op register, _kernel.pyx need update (done)
pytest

5. reduction kernel, replaced by aclnnop
===
4. concat/pad/reshape op
   numpy_to_acl_dtype ->  numpy_dtype_to_acl_dtype

6. triton-fusion (add data adaptor API)

7. aclBlas integration

8. test build on diff OS,  hardware with float32 support  (310P?)

9. testing (after eanble most numpy API)

10. inplace op :  _kernel.pyx need update

## intermediate (within 6 months)
1. templated ascend kernel JIT
2. compile customer kernel
3. impl missing numpy op for ascend
4. random
5. FFT
6. single node multiple NPU distribution test

## longterm (within one year)

1. multi-node multiple NPU
2. double datatype (float32, int64)
2. sparse matrix


## TODO (详细方案)

[CuPy在AMD ROCm平台上的数组创建问题分析与解决方案 - GitCode博客](https://blog.gitcode.com/d12d5e41c894b5e8803c5e39838f621d.html)



### 自定义算子JIT  ：编译和动态加载**Kernel**

 不清楚CANN 和 triton-ascend  路标

ElementwiseKernel 可以做到类似 CUDA的层面,  就是有些工作量. 

### 随机数 (不紧急)

可以numpy来生成随机数, AsNumpy has impl

rand： `#include <aclnnop/aclnn_rand.h>`  有算子

```python
    HIP_random = {
        'name': 'random',
        'required': True,
        'file': [
            'cupy.random._bit_generator',
            ('cupy.random._generator_api',
             ['cupy/random/cupy_distributions.cu']),
        ],
        'include': [
            'hiprand/hiprand.h',
        ],
        'libraries': [
            # Dependency from cuRAND header files
            'amdhip64',  # was hiprtc and hip_hcc before ROCm 3.8.0
            'hiprand',
        ],
        'check_method': build.check_hip_version,
        'version_method': build.get_hip_version,
    }
```

### FFT 信号处理：价值很大

AsdSip也支持FFT一些算子

### hccl 多卡支持 (rocm 没有做迁移)

CUPY的多卡, 本身需要测试工作量.  

HCCL和NCCL的API兼容性, 粗看很相似. 

### 稠密和稀疏矩阵求解 (cupyx.scipy)

cusparse + cusolver：  aicpu， torch-npu

[我中心在稀疏矩阵乘算子研发中取得新进展--中国科学院计算机网络信息中心](https://cnic.cas.cn/gzdt/202411/t20241120_7442650.html)

```c
#ifdef defined(CUPY_USE_ASCEND)
// not sure if CANN support solver, leave it later
//#include "ascend/cupy_ascend_solver.h"
#include "stub/cupy_cusolver.h"  // gracefully give error message

```

#### TODO:  CUBLAS_COMPUTE ,  cudaDataType 下沉到backend层次

`_numpy_to_backend_dtype()`  不同的backend实现不同的pyx文件

####  TODO: HCCL NCCL, MPI, RCCL 应该可以抽象

sparse, 等ascend缺失的暂不做中性化处理, 直接放入cuda/libs

#### split out cuda enum from backend

这些基本GPU专用, 或者不是核心必须得代码, 拆出到cuda backend去维护
cupy.backends.backend._runtime.pyx 

```python
IF CUPY_CANN_VERSION <= 0:
    # Provide access to constants from Python.
    # TODO(kmaehashi): Deprecate aliases above so that we can just do:
    # from cupy.backends.cuda.api._runtime_enum import *
    # from cupy.backends.cuda.api._device_prop import *
    def _export_enum():
        import sys
        import cupy.backends.backend.api._runtime_enum as _runtime_enum
        this = sys.modules[__name__]
        for key in dir(_runtime_enum):
            if not key.startswith('_'):
                setattr(this, key, getattr(_runtime_enum, key))

    _export_enum()
ELSE:
    # in  the future, add ascend cann enum here
```


### cuTensor： 昇腾CANN对应？

```
#ifdef CUPY_USE_HIP

// Since ROCm/HIP does not have cuTENSOR, we simply include the stubs here
// to avoid code dup.
#include "stub/cupy_cutensor.h"
```



