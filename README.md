# numpy for Ascend: fork from Cupy

ascend-numpy's architecture from top to bottom

1. numpy api in Python lang
2. cupy._core in Cython lang
3. cupy.xpu: high level backend api in cython lange
4. cupy.backends.backend: abstraction of xpu low level backend api in c lang
5. cupy.backends.ascend: impl in cython/c++

## status

### progress
Oct 12: MVP for add, cos, matmul, benchmark
Oct 23: benchmark.py 经过xpu重构后, NPU测试可以运行
Nov 08: reduction op such as `sum()` is working, 90% math ops ACLOP supported has been added into numpy-ascend
        UnitTest: `pytest tests/cupy_tests/math_tests/test_arithmetic.py `

TODO:  creation/manipulation/indexing ops

### TODO

#### ascend backends

1. `aclEvent` mapping is may have error to fix, causing memory error
2. multiple NPU intialization yet design/tested

#### dtype
1. `add` (all algorith op) support double vector, int64 vector,  but it is slow, probably done by AICPU
2. matrix: `dot/matmul` support only float32, float16, bfloat
3. `bfloat` is not standard numpy type, so will not be supported
4. `numpy.int64` is long 'l' on POSIX OS, 'q' on Windows?
5. `cupy_scalar_to_acl_scalar(_cupy_scalar s)`
5. cupy scalar operands must be cupy._scalar type
6. if two operands have diff dtype, cupy will do promote_types in `XXXKernel`

#### shape

1. `add` (all algorith op) , does not support shape dim > 1? while these can be done
2. CANN aclnn op kernel inside can deal with broadcast, just as pytorch/numpy

#### 一些说明
currently, only support tensor op tensor, some op support tensor op scalar (aclScalar not python double/int)
1. `power(scalar, tensor)` not supported, need some refactoring, Operand as union of aclTensor* and aclScalar*
2. inplace operator like `add` is working, while not sure it use InplaceOp or ASCEND nonIplanceOp
   inplace and nonInplace may have some diff, the save memory addr self and out  passed to op may lead to some error
3. creation/manipulation/indexing, geneal_ops not registered, not tested
8. masked tensor/ndarray: its possible using kargs, using aclnn op
5. scalar op scalar: numpy/cupy 是不是也不支持这样的操作? 

### 核心op支持情况
#### math ops: 
+ 未注册 clamp, einsum, cbrt, trunc,  round/around, convolve (?),
+ 自己实现: radians (deg2rad), degrees (rad2deg), deg2rad, rad2deg. lcm 
+ missing 数值计算: gradient, interp, trapezoid, diff 缺失
+ missing: frexp, ldexp, fix (Trunc), rint
+ complex numpy ops: angle, conj,  缺少几个ops但是自己实现很简单,  real, complex
+ scan, clip

#### indexing ops
slicing ? working, but it does not use `Slice` aclop
`math.scan()` is a dummy/empty func, no such aclop
aclop has `take, put(InplacePut), slice`, but no `choose`

#### manipulation ops
reshape (Reshape api)
having: roll, permute, flip, repeat, 

#### Logica/bitwise 算子基本都有:  
ACLOP has no numpy op: `_left_shift`, `_left_right`

#### statistics reduction ops: 
已有: median, var, mean, std,  bincount, histgram (histc), 主要是看nan怎么处理, 部分做了注册
缺失" average, quantile,  percentile, vecter op实现难度应该不太大
ptp (Range of values (maximum - minimum) along an axis.) -> Aminmax


### similar ops 需要验证numpy行为是否一致
1. fmin, nanmin, min, amin
2. remainder, fmod, modf
3. rint, round, around
4. dot, matmul, mm, gemm, inner

## 1. 开发环境
没有NPU开发: 需要注释掉 runtime.pyx `initialize_backend(0)` 否则不能`import cupy`

### 1.1 Ubuntu 24.04  in WSL2 (无昇腾硬件)

Ubuntu 22.04 似乎才是2025年推荐平台, 主要是python3.12不受支持, 但是通过conda安装得到pyhton3.10, 一样可以安装CANN

 python 3.12 is not supported on CANN 8.2 RC  , so install miniconda-3.10 

```
[Toolkit] [20250912-21:52:11] [ERROR] There is no python3.7,python3.8,python3.9,python3.10,python3.11 in the current environment !
dpkg: error processing package ascend-cann-toolkit (--configure):
 installed ascend-cann-toolkit package post-installation script subprocess returned error exit status 1
Errors were encountered while processing:
 ascend-cann-toolkit
```

开发阶段， 不准备支持windows， 不支持conda， 仅仅支持pip

#### libstdc++.so version issue for conda on Ubuntu 24.04
ImportError: libstdc++.so.6: version `GLIBCXX_3.4.32' not found

```sh
ldd  does not help
strings /home/qingfeng/miniconda3/bin/../lib/libstdc++.so.6 | grep GLIBCXX_3.4
# systemwide version is high enough, g++ use this version to compile
strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4
# while python running using the miniconda 's libstdc++
ln -s /lib/x86_64-linux-gnu/libstdc++.so.6 /home/qingfeng/miniconda3/lib/libstdc++.so.6
```

### 1.2 IDE: vscode 
install extension 
+ Python C++ Debugger （混合debug 不确定对于cython有效）

+ **Cython**: Cython syntax highlighting

+ vscode: cann debugger is under way

### 1.3 ubuntu的C++开发环境安装 (三方依赖库)

```sh
# c++ basic dev environment
apt-get install -y gcc g++ make cmake libsqlite3-dev zlib1g-dev libssl-dev libffi-dev net-tools
# python dependencies
pip3 install attrs cython numpy==1.24 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py cython==3.1
# this package is needed but not documented
pip3 install fastrlock
```

cython 3.0 is not higher enough, use cython 3.1 for string auto conversion

## 2. CANN 安装（社区版）

CANN社区版本是新特性较多的先行版.

[社区版资源下载-资源下载中心-昇腾社区](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)

### 2.1  有硬件NPU:  昇腾driver, CANN toolkit, 算子kernel

安装到用户HOME， 不需要root权限， 如果要运行和benchmark， 需要根据昇腾硬件

```sh
#  install driver, skip here
./Downloads/Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run --install
# add set_env.sh into ~/.bashrc
./Downloads/Ascend-cann-nnal_8.2.RC1_linux-x86_64.run --install
# add set_env.sh into ~/.bashrc
#  install kernel, skip here
```

安装cann-toolkit成功之后, 记得source set_env.sh, 如果有两个CANN版本的话(root, 非root) 会导致后续nnal安装不了.

### 2.2 无NPU (无root权限): 可以开发功能, import 来测试, 但是不能调试

安装driver是必须root权限,  应为没有NPU, 也或者我的Ubuntu24.04 不知支持OS,   `sudo dpkg -i *.deb` 失败.

我就 `dpkg -x *.deb` 把driver解压, copy 里面的driver目录到 `$HOME/Ascend`  同时设置.bashrc环境

`# emulate driver/set_env.sh
export LD_LIBRARY_PATH=$HOME/Ascend/driver/lib64/driver:$HOME/Ascend/driver/lib64/common:$LD_LIBRARY_PATH`

本机没有昇腾卡，driver kernnel需要安装，否则没法`import cupy` 测试cython编译出来so文件是否, 可以导入.   

### 2.3 BLAS: only available with CANN version 8.2 toolkit NNAL

CANN 8.2RC1 （推荐最新稳定版本,只有个8.2 才有FFT和AsdSip的blas函数）， 

 [社区版资源下载-资源下载中心-昇腾社区](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)

| Ascend-cann-nnal_8.2.RC1_linux-x86_64.run    | 加速库软件包 BLAS |
| -------------------------------------------- | ----------------- |
| Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run | runtime           |

安装nnal成功之后, 记得source set_env.sh, 

```
If you want to use asdsip module:
-  To take effect for current user, you can exec command below: source /home/qingfeng/Ascend/nnal/asdsip/set_env.sh or add "source /home/qingfeng/Ascend/nnal/asdsip/set_env.sh" to ~/.bashrc.
```

### 2.4 install triton-ascend, torch-cpu (2.6) 

```bash
# gitee上的torch-npu安装指南 依赖torch cpu 2.6.0
# https://pytorch.org/get-started/locally/  有详细指南
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-npu==2.6.0
# 
pip3 install triton-ascend
```

应该是没有安装ascend driver, `import torch` 会有这个错误, 

> ImportError: libascend_hal.so: cannot open shared object file: No such file or directory, You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.



### 昇腾硬件测试环境

ModelArts EulerOS (对应是OpenEuler 20.03) in docker  CANN 8.2, python 3.9 （华为modelarts 4 910B 服务器）. 

进一步测试install脚本, 被benchmark加速效果

for vector op, `double` is supported, possibly via AICPU, so it is very slow, slower than CPU but can keep data in device memory.

## 3. 编译和安装 numpy-ascend


###  3.1 开发测试阶段

早期开发测试阶段： 建议clone 并修改代码

```sh
git clone git@github.com:qingfengxia/numpy-ascend.git
git checkout ascend
```

从源代码编译 (假设已经安装CANN 8.2)

```bash
cd cupy-ascend
export CUPY_INSTALL_USE_ASCEND=1  # 对应C代码中 CUPY_USE_ASCEND， 编译时刻， 选择ascend backend
#export ASCEND_TOOLKIT_HOME=/home/qingfeng/Ascend/ascend-toolkit/latest
#export PATH=$ASCEND_TOOLKIT_HOME/bin:$PATH
which bisheng

# cython --inplace for gdb debugging
clear && export CUPY_INSTALL_USE_ASCEND=1 && python setup.py develop && python -c "import cupy._core"
python -c "import cupy._core" # to test if it is importable without installation
clear && export CUPY_INSTALL_USE_ASCEND=1 && python setup.py develop && python benchmark.py

```

如果修改 .h 文件, 没有修改pyx文件, 可能导致不会触发编译, 这时候可以运行clean_cpp_so_files.sh 做全面清理. 

### 3.2 基本稳定阶段

```sh
# 第二阶段： 如果测试比较稳定， 可以直接git拉去代码， 编译二进制wheel
pip install  git+https://github.com/qingfengxia/cupy-ascend.git

# 第三阶段： 如果大规模测试通过， 已经有pip二进制包
pip install numpy-ascend
# pip install cupy-cuda12x
```

### 3.3 benchmark.py 代码

开发了100小时, 达成MVP (最小功能单元), 测试了matmul, cos, add, 在910B实现了非常客观的加速, 几十到一百的加速. 
看benchmark.py 

但是还是有大量工作, 预计为1人年, 欢迎加入测试和开发. 

## 4. TODO

see [TODO.md](./TODO.md) for list of task.

还有大量的算子需要加入, 参看如下commit, 有固定的模版添加ASCEND的算子到numpy-ascend, 欢迎测试. 
https://github.com/qingfengxia/numpy-ascend/commit/863e0ff4c07994a45a204b8032db7c3da17f6c90

如果添加代码后, 运行一下命令, 可以编译可以import表示成功.
```sh
export CUPY_INSTALL_USE_ASCEND=1 && python setup.py develop && python -c "import cupy._core"
```

### 4.1 4 kinds of minimum (consider NaN)
only float number can represent NaN (like inf, special value of float)
+ fmin 两个数组的逐元素最小值，忽略 NaN。binary op
+ minimum 两个数组的逐元素最小值，传播 NaN。binary op
+ amin 数组沿给定轴的最小值，传播 NaN。 reduction op
+ nanmin 数组沿给定轴的最小值，忽略 NaN。reduction op

## 5. 架构调整说明 (backend): What has been done after fork

### 5.1 _core/core.pyx 拆分出3个文件, 方便porting

这个已经提交上游社区了 refactor MR, 但是我编译有点问题, 还没有接受

### 5.2 Neutral backend API

#### cuda法律上是禁止二进制的转译

**NVIDIA最终用户许可协议门户**：https://www.nvidia.com/en-us/about-nvidia/eula-agreement/

NVIDIA的EULA**并未禁止重新编译CUDA源代码**

- **合法途径**：像AMD的HIP（HIPIFY工具）和Intel的SYCL（SYCLomatic工具）这类技术，其工作方式是**将CUDA源代码转换为另一种兼容的编程模型代码**，然后使用目标平台自己的编译器和工具链进行编译。这个过程不涉及对CUDA SDK输出成果的逆向或反编译，因此是合规的。
- **核心区别**：关键在于“**转译（Translation）**”与“**移植（Porting）**”的区别。EULA禁止的是对已编译的二进制/PTX代码进行直接转译，但不禁止对源代码进行转换和重新编译。

CUDA的runtime API emulate 可能不违反EULA, 但是没有必要冒着未来的法律, 直接中性化.

#### 中性化重构
+ cupy.cuda -> cupy.xpu                            XPU 泛指任何CPU之外的计算加速器

+ cupy_backends.cuda -> cupy.backends.backend           为什么叫backends, 这是和torch保持移植. 

+ xpuXXX 作为runtime的抽象API

#### 保留Cupy的名称, 致敬Cupy的作者们

同时保持Cupy作者沟通, 确保Cupy是否有商标/著作权, 是否也已授权其他XPU使用. 

架构中性化, 也可以和cupy作者沟通, 看这种架构的重构上游是否可以接受. nvidia在致力做自己官方的pynumeric, 那么社区驱动cupy未来就有不确定性. 

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

#### TODO (refactor underway): cuda backend need xpu -> cuda API mapping
有一个python的脚本(api_replace_tool.py)来负责处理. 
这样处理后, cuda backend有工作量, cuda api -> xpu api, 暂时不能编译. 所以我放在新的分支 xpu开发

cudaDataType  -> xpuDataType 这是一个typedef
cuDoubleComplex -> xpuComplex128,  numpy,torch use this style 
enum xpuFunction_attribute
cuGetErrorString -> 

```
// Context
xpuBlasStatus cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}
```

