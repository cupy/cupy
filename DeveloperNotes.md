
# Developer Notes

see also [Package.md](docs/Package.md) for build binary wheel for diff CANN version for manylinux, with minimum version requirement on libstdc++.so version


## 1. 开发环境
没有NPU开发: 需要注释掉 runtime.pyx `initialize_backend(0)` 否则不能`import cupy`

### 1.1 Ubuntu 24.04 in WSL2 (无昇腾硬件)

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

## 2. CANN 安装（社区版 8.2）

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

### 2.4 install triton-ascend, torch-cpu (2.6) torch-npu

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


## install CANN 8.5 (first开源版本)

### 源代码:  算子库被进一步拆分为四个包
ops-nn
ops-math:  https://gitcode.com/cann/ops-math
ops-cv
ops-transfomer

FFT ops 还是在nnal asdsip
### v8.5 算子二进制包 kernel (安装方法同CANN 8.2)

```py
# cupy_builder will check if runtime dll/so file existing
# $HOME/Ascend/ascend-toolkit/latest/lib64/
["ascendcl", "runtime", "aclnn_ops_train", "aclnn_ops_infer",  "nnopbase","aclnn_math",
    "aclnn_rand", "acl_op_compiler",  "graph", "profapi"]
```

CANN 8.5 has diff so file names
```py
["ascendcl", "runtime", "aopapi_nn", "opapi", "nnopbase","opapi_math",
    "op_common",  "op_compile_adapter", "profapi"]
```

> /home/qingfeng/Ascend/cann-8.5.1/include/aclnn/opdev/op_log.h:19:10: fatal error: base/dlog_pub.h: No such file or directory  
>   19 | #include "base/dlog_pub.h"
>      |          ^~~~~~~~~~~~~~~~~

driver installation problem?  no, including path
`find ~/Ascend/ -name dlog_pub.h`
> ~/Ascend/cann-8.5.1/x86_64-linux/pkg_inc/base/dlog_pub.h

版本判断需要新的方法 `CUPY_CANN_VERSION=820`

ImportError: /home/qingfeng/Ascend/ascend-toolkit/latest/lib64/libaclnn_rand.so: undefined symbol: _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor
也安装了driver版25.3, 可能是编译和运行时后库文件发生了变化(升级)

`nm -D *.so | grep _ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor `

```
find . -name "*.so" -type f -delete
```

```
python -c 'import ctypes; clib = ctypes.CDLL("/home/qingfeng/Ascend/cann-8.5.1/lib64/libop_common.so")'
python -c 'import ctypes; clib = ctypes.CDLL("/home/qingfeng/Ascend/cann-8.5.1/lib64/libgraph.so")'
_ZNK4gert12RuntimeAttrs17GetPointerByIndexEm
```

### v9.0 安装大幅度简化
支持Ascend 950
#### 支持 apt, conda, pip安装

#### 支持FFT
[ops-fft:基于 CANN 的 FFT 类算子库项目 - AtomGit](https://gitcode.com/cann/ops-fft#安装);   

`build.sh --soc=Ascend910B --pkg`  默认是target soc 950
`--full --install-path=/home/qingfeng/miniconda3/envs/aigent/Ascend/cann-9.0.1`

相对cuFFT, 支持cache, 但是不支持callback

### 2.5 昇腾硬件测试环境

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

从源代码编译 (假设已经安装CANN 8.2/8.5)

```bash
cd cupy-ascend
export CUPY_INSTALL_USE_ASCEND=1  # 对应C代码中 CUPY_USE_ASCEND， 编译时刻， 选择ascend backend
#export ASCEND_TOOLKIT_HOME=/home/qingfeng/Ascend/ascend-toolkit/latest
#export PATH=$ASCEND_TOOLKIT_HOME/bin:$PATH
which bisheng

# cython --inplace for gdb debugging
clear && export CUPY_INSTALL_USE_ASCEND=1 && python setup.py develop  --inplace  && python -c "import cupy._core"
python -c "import cupy._core" # to test if it is importable without installation
clear && export CUPY_INSTALL_USE_ASCEND=1 && python setup.py develop && python benchmark.py

```

如果修改 .h 文件, 没有修改pyx文件, 可能导致不会触发编译, 这时候可以运行clean_cpp_so_files.sh 做全面清理. 


##  ascend backends notes

1. `aclEvent` mapping is may have error to fix, causing memory error

### 3.2 dtype
`get_default_dtype()` torch has such API, while cupy/numpy has no such, float64 is the default

0. most ACLOP does not support `double`, `int64` while `+-*/` seems supported 64bit but slow (by CPU?)
1. `add` (all algorith op) support double vector, int64 vector,  but it is slow, probably done by AICPU
2. matrix/linalgo: `dot/matmul` support only float32, float16, bfloat
3. `bfloat` is not standard numpy type, so will not be supported
4. `numpy.int64` is long 'l' on POSIX OS, 'q' on Windows?
5. cupy scalar operands must be `cupy._scalar type`, it may be extended to python scalar in numpy-ascend (TODO)

#### promotion rule
6. if two operands have diff dtype, cupy will do promote_types in `ElementwiseKernel`, how about ascend?

### 3.3  shape
1. `astype()`  involved cast op
https://data-apis.org/array-api/latest/API_specification/index.html
2. CANN aclnn op kernel inside can deal with broadcast, just as pytorch/numpy, while cupy deal with itself not in kernel

### 3.4 notes
currently, only support tensor op tensor, some op support tensor op scalar (aclScalar not python double/int)
1. `power(scalar, tensor)` not supported, need some refactoring, Operand as union of aclTensor* and aclScalar*
2. inplace operator like `add` is working, while not sure it use InplaceOp or ASCEND nonIplanceOp
   inplace and nonInplace may have some diff, the save memory addr self and out  passed to op may lead to some error
3. creation/manipulation/indexing, geneal_ops not registered, not tested
4. masked tensor/ndarray: its possible using kargs, using aclnn op
5. scalar op scalar: numpy/cupy 是不是也不支持这样的操作? 


## 4. TODO

see [TODO.md](./TODO.md) for list of task.

还有大量的算子需要加入, 参看如下commit, 有固定的模版添加ASCEND的算子到numpy-ascend, 欢迎测试. 
https://github.com/qingfengxia/numpy-ascend/commit/863e0ff4c07994a45a204b8032db7c3da17f6c90

如果添加代码后, 运行一下命令, 可以编译可以import表示成功.
```sh
export CUPY_INSTALL_USE_ASCEND=1 && python setup.py develop && python -c "import cupy._core"
```

### 4.1 Four kinds of minimum (consider NaN)
only float number can represent NaN (like inf, special value of float)
+ fmin 两个数组的逐元素最小值，忽略 NaN。binary op
+ minimum 两个数组的逐元素最小值，传播 NaN。binary op
+ amin 数组沿给定轴的最小值，传播 NaN。 reduction op
+ nanmin 数组沿给定轴的最小值，忽略 NaN。reduction op

## FFT
### cann_ops_fft.h , why this file must be copied from fft sdk?

头文件cann_ops_fft.h的实际安装位置不可靠
aclfft.pyx 只依赖头文件里约 10 个函数原型 + 2 个枚举，这是 ops-fft 明确“借鉴 cuFFT”的公共稳定接口，漂移风险很低。
vendored 副本保留了原始 license 头（CANN Open Software License 2.0），合规。