
# Developer Notes

see also [Package.md](docs/Package.md) for build binary wheel for diff CANN version for manylinux, with minimum version requirement on libstdc++.so version

### `ascend-numpy` architecture from top to bottom

1. numpy api in Python lang
2. cupy._core in Cython lang
3. cupy.xpu: high level backend api in cython lange
4. cupy.backends.backend: abstraction of xpu low level backend api in c lang
5. cupy.backends.ascend: impl in cython/c++

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

### 2.3b FFT: `ops-fft` 是独立算子包, 不属于基础 CANN SDK

FFT 和 BLAS/NNAL 一样是**可选单独安装**的: CANN toolkit 只带 `ascendcl/runtime/opapi*`,
`libcann_ops_fft.so` 来自单独的 `.run` 包 (例如 `cann-910b-ops-fft_9.0.0_linux-x86_64.run`),
也可以从源码构建 (in-tree build 产物在 `build/` 下)。

```bash
# 方式 A: 装官方 .run 包 (会装进 CANN 树)
./cann-910b-ops-fft_9.0.0_linux-x86_64.run --install
# 方式 B: 源码构建 (开发用), 产出 ~/repos/ops-fft/build/libcann_ops_fft.so
cd ~/repos/ops-fft && bash build.sh            # 或 cmake build
# 构建时显式指定位置 (最稳):
export ASCEND_OPS_FFT_PATH=~/repos/ops-fft/build
```

**没装 / 没编译也能正常编译 cupy**, 只是没有 FFT (详见 §3.5b 与 `docs/ascend/ascend_fft.md`)。
注意 `import cupy.backends.ascend.api.aclfft` 还要求 `libcann_ops_fft.so` 在**运行时**
的 loader 路径里 (`LD_LIBRARY_PATH` / `ldconfig`), 因为 CANN 绝对路径**不会**被写进 rpath。

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

### 3.5 按 CANN 版本条件编译（构建期自动探测）✅

**结论：不需要手工传任何参数。** CANN 版本在构建时自动探测，并同时下发到两条通道：

| 通道 | 变量 | 写法 | 作用范围 |
|---|---|---|---|
| Cython 编译期常量 | `CUPY_CANN_VERSION` | `IF CUPY_CANN_VERSION >= 901:` | `.pyx` / `.pxd` |
| C/C++ 预处理宏 | `CUPY_CANN_VERSION` | `#if CUPY_CANN_VERSION >= 901` | `.h` / `.cpp` |

链路（单一来源，不要另起一套）：

```
build.check_cann_version()          # 读 <CANN>/version.cfg | compiler/version.info | opp/version.info
  → 编码成 major*100 + minor*10 + patch      # 8.5.1 → 851, 9.0.1 → 901
  → install/cupy_builder/backends/ascend.py  AscendBackend.get_version()
       get_define_macros()     → -DCUPY_CANN_VERSION=901                  (C/C++)
       get_compile_time_env()  → compile_time_env['CUPY_CANN_VERSION']=901 (Cython IF)
  → 二者分别在 cupy_setup_build.py / _command.py 注入
```

验证（无 NPU 也能做）：

```sh
# 1) 宏是否真的进了编译命令行（touch 一个 pyx 强制重编）
touch cupy/_util.pyx && python setup.py build_ext --inplace 2>&1 | grep -o -- '-DCUPY_CANN_VERSION=[0-9]*'
# 2) python 侧取值
python -c "import sys;sys.path.insert(0,'install');import cupy_builder.install_build as b;\
b.check_cann_version(None,None);v=b.get_cann_version();print(v, b.format_cann_version(v))"
```

写法示例：

```cython
# .pyx / .pxd —— Cython 的编译期 IF，不是 Python 的 if
IF CUPY_CANN_VERSION >= 901:
    result = _use_new_aclnn_op(...)
ELSE:
    result = _use_legacy_path(...)
```

```cpp
// .h / .cpp —— 高版本才存在的算子这样切
#if CUPY_CANN_VERSION >= 900
    // aclnnXxxV2 只在 CANN 9.0+ 提供
    return aclIrregularOpRun(aclnnXxxV2GetWorkspaceSize, aclnnXxxV2, stream, ...);
#else
    return aclIrregularOpRun(aclnnXxxGetWorkspaceSize, aclnnXxx, stream, ...);
#endif
```

注意事项：

1. **必须写成数值比较**：`#if CUPY_CANN_VERSION >= N` / `IF CUPY_CANN_VERSION <= 0`。
   非 Ascend 构建**不定义**该宏，C 预处理里未定义标识符按 0 求值，于是 `>= N` 为假（正确）；
   但 `#ifdef CUPY_CANN_VERSION` 在非 Ascend 构建为假、在"定义为 0"的约定下又为真 ——
   **所以不要用 `#ifdef`**。
2. Cython 的 `IF` 是**编译期**分支：写法像 Python，但只能判断编译期常量（如版本号），
   不能判断运行时值。`_routines_math.pyx` / `_routines_indexing.pyx` /
   `_routines_manipulation.pyx` 里的 `IF CUPY_CANN_VERSION <= 0` 就是用来分离 CUDA/Ascend 代码路径的。
3. 两条通道的值都来自 `AscendBackend.get_version()`，改版本编码（例如将来 CANN 10）
   只需改 `check_cann_version()` 一处。
4. `acl_*.h` **只被 Ascend 构建** include（`acl_utils.pyx` 的 `cdef extern from`），
   所以头文件里用这个宏是安全的；共享给 CUDA 的头文件不要用。
5. 版本是**构建时刻**的 SDK 版本，不是运行时版本：用高版本 CANN 编译的 wheel 拿到低版本
   CANN 上跑会出现 `undefined symbol`。因此 wheel tag 带 `cannX.Y`
   （`AscendBackend.get_wheel_platform_tag()`），并在 import 时校验 `cupy/.data/_wheel.json`。

### 3.5b 可选依赖的优雅降级（以 ops-fft / FFT 为例）✅

`libcann_ops_fft.so` **不属于基础 CANN SDK**（见 §2.3b），所以"没装/没编译"是常态，
三个环节都必须优雅处理：

| 环节 | 机制 | 缺 FFT 时的行为 |
|---|---|---|
| 检测 | `install/cupy_builder/features/ascend_fft.py`：`ASCEND_OPS_FFT_PATH` → CANN 树 → `~/repos/ops-fft/build` → `ldconfig` | `has_ops_fft() == False`，`modules = []` |
| 构建 | `preconfigure_modules()` 对每个 feature 做 compile + **link** 探测（`check_library(libraries=['cann_ops_fft'])`），且 `feature.required = False` | 打印 `ascend_fft: No` + `Cannot link libraries`，**构建继续**、不生成 `aclfft.so`；`CUPY_ENABLE_ACLFFT=1` 改成硬失败，`=0` 完全跳过检测 |
| 运行 | `cupy/fft/_backend.py::get_cufft()` 惰性解析（`lru_cache`），`_fft.py` 每个入口都经过它 | `import cupy` / `import cupy.fft` 正常；调用 FFT 抛 `RuntimeError`，提示装 ops-fft 或设 `ASCEND_OPS_FFT_PATH` |

两个容易踩的点：

1. **`find_ops_fft_lib()` 返回 `None` 有两种含义** —— "没找到"与"在默认 loader 路径里、
   不需要额外 `-L`"。判断"有没有 FFT"必须用 `has_ops_fft()`；混用会导致：库其实可用
   （只装在 `ldconfig` 路径）却被跳过，以及 `CUPY_ENABLE_ACLFFT=1` 误报失败。
   另外"没找到"时仍要把依赖声明进 `feature.libraries`，否则通用探测会在空 library
   列表上平凡通过，配置摘要会误报 `ascend_fft: Yes`。
2. **构建期找到 ≠ 运行期能加载**：`aclfft.so` 只记录 `NEEDED libcann_ops_fft.so.1`，
   而 CANN 绝对路径**故意不写进 rpath**（`embed_sdk_in_rpath = False`，避免 wheel 绑死构建机）。
   运行环境要么把 ops-fft 装进 CANN 树 / `ldconfig`，要么设 `LD_LIBRARY_PATH`；
   否则 `import aclfft` 报 `cannot open shared object file`，随后走上面的优雅路径。
   （开发机当前就是这个状态：编译链接通过，但 loader 找不到 → `get_cufft()` 给出清晰错误。）

回归测试：`tests/ascend/test_fft_optional.py`（8 用例，无 NPU、无 FFT 也能跑）——
检测分支、`CUPY_ENABLE_ACLFFT` 两种取值、默认 loader 路径回归、构建期不产生模块、运行时清晰报错。


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

## FFT (partially done)

see docs/ascend/ notes on FFT; 安装见 §2.3b, 可选依赖的编译/运行期降级机制见 §3.5b,
回归测试见 `tests/ascend/test_fft_optional.py`.

### cann_ops_fft.h , why this file must be copied from fft sdk?

头文件cann_ops_fft.h的实际安装位置不可靠
aclfft.pyx 只依赖头文件里约 10 个函数原型 + 2 个枚举，这是 ops-fft 明确“借鉴 cuFFT”的公共稳定接口，漂移风险很低。
vendored 副本保留了原始 license 头（CANN Open Software License 2.0），合规。