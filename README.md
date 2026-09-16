# numpy for Ascend NPU: forked from Cupy

By Qingfeng Xia

> **Quick Install**（能拿到匹配的 wheel 时，三步）：
> `pip install ./cupy-<ver>-cp3XX-cp3XX-<plat>.cann<X.Y>.whl` →
> `source <cann>/set_env.sh` → `python -c "import cupy"`。
> 完整说明见 **§3 安装指南**（wheel 命名/下载/运行时前置/验证/排错）；
> 若没有覆盖你的 **平台 + CANN 版本**（例如 aarch64）的 wheel，走 **§3.2 从源码编译**，
> 环境搭建（CANN 安装、无 NPU 开发、依赖版本）见 [DeveloperNotes.md](./DeveloperNotes.md)。

## 1. Status of numpy-ascend Array API suport

see [Progress.md](./Progress.md): Array API 标准覆盖 **118/129 = 91.5 %**（未移植含 `eigen`）；
运行时注册算子 **176**（public 149 / inplace 27），`cupy/_core` ufunc 覆盖 110/151 = 72.8 %，
可移植缺口剩 2 个（`i0`、`nextafter`）。算子事实数据由 [tools/scan_ops.py](./tools/scan_ops.py)
自动生成到 [tools/cst_db.md](./tools/cst_db.md)。

> **验证等级**：开发机无 NPU，目前只到 **L3**（Cython 编译 + `import cupy` + 算子注册表对账，
> 见 [Memory.md](./Memory.md) §1）。**数值正确性需在 910B 上跑 `pytest` / `benchmark.py` 验证**，
> 未验证前不要假设与 NumPy 逐位一致。

### 1.1 completed

1. customed kernel (ascend c, triton-ascend python)
2. all cupy major features, except for random (can be done)

### 1.2 limitation
1. float32 only for all array API, similarly, default dtype float32, instead of float64 on CPU
2. float64/int64 support add/substract/mul/div ops
3. sparse array/matrix not supported

### 1.3 pytest 上的 dtype 过滤 (减少假失败)

因为上面 1/2 条的限制, 直接在 NPU 上跑上游 CuPy 测试会因"dtype 本身不被支持"而大量
FAIL (`float64`/`complex64`/`complex128`), 淹没真正的移植缺陷。Ascend 后端会自动把
这些 dtype 从测试参数化中去掉, 只保留 NPU 能跑的用例:

```sh
pytest tests/cupy_tests/math_tests/test_arithmetic.py -q          # 默认: 跳过不支持的 dtype
pytest ... --ascend-dtype-filter=off                              # 跑完整 dtype 矩阵(看真实失败)
CUPY_TEST_ASCEND_SKIP_DTYPES=float64 pytest ...                   # 只跳过 float64, 其余照跑
CUPY_TEST_ASCEND_DTYPE_FILTER=on pytest ...                       # 无 NPU 时模拟 Ascend 行为
```

策略集中在 `cupy/testing/_ascend_dtypes.py` (默认 `auto`: 仅 Ascend 生效),
覆盖 `cupy.testing.for_all_dtypes()` 等装饰器与 `pytest.mark.parametrize` 两类参数化;
单个测试/模块可用 `@pytest.mark.ascend_dtype_filter_off` 豁免。


## 2. Python Array API standard 与替代方案

### 2.1 introduction to Python Array API standard
https://github.com/data-apis/array-api

``` py
np_arr = np.array([1, 2, 3])
xp_np = np_arr.__array_namespace__()
print(xp_np.__name__)  # 通常输出 'numpy.array_api'

cp_arr = cp.array([1, 2, 3])
xp_cp = cp_arr.__array_namespace__()
print(xp_cp.__name__)  # 通常输出 'cupy.array_api'
```

```py
import cupy as cp
# 直接导入 CuPy 的 Array API 模块
import cupy.array_api as cpx

# 使用 cpx 模块中的函数创建数组、执行运算
x_gpu = cpx.asarray([1, 2, 3, 4], device='cuda')  # 显式指定设备
y_gpu = cpx.reshape(x_gpu, (2, 2))
z_gpu = cpx.matmul(y_gpu, y_gpu)
```

### 2.2 torch_npu used as numpy array API
如果不需要兼容现有cupy/numpy代码, 对应新写的数据处理程序和AI数据预处理, 可以直接用torch的array API

```py
import torch
import torch_npu
# 直接导入 torch 的 Array API 模块
import torch._numpy as cp

device = "npu"  # can also "cuda" for torch-cuda
a_xpu = cpx.asarray([1, 2, 3, 4], dtype=cp.int32).tensor.to(device)
# here _numpy wrap/proxy  torch.Tensor into a ndarray class type 
```

## 3. 安装指南 (QuickStart)

产物是 **CANN/昇腾后端版的 CuPy**：`import` 名字与用法沿用 CuPy
（`import cupy as cp` → `cp.ndarray` / `cp.asnumpy()`），差异只在 dtype 支持范围与未移植算子
（见 §1.2 与 [Progress.md](./Progress.md)）。**dist name 仍是 `cupy`**（与官方 CuPy 同名），
所以推荐直接安装本项目的 wheel 文件。

### 3.0 三步快速开始

```sh
# ① 装 wheel（按 平台 + CANN 版本 选对文件，命名规则见 §3.3）
pip install ./cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl

# ② 让 CANN 进入运行时环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# ③ 验证
python -c "import cupy; print(cupy.__version__, cupy.backends.ascend.check_cann_version())"
```

**没有你需要的（平台 + CANN 版本）组合的 wheel 时**（aarch64、CANN 9.1、自编译的 CANN 等），
按 §3.2 从源码编译。环境搭建（CANN 安装、无 NPU 机器上的开发、依赖版本、常见报错）见
[DeveloperNotes.md](./DeveloperNotes.md) §1/§2。

### 3.1 方式 A：安装二进制 wheel（推荐）

**① 选对 wheel** —— 三个字段都要与本机一致（规则见 §3.3）：

| 字段 | 例子 | 怎么确认 |
|---|---|---|
| Python ABI | `cp311` | `python -V`（3.11 → cp311，不能跨 minor 混用） |
| 平台 | `manylinux_2_17_x86_64` | `uname -m`；glibc ≥ 2.17 |
| CANN release | `cann9.0` | 本机 CANN 是 9.0.x（`echo $ASCEND_HOME_PATH`） |

```sh
# 从 release 页下载：https://github.com/qingfengxia/numpy-ascend/releases
# 或直接给 URL（<tag> 换成 release 页上的实际 tag）：
pip install https://github.com/qingfengxia/numpy-ascend/releases/download/<tag>/cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl

# 或者下载后用本地文件安装：
pip install ./cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl

# 若项目发布了 pip 源 / 镜像源，注意 dist name 是 cupy，务必带 index + 版本号，
# 否则会装成上游官方 CuPy：
#   pip install --index-url <项目 index> cupy==14.0.0a1
```

依赖 `numpy>=1.24,<2.6`、`fastrlock>=0.5` 由 pip 自动安装。NumPy 是**纯 Python 层依赖**
（不使用 NumPy C ABI，DLPack 才是零拷贝通道），所以 wheel **不绑定** NumPy 版本，
也**不要**为不同 NumPy 重新编译（详见 [docs/Package.md](./docs/Package.md) §1）。

**② 运行时前置条件**（wheel 不含 CANN SDK，目标机必须具备）：

1. 已安装 CANN toolkit + 算子包，且与 wheel tag 属于**同一 release train**
   （`cann8.5` wheel ⇔ CANN 8.5.x）。release train 一致即可，patch 差异只告警：
   9.0.3 的 CANN 可以跑 `cann9.0` 的 wheel。
2. `source <cann>/set_env.sh`（导出 `LD_LIBRARY_PATH`）。wheel 里**故意不写** CANN 绝对路径
   （`embed_sdk_in_rpath=False` + `DT_RUNPATH`），所以 `set_env.sh` / `LD_LIBRARY_PATH`
   一定生效，wheel 也不会被绑死在构建机上（[docs/Package.md](./docs/Package.md) §2.5）。
3. 目标机 `libstdc++.so.6` 不旧于构建机（conda 里的老 `libstdc++` 会遮蔽系统库 → 报错，
   处理见 §3.5）。
4. CANN 8.5.1 的 `libop_common.so` 有未定义符号（CANN 打包缺陷）：现行版本在
   `cupy/__init__.py` 里用 `ctypes` 预加载 `liboptiling.so` 兜住，**不需要手工 `LD_PRELOAD`**
   （历史方案见 [docs/Package.md](./docs/Package.md) §3.4）。
5. FFT 属于**可选**算子包 `ops-fft`（不在基础 CANN SDK 里）：不装也能 `import cupy`，
   只是 `cupy.fft` 不可用（[docs/ascend/ascend_fft.md](./docs/ascend/ascend_fft.md)）。

### 3.2 方式 B：从源码编译（新平台 / 新 CANN 版本）

什么时候必须自己编译：

| 场景 | 说明 |
|---|---|
| **aarch64**（多数带 NPU 的服务器） | 目前只发布 x86_64 wheel |
| CANN 不是 8.5.x / 9.0.x 的 release train | wheel tag 只覆盖已发布的 train（CANN 10 等需自行构建） |
| 自编译/定制 CANN，或非 manylinux_2_17 的 OS（EulerOS / ModelArts 镜像） | 需在本机工具链下重新构建 |
| 只是想改代码 | 用下面的"就地编译"（editable），比装 wheel 更快 |

**步骤 1：环境准备**（CANN ≥ 8.2，含 `bisheng` 与 aclnn 头文件）

```sh
git clone git@github.com:qingfengxia/numpy-ascend.git
cd numpy-ascend && git checkout ascend      # 注意：工作分支是 ascend，不是 main

pip install "cython>=3.1" fastrlock numpy   # Cython 3.0 不够，需 3.1+；见 DeveloperNotes §1.3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
which bisheng                               # AscendC device compiler，缺了无法编译自定义内核
```

CANN 的安装与切换（8.2 / 8.5 / 9.0、无 NPU 机器的 driver 处理、双版本共存、libstdc++ 坑）
见 [DeveloperNotes.md](./DeveloperNotes.md) §1/§2。

**步骤 2：就地编译**（开发用，增量）

```sh
export CUPY_INSTALL_USE_ASCEND=1            # 编译期选择 ascend 后端（宏 CUPY_USE_ASCEND）
python setup.py build_ext --inplace         # setuptools>=80 下 develop --inplace 已不可用
python -c "import cupy._core"               # L2/L3：能 import 说明链接与算子注册 OK
```

**步骤 3（可选）：打 wheel 给别人用**

```sh
python -m build --wheel
# -> dist/cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
```

CANN 版本在**构建期自动探测**并编码成 `major*100 + minor*10 + patch`（8.5.1 → 851，
9.0.1 → 901），不需要手工传参；wheel 的 platform tag 会自动追加 `cann<major>.<minor>`，
避免不同 CANN 的 wheel 互相覆盖。打包策略/tag/RPATH/检查清单见
[docs/Package.md](./docs/Package.md) §2，构建系统架构见 [install/README.md](./install/README.md)。

**适配新平台 / 新 CANN 的注意点**

1. `CUPY_CANN_VERSION` 是**编译期**常量：`.pyx` 写 `IF CUPY_CANN_VERSION >= 901:`，
   `.h/.cpp` 写 `#if CUPY_CANN_VERSION >= 901`。非 Ascend 构建**不定义**该宏（按 0 求值），
   所以**不要用 `#ifdef`**；两条通道的取值都来自 `AscendBackend.get_version()` 一处，
   改版本编码只需改 `check_cann_version()`（[DeveloperNotes.md](./DeveloperNotes.md) §3.5）。
2. AscendC 自定义内核的 include 目录按**宿主架构**自动探测：
   `<arch>/asc/include`（full SDK）、`<arch>/ascendc/include/include/basic_api`（AscendC SDK）、
   `<arch>/tikcpp/tikcfw`（`lib/math/*.h`），见 `cupy/backends/ascend/bisheng.py`。
   目标芯片用 `CUPY_ASCEND_SOC` 指定（默认 `Ascend910B4`）。
3. 改了 `.h` / `.pxd` 可能不触发重编 → 先 `bash clean_cpp_so_files.sh` 再编译。
4. **无 NPU 的开发机**：需要注释掉 `cupy/backends/backend/api/runtime.pyx` 中的
   `initialize_backend(0)` 才能 `import cupy`（仅本机开发，见 DeveloperNotes.md §1）。
   `import` 通过 ≠ 数值正确，真实运算必须回到 910B（§3.4 验证分级）。
5. 临时关闭自定义 AscendC 内核（排查 bisheng 问题时）：`CUPY_ASCEND_DISABLE_CUSTOM_KERNELS=1`。

### 3.3 支持矩阵与 wheel 命名规则

wheel 名格式：`cupy-<version>-<py>-<abi>-<plat>.cann<major>.<minor>.whl`

| 平台 | glibc | Python | CANN | 现状 |
|---|---|---|---|---|
| linux **x86_64** | ≥ 2.17（manylinux_2_17） | 3.9–3.13 | 8.5.x / 9.0.x | 有预编译 wheel（按 tag 下载） |
| linux **aarch64** | ≥ 2.17 | 3.9–3.13 | 8.5.x / 9.0.x | **需自行编译**（§3.2） |
| OpenEuler / ModelArts（910B 验证机） | 系统 glibc | 3.9 | 8.2 | 需自行编译 |

* CANN 构建下限 = **8.2**（`AscendBackend.minimum_version`）。
* 版本校验只比较 release train（`built//10 == installed//10`）：不一致时给
  `RuntimeWarning` 而不是异常，`CUPY_ASCEND_SKIP_VERSION_CHECK=1` 可静默
  （patch 差异不应让 `import cupy` 失败）。
* 构建信息随 wheel 打包在 `cupy/.data/_wheel.json`，运行时可读：

```python
import cupy
cupy.backends.ascend.get_wheel_metadata()
# {'backend': 'ascend', 'cann_version': 901, 'cann_version_str': '9.0.1',
#  'cann_build_path': '...', 'cupy_version': '14.0.0a1', ...}
cupy.backends.ascend.check_cann_version()      # -> bool（不一致时 warn）
```

### 3.4 安装后验证

```sh
# 1) import + 构建元数据 + 注册表（无需 NPU）
python -c "import cupy; print(cupy.__version__, cupy.backends.ascend.get_wheel_metadata())"
# len() 是"注册条目（op × OpType）"数；去重后的唯一算子名见 Progress.md §3
python -c "from cupy.backends.ascend.api.acl_utils import py_list_acl_ufuncs as f; print(len(f()))"

# 2) 最小数值 smoke test（需 NPU，与本机 NumPy 对拍）
python -c "
import numpy as np, cupy as cp
x = np.random.rand(1000).astype(np.float32)
a = cp.asarray(x)
print(cp.asnumpy(a.sum()), x.sum(), cp.asnumpy(a.mean()))
assert np.allclose(cp.asnumpy(a.sum()), x.sum(), rtol=1e-5)
"

# 3) 回归测试
pytest tests/ascend -q                                   # 无需 NPU：注册表/组合算子/内核解析
pytest tests/cupy_tests -q                               # 需 NPU；默认跳过不支持的 dtype（§1.3）
pytest tests/cupy_tests -q --ascend-dtype-filter=off      # 跑全 dtype 矩阵（看真实失败）

# 4) 性能
python benchmark.py --list                               # 无需 NPU：打印 op×dtype 矩阵
python benchmark.py --csv result.csv                     # 需 NPU
```

**验证分级**（声明结论时必须说清到哪一级，见 [Memory.md](./Memory.md) §1）：

| 级别 | 方法 | 能证明什么 |
|---|---|---|
| L1 | Cython 生成 `.cpp` | `.pyx` 语法 / `IF` 分支 |
| L2 | `build_ext --inplace` 链接成功 | C/C++、头文件、符号 |
| L3 | `import cupy` | 模块加载、import 期算子注册、`py_list_acl_ufuncs()` |
| L4 | 实际数值运算（**需 910B**） | 与 NumPy 的一致性 —— 无 NPU 时任何真实运算都会失败 |

### 3.5 常见问题排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `libop_common.so: undefined symbol: _ZN2ge19GetViewErrorCodeStr...` | CANN 8.5.1 打包缺陷，该符号只有 `liboptiling.so` 提供 | 现行版本已在 `cupy/__init__.py` 内 `ctypes` 预加载；历史方案：`LD_PRELOAD=<cann>/opp/.../liboptiling.so`（Package.md §3.4） |
| `libstdc++.so.6: version 'GLIBCXX_3.4.32' not found` | conda 自带旧 `libstdc++` 遮蔽系统库 | `ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6` |
| `RuntimeWarning: installed CANN version differs ...` | wheel tag 与运行时 CANN 不是同一 train | 换匹配的 wheel；确属 patch 差异可 `CUPY_ASCEND_SKIP_VERSION_CHECK=1` |
| `cannot open shared object file: libcann_ops_fft.so` | FFT 是可选算子包 | 装 `ops-fft` 或 `export ASCEND_OPS_FFT_PATH=~/repos/ops-fft/build` 并加进 `LD_LIBRARY_PATH`；不用 FFT 可忽略 |
| `NotImplementedError: no implementation registered for 'ascend_xxx'` | 该算子未移植（派发按名字 `cupy_x → ascend_x`） | 用 `tools/cst_db.md` / `Progress.md` §4 查缺口；`CUPY_ASCEND_LENIENT_ARGS` 相关见 docs/ascend |
| bisheng 报找不到 `kernel_operator.h` / `lib/math/*.h` | 自定义内核 JIT 的 include 路径不匹配 | 检查 CANN 布局（`<arch>/asc/include` 等）；可先 `CUPY_ASCEND_DISABLE_CUSTOM_KERNELS=1` 绕过 |
| `Compile-time name 'CUPY_CANN_VERSION' not defined` | 手工 `cythonize()` 未传 `compile_time_env` | 用 `python setup.py build_ext --inplace`（自动注入），Memory.md §2.2 有手工传参示例 |
| 改了 `.h` / `.pxd` 却"没生效" | 依赖跟踪只看 `.pyx` | `bash clean_cpp_so_files.sh` 后重新编译 |
| `cupy.show_config()` 抛 `AttributeError: '_UnavailableModule' object has no attribute 'get_build_version'` | `cupyx/_runtime.py` 仍按 CUDA 假设收集信息（CUB 等） | 已知未适配项；用 `get_wheel_metadata()` / `check_cann_version()` / `py_list_acl_ufuncs()` 代替 |
| 报 `EL0003` 或设备相关错误 | 机器上没有 NPU（或 driver 未装） | 编译与 `import` 不需要 NPU；真实运算需要（§3.4 分级） |

## 4. benchmark.py

2025年开发了100小时, 达成MVP (最小功能单元), 测试了matmul, cos, add, 在910B实现了非常客观的加速, 几十到一百的加速.
用法见 [benchmark.py](./benchmark.py)：`--list` 离线打印 op×dtype 矩阵（无需 NPU），
`--csv result.csv` 跑基准（需 NPU）。

但是还是有大量工作, 预计为1人年, 欢迎加入测试和开发.

## 5. 文档索引

| 文档 | 内容 |
|---|---|
| [Progress.md](./Progress.md) | 现状快照：覆盖率、算子数、里程碑、剩余缺口 |
| [Memory.md](./Memory.md) | 开发操作手册：命令速查、架构关键点、验证分级、陷阱清单 |
| [DeveloperNotes.md](./DeveloperNotes.md) | 环境搭建（含无 NPU 开发）、CANN 8.2/8.5/9.0 安装、编译与**条件编译**、FFT |
| [docs/Package.md](./docs/Package.md) | wheel 打包策略（cann tag）、RPATH 策略、运行时硬前置条件 |
| [install/README.md](./install/README.md) | 构建系统架构（features / backends 两层）与算子注册流程 |
| [docs/ascend/](./docs/ascend/) | 自定义 AscendC 内核、FFT、matmul、code review、NumPy/CuPy/PyTorch API 差异 |
| [tools/cst_db.md](./tools/cst_db.md) | 自动生成的算子/覆盖率数据库（`python tools/scan_ops.py`） |
| [Roadmap.md](./Roadmap.md) · [TODO.md](./TODO.md) · [plan.md](./plan.md) | 路线图与任务清单 |
