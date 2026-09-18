<div align="center">

### numpy for Ascend NPU and GPU: fork from and compatible with Cupy

By Qingfeng Xia

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![CANN](https://img.shields.io/badge/CANN-8.2RC1+-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Ascend%20910B-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-95%25-orange)]()


[API Docs](https://numpy.org/doc/2.4/reference/index.html) | [Developer Docs](docs/ascend/DeveloperNotes.md) |  [Benchmark](benchmark.py) | [Architecture](docs/image/numpy-xpu-architecture.jpg) | [Issues](https://github.com/qingfengxia/numpy-ascend/issues) 

</div>

## Features

1. default dtype float32 (same as cupy on GPU), to enjoy 10X-200X NPU acceleration for math algo
2. float64 is supported although no acceleration benefit
3. int64 and int32 has hardware acceleration for add/substract/mul/div ops
4. API compatible with Cupy(GPU), also mostly compatible with Numpy (CPU), share the cupy and scipy ecosystem


## 1. Status of numpy-ascend

### 1.0 API coverage
see [Progress.md](./Progress.md): 
129个Array API 标准覆盖98% , 缺 2 个（`i0` numpy中建议用scipy.special下面的那个、`nextafter`）。
cupy顶层包的480的API, 覆盖97%, 缺 16 个
底层Ascend算子事实数据由 [tools/scan_ops.py](./tools/scan_ops.py), 自动生成到 [tools/cst_db.md](./tools/cst_db.md)。

### 1.1 completed

1. customed kernel (ascend c, triton-ascend python)
2. all cupy major features, such as custom kernel, profiler, stream/device management, except for operator fusion, but you can use triton kernel instead

### 1.2 limitation
1. uint64 is not supported, but int64 is supported with hardware acceleration for addition/multiplication
2. operator fusion
3. sparse array/matrix, random, can be supported but not impl yet (low priority)
4. multiple NPUs


## Examples

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


环境与构建见 [DeveloperNotes.md](./DeveloperNotes.md)，wheel 打包与运行时前置条件见
[docs/ascend/Package.md](./docs/ascend/Package.md)。

### 3.1 安装二进制 wheel（推荐）
没有覆盖你的**平台 + CANN 版本**（如 aarch64）的 wheel 时走 §3.2 从源码编译。

```sh
# ① 选与本机（Python 小版本 + CPU 架构 + CANN release）匹配的 wheel，安装
pip install ./numpy_ascend_cann90-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl

# ② 让 CANN 进入运行时环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# ③ 验证
python -c "import cupy; print(cupy.__version__, cupy.backends.ascend.check_cann_version())"
```

* wheel 命名 `numpy_ascend_cann<XY>-<ver>-cp3XX-cp3XX-<plat>.cann<X.Y>.whl`：`cp311` = 解释器 3.11、
  `manylinux_2_17_x86_64` = 平台/glibc、`cann9.0` = 本机 CANN 的 release train
  （8.5.x → `cann8.5`）。下载见 [releases 页](https://github.com/qingfengxia/numpy-ascend/releases)；
* **发行名按后端 + CANN train 区分**：Ascend 构建出的是 `numpy-ascend-cann85` /
  `numpy-ascend-cann90`（CANN 版本探测不到时退化为 `numpy-ascend`），用 CUDA/HIP
  后端构建**同一份源码**时仍是上游的 `cupy`。两种情况下代码里都是 `import cupy`
  —— 与 `scikit-learn`/`sklearn`、`opencv-python`/`cv2` 同样的做法
  （`[project].name` 给默认值，Ascend 由 `setup.py` 的 `_BackendAwareDistribution`
  改名）。因此 Ascend 环境下 `pip uninstall` 要用
  `pip uninstall numpy-ascend-cann90` 这样带 train 的名字，且**不要**与官方
  `cupy`/`cupy-cudaXX` 或另一条 CANN train 装在同一个环境（它们都提供 `cupy`
  包，会互相覆盖文件）；
* **源码包（sdist）与 wheel 不同名、也不带任何 tag**：`numpy_ascend-<ver>.tar.gz`
  一份即可覆盖 Python 3.9–3.13 × 所有 CANN train（打包命令、原理与检查清单见
  [docs/ascend/Package.md](./docs/ascend/Package.md) §2.9）。
  wheel 的字段判定、支持矩阵与版本校验机制见同文件 §2.8。
* wheel **不含 CANN SDK**：目标机需要同 release train 的 CANN + `set_env.sh`，以及 libstdc++
  版本、可选 FFT（`ops-fft`）等硬前置条件；安装/运行常见报错（符号缺陷、CANN 版本告警、
  未注册算子、`show_config()` 未适配等）见 [docs/ascend/Package.md](./docs/ascend/Package.md) §3。

### 3.2 从源码编译（新平台 / 新 CANN 版本）

aarch64、CANN 不在 8.5.x/9.0.x、自编译 CANN，或要改代码时：

```sh
git clone git@github.com:qingfengxia/numpy-ascend.git
cd numpy-ascend && git checkout ascend      # 工作分支是 ascend，不是 main
export CUPY_INSTALL_USE_ASCEND=1
python setup.py build_ext --inplace         # 就地编译（开发用；增量）
python -c "import cupy._core"               # L2/L3 验证：能 import 即链接与算子注册 OK
python -m build --wheel                     # 可选：打 wheel（发行名/platform tag 自动带 cannX.Y）
python -m build --sdist                     # 可选：打源码包（与 CPython/CANN 无关，一份通用）
```

环境搭建（CANN 8.2/8.5/9.0 安装与切换、依赖版本、无 NPU 机器上的开发、新平台/新 CANN 适配、
按 CANN 版本条件编译、构建报错排查）见 [DeveloperNotes.md](./DeveloperNotes.md) §1–§3；
wheel 与 sdist 的打包策略、命名规则和检查清单见
[docs/ascend/Package.md](./docs/ascend/Package.md) §2（sdist 见 §2.9）。

### 3.3 验证

```sh
pytest tests/ascend -q                    # 无需 NPU：注册表 / 组合算子 / 自定义内核解析
pytest tests/cupy_tests -q                # 需 NPU；默认跳过 NPU 不支持的 dtype（§1.3）
pytest tests/cupy_tests -q --ascend-dtype-filter=off    # 跑全 dtype 矩阵（看真实失败）
python benchmark.py --list                # 无需 NPU：打印 op×dtype 矩阵
python benchmark.py --csv result.csv      # 需 NPU

# 最小数值对拍（需 NPU）
python -c "
import numpy as np, cupy as cp
x = np.random.rand(1000).astype(np.float32)
assert np.allclose(cp.asnumpy(cp.asarray(x).sum()), x.sum(), rtol=1e-5); print('ok')
"
```

验证分级：L1 Cython 生成 `.cpp` → L2 链接成功 → L3 `import cupy` + 算子注册对账 →
L4 真实数值运算。无 NPU 的机器只能到 **L3**，**L4 必须去 910B 跑**；
分级定义与"不许越界声明"的约定见 [Memory.md](./Memory.md) §1。


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
