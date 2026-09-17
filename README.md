# numpy for Ascend NPU: forked from Cupy

By Qingfeng Xia

> **Quick Install**: `pip install cupy-<ver>-cp3XX-cp3XX-<plat>.cann<X.Y>.whl` →
> `source <cann>/set_env.sh` → `python -c "import cupy"`（详见 §3）。
> 没有覆盖你的**平台 + CANN 版本**（如 aarch64）的 wheel 时走 §3.2 从源码编译。

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
所以推荐直接安装本项目的 wheel 文件。开发/打包细节不写在这里：
环境与构建见 [DeveloperNotes.md](./DeveloperNotes.md)，wheel 打包与运行时前置条件见
[docs/ascend/Package.md](./docs/ascend/Package.md)。

### 3.1 安装二进制 wheel（推荐）

```sh
# ① 选与本机（Python 小版本 + CPU 架构 + CANN release）匹配的 wheel，安装
pip install ./cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl

# ② 让 CANN 进入运行时环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# ③ 验证
python -c "import cupy; print(cupy.__version__, cupy.backends.ascend.check_cann_version())"
```

* wheel 命名 `cupy-<ver>-cp3XX-cp3XX-<plat>.cann<X.Y>.whl`：`cp311` = 解释器 3.11、
  `manylinux_2_17_x86_64` = 平台/glibc、`cann9.0` = 本机 CANN 的 release train
  （8.5.x → `cann8.5`）。下载见 [releases 页](https://github.com/qingfengxia/numpy-ascend/releases)；
  三个字段的判定方法、支持矩阵与打包/版本校验机制见
  [docs/ascend/Package.md](./docs/ascend/Package.md) §2.8。
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
python -m build --wheel                     # 可选：打 wheel（platform tag 自动带 cannX.Y）
```

环境搭建（CANN 8.2/8.5/9.0 安装与切换、依赖版本、无 NPU 机器上的开发、新平台/新 CANN 适配、
按 CANN 版本条件编译、构建报错排查）见 [DeveloperNotes.md](./DeveloperNotes.md) §1–§3；
wheel 打包策略与检查清单见 [docs/ascend/Package.md](./docs/ascend/Package.md) §2。

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
