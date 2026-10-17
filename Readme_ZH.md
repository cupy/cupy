<div align="center">

### 面向 Ascend NPU 的 numpy：fork 自 CuPy，扩展后端，API 兼容

作者：夏清凤（Qingfeng Xia）

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![CANN](https://img.shields.io/badge/CANN-9.0%20%7C%209.5-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Ascend%20910B-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/pytest%20pass%20rate-90%25-orange)]()

[API 文档](https://numpy.org/doc/2.4/reference/index.html) | [开发文档](docs/ascend/DeveloperNotes.md) | [性能测试](tools/benchmark.py) | [架构图](docs/image/numpy-xpu-architecture.jpg) | [Issues](https://github.com/qingfengxia/numpy-ascend/issues)

</div>

[English](README.md) | [中文](Readme_ZH.md)

**numpy-ascend** 把 CuPy 的 ndarray API 带到华为昇腾（Ascend）NPU 上。项目 fork 自
CuPy（v14 谱系，NumPy 2.x），用基于 CANN `aclnn` 算子库的 Ascend 后端替换 CUDA 后端，
使现有 NumPy/CuPy 代码以及 CuPy/SciPy 生态在 NPU 上以 `import cupy` 方式运行。

## 特性

1. 默认 dtype `float32`（与 GPU 上的 CuPy 一致）——数学计算可获得 10–200 倍 NPU 加速。
2. 支持 `float64` 与 `complex128`，提供**三种处理模式**（见下）：NPU 没有原生双精度
   吞吐，且部分 `aclnn` 算子直接不收 `DOUBLE`/`COMPLEX128`。
3. `int64` 与 `int32` 的加减乘除有硬件加速。
4. 与 CuPy（GPU）API 兼容、与 NumPy（CPU）大部分兼容；兼容 CuPy 与 SciPy 生态。

### float64 / complex128 处理模式

通过环境变量 `CUPY_ASCEND_FLOAT64_MODE` 选择（首次使用时读取一次；旧开关
`CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32=1` 等价于 `float32`）：

| 模式 | 行为 | 精度 | 速度 |
|---|---|---|---|
| `float32`（**默认**） | 在派发层把 `float64`/`complex128` 操作数降档为 `float32`/`complex64`，NPU 计算后把结果转回原 dtype。 | 每次运算损失约 9 位有效数字 | 快（NPU） |
| `cpu` | 整算子拦截：D2H → NumPy 真双精度计算 → H2D。适用于 ufunc、归约及多数 general 算子（不支持的响亮报错，不静默）。 | 精确（与 NumPy 一致） | 慢（每算子两次 PCIe 传输） |
| `off` | 不降档、不回退：无法双精度执行的算子直接报错。 | 精确或报错 | — |

```sh
export CUPY_ASCEND_FLOAT64_MODE=cpu     # 精确双精度，host 计算
export CUPY_ASCEND_FLOAT64_MODE=float32 # 默认：NPU 速度，降档精度
export CUPY_ASCEND_FLOAT64_MODE=off     # 绝不静默丢精度
```

**为什么默认的 `float32` 模式可能不适合你** —— 误差累计问题：

- `float32` 只有 24 位尾数：约 7 位十进制有效数字（`float64` 约 16 位）。每次
  算术运算都会舍入到最近的可表示值，舍入误差**会沿计算过程不断累计**。
- 归约是最坏情形：`N` 个数求和的最坏相对误差约 `N·eps`——对 10^8 个随机样本，
  `float32` 模式求和的相对误差可漂移到 1e-2 量级，而 `cpu` 模式约 1e-16。
  长链式逐元素运算（迭代求解器、`cumsum`、归一化）同样按步复合；超过 2^24
  （约 1.7e7）的整数会完全丢失精度。
- `complex128` 降档为 `complex64`，实部与虚部共用同一 24 位尾数——相对误差
  同时作用于模和相位。
- 经验法则：ML 训练/推理、图像数据等场景，`float32` 模式正是预期的取舍；
  数值验证、金融/科学的大数累计（大数组求和、方差、对残差敏感的特征值/求解器）
  请用 `cpu` 模式（精确，且按进程切换即可）；或者把 `float64` 数据留在 NumPy
  （CPU），只把 `float32` 热点路径放上 NPU。

## 支持矩阵

**仅支持 Linux。** 只要 C++ 运行时足够新（`libstdc++`、glibc >= 2.27），任意 2022年之后Linux
发行版都应可用；测试在 aarch64 Linux 上进行。

测试主要为910B卡, 950暂无资源,但是可以编译

| CANN | x86_64 | aarch64 | 说明 |
|---|---|---|---|
| 9.0 | 二进制 wheel + 源码 | 源码 | 主力测试目标（Ascend 910B） |
| 8.5 | 二进制 wheel + 源码 | 源码 |  主力测试目标（Ascend 910B）|
| 8.2 | 仅源码 | 仅源码 | **测试** |

CANN 8.2测试时可以运行, 但是CANN支持的算子少几个, ops-fft, ops-sparse没法支持CANN 8.2, CANN 9.0是主力开发平台

| Python | 二进制 wheel | 源码编译 |
|---|---|---|
| 3.10 | 有 | 有 |
| 3.11 | 有 | 有 |
| 3.12 | 有 | 有 |

Python3.12编译不是问题, 都可以github workflow生成

- 二进制 wheel 见 [Releases 页面](https://github.com/qingfengxia/numpy-ascend/releases)。
  命名规则：非 PyPI train 为 `numpy_ascend-<ver>+cann<XY>-cp<py>-cp<py>-manylinux_2_28_<arch>.whl`
  （本地版本段 `+cann85` / `+cann90`，与 PyTorch 的 `torch==2.5.1+cu121`
  同一方案）；发布到 PyPI 的 train（当前为 CANN 9.0）为
  `numpy_ascend-<ver>-cp<py>-cp<py>-manylinux_2_28_<arch>.whl`（裸版本号）。
  按解释器（`cp311` = Python 3.11）、CPU 架构与 **CANN release train** 选择。
- wheel **不含** CANN SDK：目标机需要同 release train 的 CANN toolkit 与
  `set_env.sh`；硬性运行时前置条件（libstdc++ 版本、可选 ops-fft 等）见
  [docs/ascend/Package.md](docs/ascend/Package.md)。
- 同一环境只能安装一个提供 `cupy` 的包：不要与官方 `cupy` / `cupy-cudaXX` 或
  另一条 CANN train 同环境安装——文件会互相覆盖。

## 安装

### 1. 二进制 wheel（推荐）

**CANN 9.0.x — 从 PyPI 安装**（主力发布 train）：

```sh
pip install numpy-ascend                # 最新 release
pip install numpy-ascend==14.0.0a1      # 锁定版本
```

**CANN 8.5.x — 从 GitHub Release 安装。** PyPI 拒绝带本地版本段的版本号，
因此只有 9.0 train 发布在 PyPI 上；8.5 train 只作为 release 资产分发。
用 `--find-links` 指向 release 资产页（它接受 HTML 页——GitHub Releases
**不是** PEP 503 索引，所以 `--index-url` 不可用），并显式 pin train：

```sh
pip install numpy-ascend==14.0.0a1+cann85 \
    --find-links https://github.com/qingfengxia/numpy-ascend/releases/expanded_assets/v14.0.0a1
```

每个 release 按（CPython × 架构 × CANN train）提供 wheel，以 `v14.0.0a1`
为例：

| 目标机的 CANN | x86_64 wheel | aarch64 wheel | 分发位置 |
|---|---|---|---|
| 9.0.x | `numpy_ascend-14.0.0a1-cp311-cp311-manylinux_2_28_x86_64.whl` | `numpy_ascend-14.0.0a1-cp311-cp311-manylinux_2_28_aarch64.whl` | PyPI + Release |
| 8.5.x | `numpy_ascend-14.0.0a1+cann85-cp311-cp311-manylinux_2_28_x86_64.whl` | `numpy_ascend-14.0.0a1+cann85-cp311-cp311-manylinux_2_28_aarch64.whl` | 仅 Release |

非 PyPI train 用 PEP 440 **本地版本段**标识 train（与 PyTorch 的
`torch==2.5.1+cu121` 同一方案），可精确 pin。任意 release wheel 也可用
直链或本地文件安装（下载地址格式：
`.../releases/download/<tag>/<wheel 文件名>`）：

```sh
pip install https://github.com/qingfengxia/numpy-ascend/releases/download/v14.0.0a1/numpy_ascend-14.0.0a1+cann85-cp311-cp311-manylinux_2_28_x86_64.whl

# 或先从 Releases 页面下载 wheel，再装本地文件：
pip install ./numpy_ascend-14.0.0a1+cann85-cp311-cp311-manylinux_2_28_x86_64.whl

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

python -c "import cupy; print(cupy.__version__, cupy.backends.ascend.check_cann_version())"
```

> **注意**：使用 `--find-links` 时必须显式 pin train——`==14.0.0a1`
> 也能匹配 `+cann85` 本地版本，而 pip 会优先选版本号更大（带本地段）
> 的那个。9.0 用户请直接从 PyPI 安装，不要加 `--find-links`。

如何选 CANN train：必须与目标机上安装的 CANN **toolkit** 匹配
（`ls /usr/local/Ascend/` 或 `npu-smi info`），而不是 NPU 驱动版本。
在 CANN 9.0 主机上装 `+cann85` wheel（或反之）会在 import 时因 aclnn ABI
不匹配而失败——装匹配的 train（先 `pip uninstall numpy-ascend`，所有 train
共用一个发行版名）。

### 2. 源码编译（新平台 / 新 CANN 版本，或开发）

```sh
git clone git@github.com:qingfengxia/numpy-ascend.git
cd numpy-ascend && git checkout ascend      # 工作分支是 ascend，不是 main
export CUPY_INSTALL_USE_ASCEND=1
python setup.py build_ext --inplace         # 就地增量编译（开发用）
python -c "import cupy._core"               # L2/L3 验证：链接与算子注册 OK
python -m build --wheel                     # 可选：打 wheel（自动探测 cannX.Y tag）
python -m build --sdist                     # 可选：打源码包（一份通用，与 CPython/CANN 无关）
```

环境搭建（CANN 8.2/8.5/9.0 安装与切换、**无 NPU 机器**上的开发、新平台适配、
报错排查）见 [DeveloperNotes.md](docs/ascend/DeveloperNotes.md) §1–§3；wheel 与
sdist 打包策略见 [docs/ascend/Package.md](docs/ascend/Package.md)。

### 3. 验证

```sh
pytest tests/ascend -q                     # 无需 NPU：注册表 / 组合算子 / 自定义内核解析
pytest tests/cupy_tests -q                 # 需 NPU；默认跳过不支持的 dtype
pytest tests/cupy_tests -q --ascend-dtype-filter=off   # 全 dtype 矩阵（看真实失败）
python tools/benchmark.py --list           # 无需 NPU：打印 op x dtype 矩阵
python tools/benchmark.py --csv result.csv # 需 NPU

# 最小数值对拍（需 NPU）
python -c "
import numpy as np, cupy as cp
x = np.random.rand(1000).astype(np.float32)
assert np.allclose(cp.asnumpy(cp.asarray(x).sum()), x.sum(), rtol=1e-5); print('ok')
"
```

验证分级：**L1** Cython 生成 `.cpp` → **L2** 链接成功 → **L3** `import cupy` +
算子注册对账 → **L4** 真机数值运算。无 NPU 的机器只能到 **L3**，**L4 必须上 910B**。

## 项目现状

### API 覆盖率

完整快照见 [Progress.md](Progress.md)：

- **Python Array API 标准覆盖 > 98%**（缺 2 个：`i0` —— NumPy 官方建议用
  `scipy.special.i0` —— 以及 `nextafter`）。
- CuPy 顶层 API 覆盖 > 98%。
- 算子级事实数据由 [tools/scan_ops.py](tools/scan_ops.py) 自动生成到
  [tools/cst_db.md](tools/cst_db.md)。

### 测试结果（Ascend 910B 实测）

| 测试集 | 通过率 |
|---|---|
| `tests/cupy_tests/math_tests`（math_test） | **100%** |
| `tests/cupy_tests` + `tests/ascend` 全量 | **90%** |

失败集中在下述已知限制范围内，而非 math 内核本身。

### 已完成

1. 自定义内核（AscendC，以及 Python 侧的 Triton-Ascend）。
2. CuPy 主要特性：自定义内核、profiler、stream/device 管理——算子融合除外
   （可用 Triton 内核替代）。

### 限制

1. 不支持 `uint64`（`int64` 支持，且加减/乘法有硬件加速）。
2. 算子融合。
3. 稀疏数组/矩阵及部分 `cupy.random` 分布仅部分支持（低优先级）。
4. 暂不支持多 NPU。

## 工具

| 工具 | 用途 |
|---|---|
| [`tools/benchmark.py`](tools/benchmark.py) | 命令行基准测试：NumPy (CPU) 对比 numpy-ascend (NPU) 在 op x dtype 矩阵（unary/binary/reduction/matmul/manipulation/sort）上的加速比。`--list` 无 NPU 打印矩阵；`--category`、`--dtype`、`--repeat`、`--csv`、`--strict` 过滤与报告；退出码 2/3 表示导入/设备错误。 |
| [`tools/numpy_ascend_migration_helper/`](tools/numpy_ascend_migration_helper/) | NumPy/CuPy 源码静态迁移分析器：报告不支持的 API/dtype、语义差异与 CPU fallback 候选（基于 AST，非字符串匹配）；`--fix` 执行 AUTO_SAFE 改写，`--fail-on-error` 用于 CI 门禁。详见其 [README.md](tools/numpy_ascend_migration_helper/README.md)。 |
| [`tools/scan_ops.py`](tools/scan_ops.py) | 扫描已构建后端，生成算子/覆盖率数据库（[tools/cst_db.md](tools/cst_db.md)）。 |

## 示例

### Python Array API 标准

<https://github.com/data-apis/array-api>

```python
import cupy as cp
import cupy.array_api as cpx   # CuPy 的 Array API 命名空间

x_gpu = cpx.asarray([1, 2, 3, 4], device='cuda')  # device 槽位，此处即 NPU
y_gpu = cpx.reshape(x_gpu, (2, 2))
z_gpu = cpx.matmul(y_gpu, y_gpu)
```

如果编写新的数据处理程序、不需要 NumPy/CuPy 兼容，`torch_npu` 的 array API
（`torch._numpy`）是另一个可选方案。

## 文档索引

| 文档 | 内容 |
|---|---|
| [Progress.md](Progress.md) | 现状快照：覆盖率、算子数、里程碑、剩余缺口 |
| [Memory.md](Memory.md) | 开发操作手册：命令速查、架构关键点、验证分级、陷阱清单 |
| [DeveloperNotes.md](docs/ascend/DeveloperNotes.md) | 环境搭建（含无 NPU 开发）、CANN 8.2/8.5/9.0 安装、条件编译、FFT |
| [docs/ascend/Package.md](docs/ascend/Package.md) | wheel 打包策略（cann tag）、RPATH 策略、运行时硬前置条件 |
| [install/README.md](install/README.md) | 构建系统架构（features / backends 两层）与算子注册流程 |
| [docs/ascend/](docs/ascend/) | 自定义 AscendC 内核、FFT、matmul、code review、NumPy/CuPy/PyTorch API 差异 |
| [tools/cst_db.md](tools/cst_db.md) | 自动生成的算子/覆盖率数据库（`python tools/scan_ops.py`） |
| [Roadmap.md](Roadmap.md) · [TODO.md](TODO.md) | 路线图与任务清单 |

## 参与贡献

欢迎在 <https://github.com/qingfengxia/numpy-ascend> 提 Issue 与 PR。修改代码时：
用 `CUPY_INSTALL_USE_ASCEND=1` 重新编译，至少验证到 L3（`pytest tests/ascend -q`），
并如实说明达到的验证级别（见 [Memory.md](Memory.md)）。`ascend` 分支的提交信息
使用 `ASCEND [AI]:` 前缀。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。本项目 fork 自 CuPy；CuPy 本身即为 MIT 许可
（版权人 Preferred Infrastructure, Inc. / Preferred Networks, Inc.），上游版权
声明保留在 [LICENSE](LICENSE) 中。

## 英文文档

English documentation: [README.md](README.md)。
