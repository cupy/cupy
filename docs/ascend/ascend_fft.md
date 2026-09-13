# Ascend FFT（ops-fft / aclfft）集成分析

> 分析对象：`~/repos/ops-fft`（CANN FFT 算子库，产出 `libcann_ops_fft.so`，
> 头文件 `cann_ops_fft.h`，文档 `docs/zh/API_Reference/FFT_*.md`）。
> 目标：评估 CuPy 的 FFT 栈（`cupy.fft` / `cupyx.scipy.fft`）如何接入 Ascend，
> 是否能抽象出中立的 XPU FFT API，以及安装脚本如何探测并编译该可选特性。

---

## 1. CuPy 现有 FFT 栈（CUDA / cuFFT）结构

```
cupy.fft.fft/fftn/rfft/...      cupy/fft/_fft.py        纯 Python：norm 缩放、shape/dtype 预处理、plan 选择
        │
        ▼
Plan1d / PlanNd                 cupy/cuda/cufft.pyx     cuFFT 绑定层（cdef class，管理 plan 句柄、work area、流）
        │  + Plan Cache         cupy/fft/_cache.pyx     按 (device, key) 缓存 plan
        │  + Callbacks          cupy/fft/_callback.pyx  cuFFT callback（JIT 编译，CUDA/Linux 专属）
        ▼
cufftMakePlan1d / cufftExecC2C / ...   libcuFFT
```

关键实现点（本仓库，行号以当前代码为准）：

| 内容 | 位置 |
|---|---|
| 一维 FFT 执行：plan 获取/创建、contiguous 化、norm 缩放 | `cupy/fft/_fft.py:83` `_exec_fft`，norm 缩放 `:196-203` |
| N 维 plan 判定与创建（`cufftMakePlanMany` + strides） | `cupy/fft/_fft.py:298` `_nd_plan_is_possible`、`:307` `_get_cufft_plan_nd` |
| N 维执行（PlanNd 不可用时逐轴退化为 1D） | `cupy/fft/_fft.py:489` `_exec_fftn`，`_default_fft_func :628` |
| cuFFT 绑定：`Plan1d`（`:275`）、work area 用 memory pool（`:344-348`）、`get_current_plan`（`:19`） | `cupy/cuda/cufft.pyx` |
| plan 缓存（依赖设备 id，基本可移植） | `cupy/fft/_cache.pyx` |
| cuFFT callback（nvcc + 静态链接 libcufft，**CUDA 专属，不可移植**） | `cupy/fft/_callback.pyx` |
| CUDA feature 声明：模块 `cupy.cuda.cufft`，链接库 `cufft` | `install/cupy_builder/features/cuda.py:51,120` |
| Ascend feature：`cupy.cuda.cufft` 被注释、`cupy.fft._cache/_callback` 标 TODO | `install/cupy_builder/features/ascend.py:53,92-93` |

重要语义（设计 Ascend 后端时必须保持）：

1. **norm 完全在 Python 层处理**（`_exec_fft` / `_exec_fftn` 内做 `out /= inv_norm` 等缩放）；
   底层库只需提供"cuFFT 语义"（forward 不缩放、backward 乘 N）。
2. 输入会先 `_convert_dtype` → `_cook_shape` → `ascontiguousarray`，**进入 plan 执行前必为 C 连续**。
3. `R2C` 输出长度约定为 `n//2 + 1`，与 aclfft 一致。
4. `_fft.py` 在函数体内 `from cupy.cuda import cufft`（`_fft.py:65,85,330,491`），**import 即绑定 CUDA**；
   Ascend 下 `cupy.cuda.cufft` 未编译，因此当前 Ascend 构建 `cupy.fft` 整体不可用。

---

## 2. ops-fft（aclfft）API 概览

ops-fft 刻意模仿 cuFFT：枚举值与 cuFFT 完全相同
（`ACLFFT_C2C=0x29 / R2C=0x2a / C2R=0x2c / Z2Z=0x69 / D2Z=0x6a / Z2D=0x6c`，
`ACLFFT_FORWARD=-1 / BACKWARD=1`），错误码表也与 `cufftResult` 同构。

### 2.1 已实现接口

| 接口 | 说明 |
|---|---|
| `aclfftCreate / aclfftMakePlan1d / aclfftMakePlan2d` | Plan 生命周期；`aclfftPlan1d/2d` 为组合入口 |
| `aclfftPlan1d(plan, nx, type, batch, dimType)` | 注意多一个 **`dimType`**（HORIZONTAL=0 行变换 / VERTICAL=1 列变换） |
| `aclfftPlan2d(plan, batch, nx, ny, type)` | **参数顺序为 (plan, batch, nx, ny, type)**，仅 C2C |
| `aclfftSetStream(plan, aclrtStream)` | 绑定流（Exec 内部已同步，一般无需手动 sync） |
| `aclfftExecC2C / aclfftExecR2C / aclfftExecC2R` | 执行接口，**输入输出为 Host 指针** |
| `aclfftDestroy / aclfftGetErrorString` | 释放与错误描述 |

### 2.2 与 cuFFT 的关键差异（集成的主要障碍）

| # | 差异 | 对 CuPy 集成的影响 |
|---|---|---|
| 1 | **Exec 的 idata/odata 是 Host 指针**，内部 `aclrtMalloc` + `aclrtMemcpy(H2D)` → kernel → `D2H` → 同步（源码 `src/rfft1_d/arch32/*/*.cpp:100-126` 等可证） | CuPy 数组在 NPU device 上，必须 **device→host→exec→host→device** 共 4 次额外拷贝；且**破坏异步模型**（Exec 返回即计算完成） |
| 2 | 每次 Exec 内部**重新 malloc 设备内存并重新上传 DFT/twiddle 矩阵**，无 plan 级 workspace 复用 | plan 缓存收益有限；性能依赖 ops-fft 后续优化 |
| 3 | **仅 FP32**：`aclfftExecZ2Z/D2Z/Z2D` 声明了但返回 `ACLFFT_NOT_IMPLEMENTED` | `complex128/float64` 输入必须显式报错（不得静默降精度）或走 CPU 回退 |
| 4 | **仅 1D/2D**，`aclfftPlan3d` 返回 `NOT_IMPLEMENTED` | `fftn(≥3 维)` 只能逐轴退化为多次 1D |
| 5 | 2D 仅 910B 支持，且只有 C2C，(nx,ny) ∈ {32,64,128}² 共 9 种组合 | 2D plan 只能作为特例优化，通用路径仍逐轴 1D |
| 6 | `aclfftSetAutoAllocation/SetWorkArea/GetSize*/SetStride/GetVersion/GetProperty` 声明但未实现 | 无法像 cuFFT 那样自管 work area；plan 缓存 key 不能包含 work area |
| 7 | 无 callback、无 PlanMany/高级数据布局（strides） | `cupy.fft._callback` 无法支持；**任意轴/带 stride 的批量变换必须先拷贝成连续** |
| 8 | 归一化固定为 cuFFT BACKWARD 语义（backward 乘 N） | 与 CuPy 底层假设一致，**无需额外处理**（norm 缩放本就在 `_fft.py` 做） |
| 9 | 1D 横向：nx≤2^27 且质因子≤47；nx≥32768 且为 2 的幂时**会改写输入数据** | 执行前必须用 staging 拷贝，恰好规避此坑 |
| 10 | 1D 纵向（VERTICAL）仅 910B C2C：nx 为 2 的幂且 256≤nx≤262144、batch 为 128 的倍数 | 约束过紧，初期不使用，恒用 HORIZONTAL + 连续拷贝 |
| 11 | 按芯片选核：910B C2C 兜底 DFT 支持任意 n≤256；R2C/C2R n≤1024 走 DFT 兜底；950 约束不同（如 950 C2R n≤1024 不支持） | 需要**能力探测/清晰的错误信息**，不能假设所有尺寸可用 |

---

## 3. 集成方案：如何接入 cupy / cupyx

### 3.1 分层设计（沿用本仓库已有模式）

```
cupy.fft / cupyx.scipy.fft                 （纯 Python，不改或最小改动）
        │
        ▼
cupy.fft._backend（新增解析层）             backend → FFT 绑定模块的解析 + 能力声明
        │
   ┌────┴─────────────┐
   ▼                  ▼
cupy.cuda.cufft    cupy.backends.ascend.api.aclfft（新增 .pyx，链接 libcann_ops_fft）
(现有 CUDA 绑定)     暴露 cuFFT 兼容的 Plan1d/Plan2d cdef class
```

### 3.2 新增 Ascend 绑定模块 `cupy/backends/ascend/api/aclfft.pyx`

仿照 `cupy/cuda/cufft.pyx` 暴露同名 API，使上层 `_fft.py` 改动最小：

```cython
cdef extern from "cann_ops_fft.h":
    ctypedef struct aclfftHandle_t
    ctypedef aclfftHandle_t* aclfftHandle
    cdef enum: ACLFFT_C2C, ACLFFT_R2C, ACLFFT_C2R   # 0x29/0x2a/0x2c
    cdef int ACLFFT_FORWARD   # -1
    cdef int ACLFFT_BACKWARD  # 1
    cdef int ACLFFT_HORIZONTAL  # 0
    aclfftResult aclfftPlan1d(aclfftHandle*, int, aclfftType, int, int)
    aclfftResult aclfftPlan2d(aclfftHandle*, int, int, int, aclfftType)
    aclfftResult aclfftSetStream(aclfftHandle, aclrtStream)
    aclfftResult aclfftExecC2C(aclfftHandle, aclfftComplex*, aclfftComplex*, int)
    aclfftResult aclfftExecR2C(aclfftHandle, aclfftReal*, aclfftComplex*)
    aclfftResult aclfftExecC2R(aclfftHandle, aclfftComplex*, aclfftReal*)
    aclfftResult aclfftDestroy(aclfftHandle)
    const char* aclfftGetErrorString(aclfftResult)
```

`Plan1d.fft()` 的数据通路（核心适配点 —— Host 指针语义）：

```
in_dev (NPU) --D2H(pinned)--> h_in --aclfftExec*--> h_out --H2D--> out_dev (NPU)
```

实现要点：
1. **staging 缓冲用 page-locked (pinned) host 内存**（`aclrtMallocHost`），按 plan 尺寸缓存复用，
   同时天然规避了"2^k≥32768 时改写输入"的坑（改写发生在 host 拷贝上）。
2. plan 创建后 `aclfftSetStream(plan, <cupy 当前 stream>)`；由于 Exec 内部已同步，
   返回后无需 `aclrtSynchronizeStream`。
3. `R2C`/`C2R` 的 host 缓冲长度按 `nx/2+1` 分配，与 cufft 布局一致。
4. 提供与 `cufft.Plan1d` 相同的构造签名 `Plan1d(nx, fft_type, batch)`（忽略 `devices`），
   以及 `get_current_plan()` 桩（返回 None）、`getVersion()` 桩、错误码 dict → `CuFFTError` 等价物。
5. `Plan2d(nx, ny, fft_type)`：C2C/FP32 且 (nx,ny) 支持时可用；否则创建期即抛
   `RuntimeError`，让上层退化为逐轴 1D。

### 3.3 上层接入 `cupy/fft/_fft.py`（最小改动）

新增 `cupy/fft/_backend.py` 解析层（或复用 `cupy/backends` 已有的 alias 机制）：

```python
# cupy/fft/_backend.py
try:
    from cupy.backends.ascend.api import aclfft as _impl   # Ascend 构建
except ImportError:
    from cupy.cuda import cufft as _impl                    # CUDA/ROCm 构建

# 能力声明（供 _fft.py 判断）
supports_nd_plan = False      # 无 PlanMany/strides
supports_callbacks = False
supported_dtypes = (np.float32, np.complex64)   # aclfft 仅 FP32
```

`_fft.py` 改动点：
1. 将 `from cupy.cuda import cufft` 替换为 `from cupy.fft._backend import cufft`（`:65,85,330,491` 四处）。
2. `_default_fft_func`（`:628`）：Ascend 下恒返回 `_fft`（逐轴 1D），除非 `fft2` 命中 Plan2d 支持集——
   因为 `_nd_plan_is_possible` / `PlanNd` 依赖 `cufftMakePlanMany`，aclfft 无对应物。
3. `_convert_dtype`（`:34`）：接入 Ascend 能力时，`float64/complex128` 抛
   `NotImplementedError('aclfft only supports float32/complex64')`
   （**不静默降精度**；CPU 回退可作为后续选项）。
4. plan 缓存：`cupy.fft._cache.pyx` 本身可移植（key 里含 `devices` 即可传 None），
   在 `features/ascend.py` 中取消 `cupy.fft._cache` 的注释即可；初期也可以绕过缓存
   （aclfft 每次 Exec 都重建 workspace，缓存收益有限）。
5. `cupyx.scipy.fft`（`cupyx/scipy/fft/_fft.py`）只是对 `cupy.fft` 的再包装，自动跟随，无需改动。

### 3.4 覆盖范围（第一阶段）

| cupy API | Ascend 路径 | 备注 |
|---|---|---|
| `fft/ifft`（complex64） | aclfftPlan1d C2C + HORIZONTAL | 任意 batch；n 覆盖范围见 ops-fft 文档（DFT 兜底 ≤256，mixed-radix ≤2^27） |
| `fft/ifft`（complex128） | 抛 `NotImplementedError` | 无双精度 |
| `rfft`（float32） | aclfftPlan1d R2C | 910B：n≤1024 任意，>1024 需质因子⊆{2,3,5,7,11,…,47} |
| `irfft`（complex64） | aclfftPlan1d C2R | 同上；norm 由 Python 层缩放 |
| `hfft/ihfft` | 复用 R2C/C2R 组合 | 纯 Python 组合，自动可用 |
| `fft2/ifft2`（complex64, 910B, 32/64/128²） | aclfftPlan2d C2C | 特例优化；其余退化逐轴 1D |
| `fftn/rfftn/...` | 逐轴 1D | `_default_fft_func` 强制走 1D 路径 |
| callback / multi-GPU | 不支持 | `set_cufft_callbacks`/`set_cufft_gpus` 报 RuntimeError |

---

## 4. 是否可能抽象出中立的 XPU FFT API？

**结论：可能，但抽象层必须落在"绑定模块 + 能力声明"这一层，而不是把底层 API 形状强行统一。**

可行的中立接口（放 `cupy/backends/backend/api/` 或作为 `cupy.fft._backend` 的契约）：

```python
class FftBackend:                       # 中立契约（草案）
    Plan1d: type                        # (n, fft_type, batch) -> plan, plan.fft(in_dev, out_dev, direction)
    Plan2d: type | None                 # 不支持则为 None
    supports: dict                      # {
        #   'nd_plan': bool,             # 是否支持带 strides 的 N 维 plan（cuFFT: True, aclfft: False）
        #   'dtypes': (float32, complex64),
        #   'callbacks': bool,           # cuFFT: True, aclfft: False
        #   'async_exec': bool,          # cuFFT: True, aclfft: False（Exec 同步）
        #   'io': 'device' | 'host',     # 绑定层内部吞掉差异，上层只见 device 数组
        #   'norm_semantics': 'cufft',   # forward 不缩放 / backward 乘 N
    # }
```

依据：
1. 两库的**枚举值、方向宏、R2C 输出布局、归一化语义**已经一致（ops-fft 明确"借鉴 cuFFT"），
   差异集中在 IO 位置、能力范围与异步性——这些都可以收敛为能力位。
2. 本仓库已有成熟先例：`cupy/backends/backend/api/runtime.pyx` 作为中性运行时层，
   import 时 alias 到 `cupy_backends.cuda.api.*`；`cupy.cuda → cupy.xpu` 的替换也已完成
   （见 `docs/ascend/install_xpu_abstraction_analysis.md`）。FFT 完全可以走同一模式。
3. **不建议**把中立 API 定成 `aclfft` 形状（Host 指针 + 同步）：那会拖累 CUDA 路径；
   正确做法是绑定层各自吞掉 IO 差异，对上层统一暴露 `plan.fft(in_dev_array, out_dev_array, direction)`。

风险与后续：若 ops-fft 未来提供 device 指针 Exec 变体（如 `aclfftExecC2CD`）、
异步语义与 workspace 复用，中立层只需改 Ascend 绑定内部实现，上层无感。建议向 ops-fft 团队提出：
device 指针接口、双精度、3D、`SetWorkArea` 落地、>=32768 的 2^k 不改写输入。

---

## 5. 安装期探测与编译（可选特性）

### 5.1 探测流程（`CUPY_INSTALL_USE_ASCEND=1` 前提下）

ops-fft 以 `.run` 包发布（如 `cann-910b-ops-fft_9.0.0_linux-x86_64.run`），
把 `libcann_ops_fft.so` 装到 CANN 目录的 `lib64`（`OPS_FFT_LIB_INSTALL_DIR`）、
头文件装到 include 目录（`OPS_FFT_INC_INSTALL_DIR`）。探测顺序建议：

```python
# install/cupy_builder/features/ascend_fft.py （草案）
import os, ctypes.util

def find_ops_fft(cann_path: str | None) -> tuple[str | None, str | None]:
    """返回 (lib_dir, include_dir)；找不到返回 (None, None)。"""
    # 1) 显式环境变量优先
    roots = [os.environ.get('ASCEND_OPS_FFT_PATH')]
    # 2) CANN 树内常见位置（.run 默认安装点）
    if cann_path:
        roots += [
            cann_path,                       # <cann>/lib64 + <cann>/include
            os.path.join(cann_path, 'include', 'math_libs'),   # CMake 注释里的 math_libs 子目录
            os.path.join(os.path.dirname(cann_path), 'ops_fft'),
        ]
    lib_names = ['libcann_ops_fft.so', 'libcann_ops_fft.so.1']
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for sub in ('lib64', 'lib', '.'):
            lib_dir = os.path.join(root, sub)
            if any(os.path.isfile(os.path.join(lib_dir, n)) for n in lib_names):
                inc = os.path.join(root, 'include')
                inc = inc if os.path.isfile(os.path.join(inc, 'cann_ops_fft.h')) else root
                return lib_dir, inc
    # 3) ldconfig 兜底（用户自装到 /usr/local/lib 等）
    if ctypes.util.find_library('cann_ops_fft'):
        return None, None   # 让链接器走 -lcann_ops_fft 默认搜索路径
    return None, None
```

附加开关：

- `CUPY_ENABLE_ACLFFT=1/0`：强制开/关；默认 auto（找到即开）。
- 即使编译期链接成功，**运行期仍可能遇到 `ACLFFT_NOT_IMPLEMENTED`**（尺寸/芯片不支持），
  因此绑定模块要在首次调用时把错误码翻译成可读的
  `RuntimeError(aclfftGetErrorString(...))`，并在文档中列出支持矩阵。

### 5.2 构建接线（feature 机制）

在 `install/cupy_builder/features/ascend.py` 基础上二选一（推荐 a，独立 Feature 更干净）：

```python
# (a) 新建独立可选 Feature：install/cupy_builder/features/ascend_fft.py
class CUPY_ascend_fft(Feature):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.name = 'ascend_fft'
        self.required = False                       # 可选特性
        self.modules = ['cupy.backends.ascend.api.aclfft']
        self.libraries = ['cann_ops_fft']           # 链接 libcann_ops_fft.so
        self.includes = []                          # 探测到的 include dir 加进去
        lib_dir, inc_dir = find_ops_fft(build.get_cann_path())
        if lib_dir is None and not _forced_on():
            self.modules = []                       # 静默降级：不编译 FFT 绑定
        ...
```

```python
# (b) 或并入现有 CUPY_ascend.__init__（features/ascend.py）
if find_ops_fft(...)[0] is not None:
    self.modules.append('cupy.backends.ascend.api.aclfft')
    self.libraries.append('cann_ops_fft')
    # aclfft 可用时一并放开 plan cache（cupy/fft/_cache 可移植）
    ascend_files[ascend_files.index('# \'cupy.fft._cache\',  # TODO')] \
        = 'cupy.fft._cache'
```

同时在 feature registry（`install/cupy_builder/features/__init__.py` 与
`_context.py` 的 backend 选择逻辑，`CUPY_INSTALL_USE_ASCEND` 见 `_context.py:70`、
`backends/ascend.py:34-50 env_flag`）中注册新 Feature，使其仅在
`use_ascend=True` 时参与探测。

### 5.3 运行期条件导入

`cupy/fft/_backend.py` 的 try-import 已天然支持"未编译即无 FFT"：
Ascend 构建但未启用 aclfft 时，`import cupy.fft` 应保持可用但调用即报错
（`_backend` 里放一个抛 `RuntimeError('FFT support not built; install ops-fft and rebuild')`
的替身模块），避免 `import cupy` 失败。

### 5.4 构建/验证步骤

```bash
export CUPY_INSTALL_USE_ASCEND=1
export CUPY_ENABLE_ACLFFT=1            # 可选，默认自动探测
python setup.py build_ext --inplace 2>&1 | grep -iE "error:|undefined reference"

# L2：链接检查
python -c "from cupy.backends.ascend.api import aclfft; print(aclfft.getVersion())"

# L3：import 与 API 可见性（无 NPU 亦可验证模块加载）
export LD_PRELOAD=~/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so
python -c "import cupy.fft; print(cupy.fft.fft)"
```

---

## 6. 实施路线图

| 阶段 | 内容 | 验证级别 |
|---|---|---|
| P0 | 绑定模块 `aclfft.pyx`（Plan1d + host staging + 错误翻译）+ feature 探测/接线 | L1/L2（编译链接） |
| P1 | `cupy/fft/_backend.py` 解析层；`_fft.py` 四处 import 切换；FP32 1D C2C/R2C/C2R | L3（registry/模块加载），有 NPU 后 L4 |
| P2 | `cupy.fft._cache` 解锁；`fft2` 910B 特例；`_convert_dtype` 精度守卫 | L3/L4 |
| P3 | 中立 FFT 契约沉淀到 `cupy/backends/backend/api/`（能力位版）；向 ops-fft 提出 device-pointer/双精度/3D 需求 | — |

## 7. 参考

- ops-fft 文档：`~/repos/ops-fft/docs/zh/API_Reference/{FFT公共接口,FFT_1D,FFT_2D}.md`
- ops-fft 头文件：`~/repos/ops-fft/include/cann_ops_fft.h`；Exec 语义源码：`~/repos/ops-fft/lib/fft_exec_api.cpp`、`src/rfft1_d/arch32/*`
- CuPy 侧：`cupy/fft/_fft.py`、`cupy/cuda/cufft.pyx`、`cupy/fft/_cache.pyx`、`cupy/fft/_callback.pyx`
- 构建系统：`install/cupy_builder/features/{cuda,ascend}.py`、`install/cupy_builder/install_build.py`（`get_cann_path` :60、`check_cann_version` :390）、`install/cupy_builder/_context.py:70`
- 相关既有分析：`docs/ascend/install_xpu_abstraction_analysis.md`
