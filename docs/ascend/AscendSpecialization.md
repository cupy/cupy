# ASCEND special

> Ascend 后端相对 CUDA 后端的**行为差异**与**适配手段**汇总。
> 每条尽量附 commit，便于回溯。
> 验证等级：**L1** 编译 / **L2** 链接 / **L3** import+注册表 / **L4** 真机数值。

## 0. 差异总览（2026-09-21 ~ 09-25）

| # | 差异类别 | 本质 | 主要适配位置 |
|---|---|---|---|
| A | **dtype 支持面窄** | aclnn 单个算子支持的 dtype 比 CUDA kernel/cuBLAS 少 | reduction：kernel 层提升（`_reduction._call`）；逐算子：C++ 内 cast（`aclop_Any/All`）；elementwise/general：派发层 `_promote_io_dtype` |
| B | **axis / 内存布局语义不同** | aclnn 按真实 ndim + 原 axis 语义，CUDA 靠 permute/合并维 | `_ascend/_reduction.pyx`、`_routines_indexing.pyx`、`_routines_creation.pyx` |
| C | **缺算子/缺参数通道** | CANN 没有对应 aclnn 或没有 3-in/1-out 归约通道 | 组合实现（Python 组合或 C++ general op） |
| D | **派发层/构建边界** | CUDA-only 代码不应进 Ascend 构建；共享代码需后端无关的引用面 | `_core/_ascend/*`、`fusion_stub`、build mapping |

F-order not supported

complex & float64 UT test

---





## A. dtype 差异：aclnn 支持面窄于 CUDA

### A.1 unsigned int 全局提升 ✅ DONE（L4 已验证）

ASCEND does not support unsigned int, so cast to int then cast back to uint, 3 launch_*_func_raw need
to intercept unsigned int dtype by a function `_promote_io_dtype`, then `_cast_back_outs()` before return
migration analyzer will suggest to use int instead to avoid this performance penalty

```python
_ASCEND_DTYPE_PROMOTE = {
    'B': 'i',  # uint8, not sure unsigned char is supported
    'H': 'i',  # uint16 -> int32  # some ops does not support int16
    'I': 'i',
    'Q': 'q'
}

# 豁免拦截的算子：本身原生支持 uint（或作为本机制的实现载体），比如 copy, cast
# 后续发现新的原生支持 uint 的算子，直接往这个 set 里加名字。
_UINT_PROMOTE_EXEMPT_OPS = {
    'ascend_cast',   # 本机制的实现载体（astype / cast-back 都落到它），不豁免会递归
    'ascend_copy',   # aclop_Copy 实为 aclnnCast，原生支持 uint
    'ascend_dump_args',
}

def tuple _promote_io_dtype(sequence ins, sequence outs)
```

**实现（acl_utils.pyx，2026-09）**：

- `_promote_io_dtype(opname, ins, outs)` -> `(p_ins, p_outs, orig_outs, cast_src)`：
  uint 输入 `astype` 成有符号、uint 输出新建有符号临时数组；`orig_outs` 记录
  `[(下标, 原 uint 数组)]`。in-place 约定（outs 为空且算子名含 `inplace`，
  如 `a += b`）时结果写在提升后的 `ins[0]`，`cast_src` 相应取 ins。
- `_cast_back_outs(orig_outs, cast_src, stream_ptr)`：经已注册的
  `ascend_cast` 把有符号临时结果写回调用方原来的 uint 数组。

**2026-09-25 起的通道分工**（`d9cf06b4a`）——提升逻辑跟着调用方走，派发器不再重复：

| 通道 | 提升位置 | 说明 |
|---|---|---|
| elementwise / general（`launch_general_func_raw` / `launch_acl_func_raw`） | acl_utils.pyx 内 `_promote_io_dtype` + `_cast_back_outs`（不变） | 覆盖算术 ufunc、逐元素内核 |
| **reduction**（`launch_reduction_op_raw`） | **已上移**到调用方 `cupy/_core/_ascend/_reduction.pyx::_call`（`_UINT_PROMOTE` 表，与 `_ASCEND_DTYPE_PROMOTE` 保持同步）：uint 输入 astype 提升 → 归约写进提升 dtype 的临时 out → `ret[...] = promoted_out` cast 回（ret 的 dtype 仍由 CUDA loop 类型决定，用户传的 uint out 被正确写回） | 派发器回归"注册表查找 + axes 解析 + launch"零 dtype 逻辑 |
| all/any | **C++ 算子内**（见 A.2），不参与 uint 提升（cast 到 BOOL 语义更直接） | `3e6bef50a` |
- 拦截点：`launch_general_func_raw` / `launch_acl_func_raw` /
  `launch_reduction_op_raw` 三处。general 层拦截放在 fall-through 到窄签名
  路径之后：promote 后 ins/outs 已无 uint，嵌套拦截天然 no-op，cast-back
  责任只属于发起 promote 的那一层。
- 910B NPU pytest 已经验证。

**相关 commits**：
- `f62428fe9` 引入提升 + 写回机制
- `c243b0462` 增加豁免集（**uint cast must skip some ops: ascend_copy and ascend_cast**）
- `7c4533df7` **InplaceFillScalar/Tensor 不支持 uint8/16/32/64**：`core.pyx` 的 fill
  路径改为先 cast 到 int32/int64 填充、再 cast 回 —— 注意 uint64 > 2^63 会截断，
  且每个 fill 多两次 cast，**应尽量避免在 Ascend 上用 uint 数组 fill**。

#### A.1.1 提升表分层：窄整型常开 + float64 降档可选（2026-09-26）

`_promote_io_dtype` 的有效提升表分两层：

| 层 | 表 | 条目 | 语义 |
|---|---|---|---|
| 常开层 | `_ASCEND_DTYPE_PROMOTE` | uint 全系 + `b`(int8)/`h`(int16) → `i`(int32) | 正确性规避：部分算子不收窄整型（aclnnArgMax 只收 FLOAT/FLOAT16）、uint 全系被 aclnn 拒收。值无损，cast-back 后用户可见 dtype 不变 |
| 可选层 | `_FLOAT64_TO_FLOAT32_PROMOTE` | `d`(float64)→`f`、`G`(complex128)→`F`(complex64) | 精度换兼容：910B 无 float64 吞吐且部分算子不收 DOUBLE/COMPLEX128。out dtype 不变（aclnnCast 原生支持 DOUBLE/COMPLEX128，cast-back 保形） |

开关 `enable_float64_to_float32`（默认**关**）：

- 环境变量：`CUPY_ASCEND_ENABLE_FLOAT64_TO_FLOAT32=1`（import 时读入）
- 运行时切换：`cupy.backends.ascend.api.acl_utils.py_enable_float64_to_float32(True/False)`，
  状态查询 `py_is_float64_to_float32_enabled()`

开销：开关打开后每个 float64/complex128 的 elementwise/general 调用多两次
cast kernel（in astype + out cast-back），原生支持 float64 的算子（如
aclnnAmax/Amin/mean/sum）也被无差别提升 —— 豁免集
`_UINT_PROMOTE_EXEMPT_OPS` 只豁免 cast/copy/put/dump_args，其余原生支持
float64 的算子如需规避开销应逐个加豁免。

**reduction 通道的参与范围**：窄整型层**不参与**（aclnnAmax/Amin/mean/sum 原生
支持 int8/int16/DOUBLE）；可选层（float64/complex128 降档）**参与** ——
`_reduction.pyx::_call` 在开关打开时把 `_FLOAT64_DEMOTE` 并入有效提升表，开关
经 cimport 的 `ascend_float64_promote_enabled()` **实时读取**，
`py_enable_float64_to_float32` 的运行时切换对 reduction 同样生效；归约写进
单精度临时 out，再经 `ret[...] = promoted_out`（elementwise copy）cast 回原
float64/complex128 out。uint 提升照旧（`_UINT_PROMOTE`）；all/any 在 C++ 侧
cast 到 BOOL（见 A.2）。

#### A.1.2 CUDA 的 dtype 提升机制（loop 签名 + in-kernel cast，对照）

CUDA 上 `_mean_core` 之类的 reduction 签名表（`cupy/_core/
_routines_statistics.pyx`）**本身就是 cast 规格书**：`'?->d'` 表示 kernel 参数
`in0` 声明为 bool、累加器与输出为 double。cast 发生在**生成的 CUDA kernel
内部**（`cupy/_core/_gpu/_reduction.pyx::_create_reduction_function_code`），
全部是寄存器级转换：

| 阶段 | 生成的代码 | 对 `'?->d'` 的效果 |
|---|---|---|
| load | `${input_expr}` | 按 loop in_type 读元素（`bool in0`） |
| map（cast 点） | `_type_reduce _a = static_cast<_type_reduce>(in0)` | 每元素 bool→double |
| reduce | `_s = REDUCE(_s, _a)`，共享内存 `_sdata` 为 `_type_reduce` | double 累加 |
| post-map/store | `POST_MAP(_s)` → `out0 = a / _type_reduce(n)` | 除法在 double 域，写入 `double out0` |

要点：

* **host 阶段零数据搬运**：`_get_expressions_and_types` 只选 loop 签名，
  不产生临时数组、不追加 kernel。
* `('e->e', (None, None, None, 'float'))` 四元组显式指定 reduce_type='float'：
  half 输入用 float32 累加，最后 store 才转回 half —— 避免 half 累加精度塌陷。
* 成本为 **0 次额外 launch、0 次 global memory 往返** —— cast 编译进了 mean
  kernel 本身。
* 上游缺口：签名表缺 `'b->d'`（int8），上游注释 `# TODO(okuta) needs cast`
  —— int8 的 mean 在 CUDA 上没有 loop 可匹配，直接 TypeError（cast 从未发生）。

与 Ascend 的对照：

| | CUDA | Ascend (aclnn) |
|---|---|---|
| dtype 适配机制 | loop 签名 + kernel 内 `static_cast`（编译期） | aclnn 无 loop/类型映射概念，只认固定 dtype 集 |
| cast 成本 | 0 次额外 kernel，寄存器内 | 显式 `aclnnCast`：额外 launch + 设备 buffer + 一次内存遍历 |
| 失败模式 | 签名表缺 → host TypeError（响亮） | dtype 不收 → 设备侧错误码（EL0003） |

因此 §A.1.1 的 promote 层本质是**在派发层模拟 CUDA 免费获得的 in-kernel
cast**（代价 0 → 2 次 cast kernel）；这也是 `_reduction.pyx::_call` 的
`_UINT_PROMOTE` 刻意只收 uint 的原因 —— int8/int16/float64 在 CUDA 靠签名表
消化、aclnnAmax/mean 原生收下，唯有 uint 两边都没人管，必须显式提升。

### A.2 逐算子 dtype 回退（aclnn 只支持浮点/不支持某些整型）

| 算子 | aclnn 限制 | 适配手段 | commit | 等级 |
|---|---|---|---|---|
| `arange` | 不支持 int8/int16/uint16/uint32 | int32 生成后 `astype` 回目标 dtype（`_ASCEND_ARANGE_INT32_FALLBACK`）；> 2³¹−1 的 uint32 中间态溢出 | `740534c70` | L3 |
| `matmul` | 不支持整型/布尔 | 输入 → float32 计算 → 结果 cast 回；float32 尾数 24 bit，> 2²⁴ 不精确（NumPy 是精确整数运算） | `fd028e60e` | L3 |
| `any` / `all` | 白名单外全被拒（实测 aclnn 稳定只收 BOOL/INT32/INT64/FLOAT16/FLOAT32；DOUBLE/COMPLEX64/128/int8/int16/uint 全系不行）；out 只收 BOOL | **已下沉到 C++**：`aclop_Any/aclop_All` 内 `_run_any_all()` 检查 `aclGetDataType`，白名单外 `aclnnCast` 到 **FLOAT32** 临时张量再归约（输出恒为 BOOL —— numpy 的 any/all 返回恒为 bool，全量归约 `numpy.bool_`、带 axis bool ndarray）。覆盖分工：uint 由 dispatcher（`_reduction._call` `_UINT_PROMOTE`）处理、C++ cast 仅防御；int8/int16 由 C++ 真实处理（dispatcher 刻意不提升窄整型）；DOUBLE cast FLOAT32 有 \|x\|<2⁻¹²⁶ 下溢的 any() 假阴性风险（开关 `enable_float64_to_float32` 打开时 dispatcher 已降档、不走此路由）；**COMPLEX 是已知差距** —— 8.5.1 无 aclnnImag/复数 abs，aclnnCast 输入 doc 未列 complex，纯虚数可能被取实部错判成 0（numpy any(1j)=True），精确语义待上层 real/imag 组合（TODO）。旧的 `_BOOL_CAST_INPUT_OPS` astype('?') 补丁已删除 | `3e6bef50a` + 本条 | L3 |
| `real` | 只接受复数输入 | 实数输入走 `aclnnCast` 恒等拷贝（`aclnn_copy.h` 只有 inplace 版） | `1ae444ac7` | L3 |
| `mean`（整型） | **aclnnMean 求和不会先提升为浮点再除** | method 2：整型输入先 `astype(float)` 再 mean（稳但慢）；method 1（sum 后把标量和转 float）会溢出，未采用 | `d0dcf58b3` | L3 |
| `Max`/`Min` | 只有全量归约，无 dim/keepdim | 改用 `aclnnAmax/Amin`（axes + keepDim），全轴归约由调用方传全部轴 | `20e0ae7a1` | L3 |

### A.3 dtype char / scalar 转换面

- `4e4ba6c1d`：**numpy 2.x 把 uint64 的 dtype char 从 `Q` 改成 `L`** ——
  `numpy_dtype_to_acl_dtype` 需同时接受 `'Q'`/`'L'`，否则 uint64 全部退化成
  `ACL_DT_UNDEFINED`。
- `2d90f7bc2`：`cupy_scalar_to_acl_scalar()` 补 int8（signed char）。
- `1e84f3700`：complex64/128 标量转换验证 + 值打印、清理过期 TODO。
- `e114173ce`：reduction 的 axis 参数通道支持 int / tuple / list / numpy 标量 /
  cupy 标量（此前只认部分形式）。

### A.4 cupy 标量语义（标量操作数的四条规则）

**① 标量方向**（`launch_acl_func_raw`）：`x - 1` 与 `1 - x` 都是「二元 + 标量」
但方向相反 —— 同时记录标量的**位置**，分别派发 `SCALAR_BINARY_OP` /
`REVERSE_SCALAR_BINARY_OP`（旧实现只看「有没有标量」，`2 / x` 被算成 `x / 2`
静默算错）。交换律算子（add/multiply/maximum...）查不到 REVERSE 时回退正向。

**② 标量物化**（M-D1）：CScalar 按 ufunc loop dtype 物化
（`_kernel.pyx` `CScalar.from_numpy_scalar_with_dtype(x, t)`），aclScalar 的
dtype 与 tensor/out 一致，内核内不做隐式 cast。标量无法重建 0-d aclTensor 的
算子（如 `where`）走 `_SCALAR_AS_TENSOR_OPS` 豁免集（见 C 节）。

**③ copyto 的标量必须先转 0-D**（`e65195a08`，详见 D.4）：
`cupy_copy → ascend_copy` 不支持 `REVERSE_SCALAR_BINARY_OP`，标量分支
`src = _core.array(src, dtype=dst.dtype)` 后走 tensor 拷贝。
（`041e37269` 之后 `ascend_copy` 本身已改 GENERAL_OP，`c12735880`。）

**④ 全标量 ufunc 调用 → host 回退**（`041e37269`）：`cupy.add(2, 3)` 这类
**没有任何 ndarray 操作数**的调用 aclnn 无法服务（窄签名通道拒绝
「两标量 + out」；general 通道没有 tensor 可挂载标量，落到
`launch_acl_func_raw` 报 `Invalid number of operands`）。
`_kernel.pyx::ufunc.__call__` 在标记处拦截：

- 条件：`in_args` 全为标量、无 `where=`、`nout == 1`
- 计算：用**同名 numpy ufunc** 在 host 上算（`cupy_add → numpy.add`），
  `dtype` / `casting` 透传；无 numpy 对应的 ufunc 响亮
  `NotImplementedError`
- 返回：`cupy.asarray(result)` —— 0-d cupy.ndarray（CUDA 对齐：CUDA 分配
  0-d out + kernel 写回，返回类型相同）；用户给了 `out` 则 setitem 写回并
  返回 out
- **NEP50 弱标量**：python int/float/complex（`weaks` 标记）先还原成
  python 类型再传给 numpy —— `cupy.add(np.int32(2), 3)` 保持 int32
  （与 CUDA loop 选择 `'ii->i'` 一致），而不是被 np.int64 强类型提升成
  int64

已知边界：`where=` 全标量、`nout > 1`（divmod 类）不走红路径，维持原
派发行为（响亮报错）；数值待 NPU 验证（host 计算本身与设备无关）。

---

## B. axis / 内存布局语义差异

| 差异 | CUDA 侧做法 | Ascend 侧必须这样做 | commit |
|---|---|---|---|
| reduction 的 axis | 把 reduce 轴 permute 到最前面以保 kernel 连续性 | **aclnn 按真实 ndim + 原始 axis 位置**，不需要 permute，直接传 reduce_axis | `0bbe5b98d` |
| C-contiguous 维合并 | `_reduce_dims` 把 (2,3) 合成 (6,) 对 kernel 是优化 | **会破坏 aclnn 的 ndim/dim 语义**（1D 输入 + 原 axis 不匹配）→ 跳过 `_reduce_dims` | `1e242e15d` |
| matmul | cuBLAS column-major → 交换 a/b 的 transpose trick | **aclnnMatmul 就是行主序 A@B**，不交换操作数（否则静默算出转置） | `_ascend/_routines_linalg.pyx` |
| scatter 的 axis | CUDA kernel 用 `a.reduced_view()` 降 2D→1D 后按 cdim/rdim/adim 计算 | `reduced_view()` 会让 `aclnnScatterUpdate` 报 dim 不匹配 → 直接传原始 `a`（1D 分支部分修复，未完全） | `368cfcde5` |
| F-order 张量 | cublas `sgeam` 转置实现 | **F-order 不可用**，临时方案返回 C-contiguous | `8c3eccfab` |
| 视图无法用 offset 表达 | CUDA kernel 直接吃 strides | `_materialize_host()` 用裸 D2H memcpy 物化成独立 C-contiguous 数组；**不能用 `.copy()` / `.get()`**（会经 ufunc 派发回来 → 无限递归） | `64c9d92d7`（修视图越界） |
| `aclnnSort` 的 indices 输出 | 可传 nullptr（不用） | **CANN 8.5 不接受 nullptr indices**，必须给一个真 tensor（返回后即弃） | `f2d7babc7` |

---

## C. 缺算子 / 缺参数通道 → 组合实现

| CANN 缺什么 | Ascend 组合方式 | commit |
|---|---|---|
| `_var_core_*`（ReductionKernel **3-in/1-out**，launch_reduction_op 拒绝） | 方案 1（Python 层）：`d = a - mean; d *= d; sum(d) *= α`，全部走已注册算子；C++ `aclop_VarCore`（general op）保留备用 | `f3c9a08d9` |
| `where(cond, 标量, 标量/数组)`（M3：标量无法重建 0-d aclTensor） | `_SCALAR_AS_TENSOR_OPS` 豁免集 + 派发层把 CScalar → numpy 标量 → `cupy.array` 0-d（H2D memcpy，不经 ufunc） | `0f9e96dbb` |
| `_exists_nan` 归约 kernel | `any(isnan(x))` + `cupy.where`（`_nanmedian`/`_nanmean` 中的分支） | `e697d21fc` |
| `bincount`（上游是 ElementwiseKernel） | `aclop_Bincount` general op（ins=[x] 或 [x,weights]），注册 `ascend_bincount_kernel` / `_with_weight_kernel` | `d447f23f8` |
| `tri`/`tril`/`triu`（上游 = tri kernel + where） | `aclnnTril`/`aclnnTriu`；`tri = tril(ones)`；`tril/triu` 直接走 `_ascend_tri` | `a141b9557` |
| complex 算子的实数输入 | conj→`ascend_copy`、imag→`ascend_fill(0)`、angle→`arctan2(0, x)`（实数分支放在 out-dtype 白名单检查**之前**） | `1ae444ac7` |
| `nan*` 系列的 dtype 分流（重复 switch） | 抽 `dtype_has_no_nan()` 到 `acl_op_template.h`（整型/布尔无 NaN，且 `aclnnNanToNum` 只收浮点） | `0d93f3931` |
| `angle_deg` 的常数 | 常数必须收敛成 Python `float`；强类型 numpy 标量会把结果强推 float64 | `72ba0bb52` |

---

## D. 派发层 / 构建边界

### D.1 `_reduction.pyx` 独立演进 ✅ DONE

两个构建**早已编译不同源文件**（`features/cuda.py` → `_gpu/_reduction.pyx`；
`features/ascend.py` → 原共享文件），所以共享文件里所有
`IF CUPY_CANN_VERSION <= 0` 分支在 Ascend 构建里都是**死代码**。

- `e93719780` / `dfc4c7eed`：`git mv cupy/_core/_reduction.pyx
  cupy/_core/_ascend/_reduction.pyx`，展平全部 `CUPY_CANN_VERSION` 守卫、删除
  CUDA-only 代码（kernel 代码生成、cub、axis permute、memoized function cache），
  1017 → 704 行。
- 保留 backend 中性辅助：`_get_axis` / `_get_out_shape` / `_get_contiguous_size` /
  `_get_block_specs` / `_sort_axis` / `_optimizer_copy_arg` /
  `_set_permuted_args`（最后一个仅因 `_reduction.pxd` 声明而必须提供实现）。
- **`_reduction.pxd` 仍共享**：改签名必须与 `_gpu` 版同步，否则其它模块
  `cimport` 失配。

**关于 include 的可行性（当时的备选）**：Cython 3.3 支持 `include` 且可包在
`IF` 内（已实测两分支都能编译运行），可做「共享主体 + 条件 include backend
`.pxi`」；但 `_gpu/_reduction.pyx` 已与共享版漂移，reconcile 风险由无法验证的
CUDA 构建承担，故选择**方案 A：双文件独立演进**（`_ascend/` 与 `_gpu/`）。
注意 `IF` 语句在 Cython 3.x 已标记 deprecated（#4310）。

```
# 实测可行的条件 include 形态（备查）
IF BACKEND == 1:
    include "ascend_impl.pxi"     # cdef 变量 + def 函数都有效
ELSE:
    include "gpu_impl.pxi"
```

### D.2 fusion ✅ DONE（`82a1761eb`）

`_core/__init__.py` 按后端绑定 `fusion`：CUDA → `_gpu.fusion`；Ascend →
`_ascend.fusion_stub as fusion`（并额外绑 `new_fusion`，满足
`cupy.get_array_module` 里的 `_core.new_fusion._ArrayProxy`）。

**stub 必须覆盖共享代码引用到的完整属性面**（`_is_fusing()` 恒 False 不代表属性
可以不定义 —— 例如 `isinstance` 元组在每次调用时构造）：
`_is_fusing` / `is_fusing` / `fuse` / `_FusionVarArray` / `_FusionVarScalar` /
`_ArrayProxy` / `_ScalarProxy` / `_call_ufunc`（显式 `NotImplementedError`）/
`call_reduction`。检测方式：

```bash
grep -rn "fusion\._\|new_fusion\." cupy/ --include=*.py --include=*.pyx
```

修掉的真实故障：`cupy.get_array_module()` 在 Ascend 上 `AttributeError`
（缺 `_FusionVarArray`）；`_sorting/search.py` 的 `fusion._is_fusing()` 名字未绑定
（import 被注释）→ `cupy.where(cond, x, y)` `NameError`。

> human guide: 更彻底的重构是用 stub 一次性处理所有 `if fusion._is_fusing()`
> 判断，而不是逐个 patch。

### D.3 NEP 50 operator fallback ✅ DONE（`e2d9a9cce`）

`core.pyx` 中 `other` 不是 cupy.ndarray 且不该用 rop 时的 fallback
（比较 1271–1294、算术 1340–1453）**全部 `numpy.*` → `cupy.*`**。
原实现把 cupy 数组交给 numpy（经 `__array__`），会用 numpy 的提升规则并可能返回
numpy 数组；改走 cupy 后提升与返回类型都留在 cupy（NEP 50 语义）。

### D.4 其它派发层特例

- `e65195a08`：**`copyto()` 的标量必须先转 0-D 数组**（`cupy_copy → ascend_copy`
  kernel 不支持 `REVERSE_SCALAR_BINARY_OP`），否则标量分支静默不可用。

  ```python
  if src_is_scalar:
      from cupy.backends.backend.api.runtime import is_ascend
      if is_ascend():
          # ASCEND: elementwise_copy(cupy_copy -> ascend_copy kernel)
          # does not deal scalar (REVERSE_SCALAR_BINARY_OP)
          # convert to 0D array/tensor, so can use tensor copy/cast
          src = _core.array(src, dtype=dst.dtype)
      _core.elementwise_copy(src, dst)
      return
  ```

---

## 代码腐化（持续记录）

- **头痛医头**：上面 A.2 的 dtype 回退是两个模式（派发层全局提升 vs 逐算子
  Python 回退）。后者散落在 `ranges.py` / `_routines_linalg.pyx` 等处，宜统一成
  声明表（"算子 → 计算 dtype → 写回 dtype"）在派发层一处处理。
- **where can I get dtype support for each aclnn kernel?** CANN 头文件多数没有
  dtype 说明，只能靠真机试错 —— 需要一个 per-op dtype 支持表（可由真机探针生成）。
- `_routines_statistics.pyx` 里 `_exists_nan` 的替代分支曾被写成
  `isnam`/`keepdisms` 的拼写错误（已修），说明这类「IF 分支内代码长期不编译」
  是真实风险 —— 建议把 Ascend 分支纳入常规构建/CI。
- `runtime.pyx` 里 `initialize_backend(0)` 常被本地注释掉（无 NPU 调试），
  **不要把这个改动提交**。

## 待办

- [ ] `power()` 的 bool → int（上游 kernel 语义，未适配）
- [ ] scatter 的 1D 分支完全修复（`368cfcde5` 只部分修）
- [ ] A.2 的 dtype 回退统一为声明表
- [ ] CUDA 分支镜像缺口：`_core/__init__.py` 的 CUDA 分支未绑 `new_fusion` /
      `_fusion_thread_local`，而 `_gpu/fusion.pyx:13` 自己就依赖后者 —— 本机无
      CUDA 构建，未验证
- [ ] `_routines_math` / `_routines_logic` / `_routines_statistics` 等共享文件
      按 D.1 的模式迁入 `_ascend/`
