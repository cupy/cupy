# float64 处理与段错误：代码检视与修复记录

> 检视对象：float64 → float32 降档机制（float32 / cpu 两模式）在
> `_kernel.pyx`、`_reduction.pyx`、`acl_utils.pyx` promote 层的实现；
> arange / creation / indexing / manipulation 上的段错误。
> 环境约束：910B 无 float64 硬件吞吐，部分 aclnn op 不收 DOUBLE/COMPLEX128。

## 一、现状链路

float64 处理目前只存在于 **3 个拦截口**，覆盖不完整：

| 拦截口 | 生效模式 | 位置 |
|---|---|---|
| ufunc `__call__`（含 `_arange_ufunc` 等 create_ufunc 产物） | 仅 cpu 模式 | `cupy/_core/_ascend/_kernel.pyx:981` |
| reduction `_call`（cpu 拦截 + `_FLOAT64_DEMOTE` 降档表） | cpu + float32 | `cupy/_core/_ascend/_reduction.pyx:338-383` |
| `launch_general_func_raw` / `launch_elementwise_func_raw` promote 层 | 仅 float32 | `cupy/backends/ascend/api/acl_utils.pyx:1236, 1551` |

## 二、段错误根因（按确定性排序）

### P1（确定）：promote 层只降档 ins/outs 的 ndarray，不降档标量 args

`_promote_io_dtype`（`acl_utils.pyx`）只遍历 `p_ins`/`p_outs` 里的
`_ndarray_base`。float32 模式下 arange f64 实际到达 aclnn 的参数：

- `out` → 提升为 **ACL_FLOAT** 临时数组；
- `start`/`step` → `_arange_ufunc` 按 `'dd->d'` loop 物化的 **ACL_DOUBLE**
  CScalar（`_kernel.pyx:953`，M-D1 标量按 loop dtype 物化）。

`aclop_Arange`（`acl_general_ops.h:324`）里 `stop` 按 `outs[0]` 的 dtype
重建了，但 `start = args[0].scalar`、`step = args[1].scalar` 保持 DOUBLE →
`aclnnArange(f64 start, f32 end, f64 step, f32 out)`。aclnnArange 要求四者
同 dtype，CANN 对这种不一致经常不是返回错误码，而是在 infer-shape/tiling
阶段**直接段错误**。

同样命中 **creation**：`cupy.zeros/ones/full` → `ascend_fill` →
`aclop_Fill`（`acl_general_ops.h:291`）把 `args[0].scalar`（f64）灌给
`aclnnInplaceFillScalar`，而 `outs[0]` 已被 promote 成 f32。

注意：arange/fill 的标量都在 `ins`（`launch_general_func(self.name,
list(inout_args), ...)` 的第一个参数）里，不在 `args` 通道。

### P2（确定）：部分 aclnn op 本身不收 float64

CANN 8.5 头文件确认：

| op | dtype 支持 | 影响 |
|---|---|---|
| `aclnnScatterAdd` | FLOAT16, FLOAT32, INT32, INT8, UINT8（**无 DOUBLE**） | indexing：`a[i]+=v` / `cupy.add.at`（`ascend_scatter_add`） |
| `aclnnArange` | 头文件无 dtype 文档；内核只可靠支持 FLOAT/INT32/INT64 一族 | arange/linspace |
| `aclnnTake` | 文档声明支持 DOUBLE | take 本体降档后可用 |
| `aclnnIndexSelect` | 文档声明支持 DOUBLE | take(axis) 降档后可用 |
| `aclnnRepeat` / `aclnnCat` | 头文件无 dtype 声明 | tile / concatenate，需逐一验证 |

即 indexing 的崩溃主要来自 scatter 系（scatter_add）与
`index_put_impl`，而非 take/index_select。

### P3（结构性）：cpu 模式没有覆盖 general 直发路径

以下调用点**绕过 `_kernel.pyx`/`_reduction.pyx` 直接 `launch_general_func`**：

- `_routines_indexing.pyx`：`ascend_nonzero` / `ascend_index_put_impl` /
  `ascend_take` / `ascend_index_select` / `ascend_scatter_update` /
  `ascend_scatter_add`
- `cupy/_manipulation/tiling.py:46`：`ascend_repeat`
- `cupy/_core/_ascend/_routines_linalg.pyx`：trace/tril/triu/qr/svd/inverse/matmul/dot
- `cupy/random/_generator.py`：random 系列
- `cupy/_creation/ranges.py:88`：`ascend_complex`

cpu 模式（`CUPY_ASCEND_FLOAT64_MODE=cpu`）下这两个口只拦 ufunc 与归约，
f64 数组原样喂给 aclnn —— 对不收 DOUBLE 的 op 就是段错误而不是报错。
float32 降档只在 `ascend_float64_promote_enabled()` 为真时挂进 promote 表，
与 cpu 模式互斥，故 cpu 模式下该路径**零保护**。

### P4（确定）：cpu fallback 自身的三个 bug

1. **`_to_host` 非连续分支是堆越界读 = 段错误直接来源**
   （`cpu_fallback.py:315-321`）：
   ```python
   buf = numpy.empty(value.nbytes, dtype=numpy.uint8)
   value.data.copy_to_host(..., value.nbytes)
   view = as_strided(buf.view(...), shape=value.shape, strides=value.strides)
   ```
   strided 视图的 `nbytes = size * itemsize` **小于**数据实际跨度
   `sum((shape[i]-1) * strides[i]) + itemsize`。`copy_to_host` 只搬
   nbytes 字节（数据本身已错），随后 `as_strided` 按 byte strides 在
   nbytes 大小的缓冲上寻址 —— 典型 heap-buffer-overflow。indexing /
   manipulation 产生的 out / 中间数组恰恰经常是非连续视图。
2. **`run_elementwise_host` 的 `getattr(numpy, fname)` 名字映射对签名
   不同的内部 ufunc 语义错误**（`cpu_fallback.py:346-351`）：
   `cupy_arange(start, step)->out` 会被算成 `np.arange(start, step)`
   （step 被当成 stop，shape 必然不对）；`cupy_linspace` 更糟 ——
   `np.linspace(start, step)` 默认 `num=50`，返回 50 个元素写进更小的 out。
3. **`dev_out.set(np_out)` 前无 shape/dtype 校验**（`cpu_fallback.py:372`）
   —— 配合上一条，大结果写小缓冲就是段错误。

## 三、修复记录

### 修复 1：promote 层补标量降档（`acl_utils.pyx`）

新增 `_demote_scalar_operands`：promote 命中后，把

- `ins` 里的 CScalar（arange 的 start/step、fill 的填充值），
- `args` / `kwargs` 里的 CScalar（clip/random 等走 args 通道的标量），

按降档表重造（`d`→`f`、`G`→`F`），使「标量 dtype == tensor dtype」的
aclnn 合同（arange/fill 等）在降档后仍然成立。elementwise 与 general
两个通道都接线。

### 修复 2：arange/linspace float64 work-dtype 降档（`cupy/_creation/ranges.py`）

仿照已有的 `_ASCEND_ARANGE_INT32_FALLBACK` 模式：Ascend 上 float64 的
arange 先用 **float32** 生成（`work_dtype`），再 `astype` 回 float64 ——
完全绕开 aclnnArange 的 dtype 限制，也不再依赖 promote 层的标量降档。
linspace 标量路径同理（`_linspace_ufunc` 补 `ff->f` / `fff->f` loop，
f64 时用 f32 work dtype 计算，末尾已有 `astype(dtype)` 兜底）。

精度语义与 float32 降档一致：用户可见 dtype 仍为 float64，精度降为单精度。

### 修复 3：`_to_host` 非连续物化（`cupy/_core/_ascend/cpu_fallback.py`，方式 1）

非连续数组先在设备侧物化成 C-连续副本再 D2H：

```python
contig = cupy.empty(value.shape, dtype=value.dtype)
py_launch_general('ascend_copy', (value,), (contig,), (), {})
return cupy.asnumpy(contig)
```

- `ascend_copy` 在 `_UINT_PROMOTE_EXEMPT_OPS` 豁免表内（实现即
  aclnnCast，原生支持 DOUBLE），不会递归；
- 不经过 ufunc 派发层，不会撞上 cpu 模式下 ElementwiseKernel 的
  f64 响亮报错；
- 一次性连续 D2H，取代「裸 memcpy + as_strided」的越界读实现。

## 四、遗留事项（未在本次修复范围）

1. **P3 的完整解法**：在 `launch_general_func`（checked 入口）加
   `has_f64_io` 探测，cpu 模式下对 general 直发路径响亮
   `NotImplementedError`（或查 host 注册表），把所有「op 收不收
   DOUBLE」从段错误变成显式失败。
2. **aclnn dtype 能力表**：`register_acl_ufunc` 增加每 op 支持的
   aclDataType 集合（CANN 头文件整理 + 首次调用 probe 缓存），
   promote 层查表逐 op 降档，替代全局一刀切；顺带解决
   `_UINT_PROMOTE_EXEMPT_OPS` 里 take/manipulation 豁免的 TODO。
3. **`run_elementwise_host` 白名单化 + set 前校验**：仿照
   `_REDUCTION_HOST_MAP`，只放行签名/语义与 numpy 一致的 ufunc；
   `cupy_arange`/`cupy_linspace`/`cupy_bincount_kernel` 等内部 ufunc
   响亮报错；写回前校验 shape/dtype。
4. `aclop_Arange`/`aclop_Linspace` 每次调用 `PrintArgs` 到 stdout，
   去掉或改宏开关。
5. indexing 的 scatter_add 在 float32 降档后仍不收 int64/其它 dtype，
   需按能力表回退 `_scatter_op_host_fallback`。

## 五、验证

```bash
export CUPY_INSTALL_USE_ASCEND=1 CUPY_ASCEND_FLOAT64_MODE=float32
python setup.py build_ext --inplace 2>&1 | grep -iE "error:|undefined reference"
# 无 NPU：L1/L3（import + registry）；数值正确性待 910B
python -c "import cupy; cupy.arange(0, 1, 0.1, dtype=cupy.float64)"
```

分类：arange/creation（P1+修复 1/2）、indexing scatter（P2，能力表）、
cpu fallback 段错误（P4-1+修复 3，纯 host bug，无 NPU 可复现）。

## 六、general ops 的两类划分与 float64 策略（架构结论）

### 6.1 分类是真实的

已注册的 67 个 GENERAL_OP 按语义分两类：

**A 类：数值计算（elementwise 带参数）** —— ins/outs/args 全部 dtype
耦合，promote 降档 + 标量降档（修复 1）+ cast-back 通用正确：

- 多元 elementwise 带参：clip / is_close / round / divmod / nan_to_num(_) /
  heaviside / where / complex / masked_fill_scalar / masked_fill_tensor /
  minn / maxn / max_v2 / var_core
- scan：cumsum / cumprod
- 统计/归约形态：aminmax / histc / bincount_kernel(+with_weight) /
  searchsorted_kernel
- 生成：arange / linspace / random_uniform / random_normal / random_int /
  multinomial

**B 类：manipulation / index / creation（结构操作）** —— 存在不可降档的
操作数（int64 index/shape 张量）、别名写入（scatter 族 self==out）、
数据依赖形状（nonzero）、view 语义：

- 拷贝/填充（aclnnCast 载体）：cast / copy / positive / fill / fill_diagonal
- 索引取值：take / take_scalar / gather / gather_nd / index_select /
  index_copy / put_raise
- 散写：scatter_update / scatter_add / scatter_max / scatter_min /
  scatter_update_mask / scatter_add_mask / getitem_mask / index_put_impl
- 形状/装配：concatenate / stack / repeat / permute / flip / roll
- 排序/去重：sort / argsort / unique2 / unique_consecutive
- 矩阵结构/稠密线代：trace / tril / triu / qr / svd / inverse / slogdet /
  einsum
- 探针：dump_args

边界裁决：arange/linspace 归 A（修复 2 已在 Python 层绕开）；sort/unique
归 B（dtype 限制多、host 排序可靠）；qr/svd/inverse/einsum 数值上像 A，
但 host fallback 已是常态（cpu_fallback.FALLBACKS），策略上归 B。

### 6.2 是否新增 OpType？—— 不建议，用注册元数据标签

**结论：按类分策略是必要的，但不要新增 OpType 枚举值作为注册表键；
在 `register_acl_ufunc` 加 `op_class='numeric'|'structural'` 元数据标签。**

理由：

1. **派发形态相同**。两类的调用签名完全一致——都是
   `FuncPtrUnion.general_op(intensors, outtensors, acl_args, acl_kwargs,
   stream)`，都只在 `launch_general_func_raw` 一个入口派发。差异在策略
   （float64 处理、参数通道、错误处理），不在派发。类型系统表达派发
   形态，策略单独表达，两者解耦。
2. **查找键风险**。`_builtin_operators` 哈希键是 `(op_name, op_type)`；
   若 B 类注册成 `STRUCTURAL_OP=10`，所有 `find()` 点都要双查或归一化，
   漏一处 → 注册"成功"但永远派发不到 → fall-through 到 elementwise
   通道报误导性错误。`register_acl_ufunc` 静默覆盖 + 名字静默失配的坑
   历史上已踩过，不宜引入同类风险。
3. **`get_op_type` 不受影响**。新枚举值只有手写注册会产生，elementwise
   通道的自动推导完全无关——这恰恰说明它不是"派发类型"而是"政策标签"。

### 6.3 落地三步（风险递减）

1. `register_acl_ufunc(..., op_class='numeric'|'structural')`（默认
   numeric，向后兼容）：存入 `OpInfo` 或旁表 `cdef set _STRUCTURAL_OPS`；
   OpType 枚举与查找键不动。67 个注册点按 §6.1 分批标注（先只标 B 类）。
   `py_list_acl_ufuncs` 带出 class，coverage 工具按类出报告。
2. promote 层按类分流：A 类 → 现行 `_promote_io_dtype` +
   `_demote_scalar_operands`；B 类 → 跳过自动 promote，改查 per-op dtype
   能力表，不支持的 dtype 挂 `cpu_fallback._GENERAL_HOST_FALLBACKS`
   （cpu 模式即 `maybe_general_host` gate）。现 `_UINT_PROMOTE_EXEMPT_OPS`
   就是该策略表的前身：A 类豁免 = 原生支持 DOUBLE；B 类缺省豁免自动
   promote。
3. B 类的 index/shape 张量永远不进 promote 表；别名写入 op（scatter 族
   self 与 out 同指针）禁止"数据张量 promote + 临时 out cast-back"——
   语义虽机械成立，性能与审查成本不划算。
