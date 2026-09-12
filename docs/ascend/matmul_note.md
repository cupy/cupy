# Ascend 后端 `matmul` 转置问题分析笔记

> 相关文件: `cupy/_ascend/_core/_routines_linalg.pyx`, `cupy/backends/ascend/acl_math_ops.h`
> 相关算子: `aclnnMatmul`, `aclnnDot`
> 状态: **已修复**

---

## 1. 现象

```
cupy.matmul(a, b) 的结果 == numpy.matmul(b, a)
```

即算出来的结果是"反的"（操作数顺序颠倒）。而 `cupy.dot(a, b)` 结果是正确的 —— 这一点是定位问题的关键线索。

---

## 2. 调用链

```
cupy.matmul(a, b)
  └─ cupy._core._routines_linalg.matmul()      # ascend 构建时由 _features.py 替换为
                                               # cupy/_ascend/_core/_routines_linalg.pyx
       └─ _ascend_matmul(a, b, out)
            └─ launch_general_func("ascend_matmul", [a, b], [out], ...)
                 └─ aclop_Matmul(self=a, other=b, out, stream)
                      └─ aclnnMatmul(a, b, out, math_type)   # 行主序 A @ B
```

对比 `dot`：

```
cupy.dot(a, b)
  └─ _routines_linalg.dot()
       └─ tensordot_core() → _ascend_dot() → launch_general_func("ascend_dot")
            └─ aclop_Dot → aclnnDot
```

`dot` 路径**不经过**下方的转置代码块，所以正确；`matmul` 路径会经过 —— 这是问题根源的切入点。

---

## 3. 根因

### 3.1 缺少 `return`，导致 fall-through 到 cuBLAS 转置代码

修复前的代码（简化）：

```python
IF CUPY_CANN_VERSION <= 0:
    # ---- CUDA/cuBLAS 路径 ----
    if ndim <= 2:
        if out is None:
            return dot(a, b, out)          # ← 有 return
        ...
ELSE:
    # ---- ASCEND 路径 ----
    if ndim == 2:
        _ascend_matmul(a, b, out)          # ← 问题：没有 return !!
    else:
        raise NotImplementedError(...)

# ↓↓↓ 下面这段本应只属于 CUDA 路径 ↓↓↓
orig_a = a
orig_b = b
...
# (A B)^T = B^T A^T
a, b = b, a                                # ← 交换了操作数
...
```

`matmul()` 的 ascend 分支调用完 `_ascend_matmul(a, b, out)` 后**没有 `return`**，
控制流直接落到下方那段代码，而那段代码是 CuPy 原版为 **cuBLAS 列主序（column-major）约定**写的：

```python
# (A B)^T = B^T A^T
a, b = b, a
```

CuPy 原版调用 `cublasGemm` 时，为了适配 BLAS 的列主序，会使用 `C = B^T A^T` 的技巧并反转 `transa/transb`，
所以先交换 `a`、`b`。但 ascend 的 `aclnnMatmul` 是**直接的行主序 A @ B**，不需要这个技巧。

结果就是：

1. 第一次 `_ascend_matmul(a, b, out)` 已经正确算完并写入了 `out`；
2. fall-through 后代码又交换了操作数，新建了 `c`/`out`，**重新调用 `_ascend_matmul(b, a, ...)`**；
3. 最终返回的是 `B @ A` —— 看起来就像"结果转置了"。

因为 `(AB)^T = B^T A^T`，对接近方阵的输入，`B @ A` 与 `np.matmul(b, a)` 完全一致，
这就是"结果和 `np.matmul(b, a)` 相同"现象的来源。

### 3.2 附证：代码里本就有 TODO 承认这一点

```python
    # TODO: code below lead to _ascend_matmul not working correctly
```

这行注释说明作者当时已经意识到 `_ascend_matmul` 与下方代码冲突，只是没定位到具体机制。

### 3.3 次要问题：`out is None` 时新建的数组被丢弃

```python
cdef _ndarray_base _ascend_matmul(a, b, out):
    if out is None:
        out = _ndarray_init(...)     # 局部变量
    launch_general_func(...)
    return out                       # 返回值被调用方丢弃
```

调用点 `_ascend_matmul(a, b, out)` 既没接收返回值、也没 `return`，
所以第一次（正确）的结果数组被丢弃，用户最终拿到的是转置那次的结果。

---

## 4. 修复

### 4.1 `matmul()` —— 提前 return，并给 CUDA 代码块加守卫

```python
ELSE:
    # ASCEND: aclnnMatmul already computes A @ B directly, no cuBLAS
    # "transpose trick" is needed. Return immediately, otherwise the code
    # below (which swaps a/b for the cuBLAS column-major convention) would
    # recompute B @ A and silently produce a transposed result.
    if ndim == 2:
        return _ascend_matmul(a, b, out)
    else:
        raise NotImplementedError("ASCEND: matmul only support dim=2 matrix mul")

# ===================================================================
# The block below is the CUDA/cuBLAS path. ASCEND never reaches here
# because the `ELSE` branch above always returns (ndim == 2) or raises
# (ndim != 2). It is kept (guarded) for the CUDA backend only.
# ===================================================================
IF CUPY_CANN_VERSION <= 0:
    ...
    # (A B)^T = B^T A^T
    a, b = b, a
    ...
```

要点：

- `return` 阻断 fall-through（核心修复）；
- 整段 cuBLAS 转置代码用 `IF CUPY_CANN_VERSION <= 0:` 包起来，对 ascend 变成编译期死代码，
  防止将来再次误触发；
- 从 CUDA 块里移除了一处对 `_ascend_matmul` 的调用（那本就是 ascend 专属函数，放 CUDA 路径是错的）。

### 4.2 `_ascend_matmul` —— 明确语义、清理调用

```python
cdef _ndarray_base _ascend_matmul(_ndarray_base a, _ndarray_base b, _ndarray_base out):
    """2-D matrix multiply via aclnnMatmul: out = a @ b.

    aclnnMatmul is a plain row-major A @ B (no cuBLAS column-major
    transpose trick), so operands must NOT be swapped here.
    """
    if out is None:
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                'shapes ({}) and ({}) not aligned'.format(a.shape, b.shape))
        ret_shape = [a.shape[0], b.shape[1]]
        ret_dtype = numpy.promote_types(a.dtype, b.dtype)
        out = _ndarray_init(cupy.ndarray, ret_shape, ret_dtype, None)
    launch_general_func("ascend_matmul", [a, b], [out], [], {}, 0)
    return out
```

### 4.3 `aclop_Matmul` —— 修正 `math_type` 类型

`aclnn_matmul.h` 中签名是：

```c
aclnnStatus aclnnMatmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);
```

注意是 **`int8_t`**，而原代码用的是 `uint8_t`，已修正：

```cpp
aclError aclop_Matmul(const aclTensor* self, const aclTensor* other, aclTensor* out, aclrtStream stream) {
    // aclnnMatmulGetWorkspaceSize(self, mat2, out, int8_t cubeMathType, ...)
    int8_t math_type = 0; // 0 == KEEP_DTYPE, keep input precision
    // row-major A @ B, no transpose trick (unlike cuBLAS column-major)
    return aclBinaryOpRun(self, other, out,
        aclnnMatmulGetWorkspaceSize, aclnnMatmul, stream, false, math_type);
}
```

---

## 5. 附带的 `dot` 实现

`aclnnDot` 的契约（见 `aclnnop/aclnn_dot.h`）：

- `self`、`tensor` 必须是 **1-D**，shape 相同，dtype 支持 FLOAT / BFLOAT16 / FLOAT16；
- `out` 必须是 **0-D**（scalar）。

而 numpy `dot` 的语义远不止 1-D·1-D。因此在 `dot()` 里显式分派：

| 输入 | 实现路径 |
|------|----------|
| scalar 参与 | `_math._multiply`（原有逻辑） |
| 1-D · 1-D | `tensordot_core` → 展开为 1-D，`aclnnDot` 写到 0-D 临时数组，再 `elementwise_copy` 回 `out` |
| 2-D @ 2-D | 复用 `_ascend_matmul`（`aclnnMatmul`） |
| 其他维度 | 显式 `NotImplementedError`（不再静默出错） |

`tensordot_core` 的 ascend 分支：

```python
IF CUPY_CANN_VERSION > 0:
    # ASCEND: aclnnDot only accepts two 1-D tensors and writes a 0-D output.
    if (a._shape.size() == 1 and b._shape.size() == 1
            and a.size == b.size):
        scalar_out = _ndarray_init(cupy.ndarray, shape, dtype, None)  # shape == () -> 0-D
        launch_general_func("ascend_dot", [a, b], [scalar_out], [], {}, 0)
        elementwise_copy(scalar_out, out)
        return out
    raise NotImplementedError(...)
```

---

## 6. 编译验证

```sh
export CUPY_INSTALL_USE_ASCEND=1
python setup.py build_ext --inplace
```

```
INFO:root:building 'cupy.backends.ascend.api.acl_utils' extension       ✅
INFO:root:building 'cupy._core._routines_linalg' extension              ✅
```

两个模块均成功编译链接。

> **注意**：无 NPU 环境下 `import cupy` 仍会失败，原因是 CANN 8.5 库自身的符号不一致：
> ```
> libop_common.so: undefined symbol: _ZN2ge19GetViewErrorCodeStrENS_13ViewErrorCodeE
> ```
> 实测该符号在库中定义于 `opcommon` 命名空间（`_ZN8opcommon19GetViewErrorCodeStr...`），
> 而调用方期望 `ge` 命名空间。这是环境/版本问题（详见 `README.md` §2.4），与本次代码修改无关。
> 数值正确性需在带 NPU 的机器（910B）上跑 `pytest` 验证。

---

## 7. 经验教训

1. **Cython 的 `IF` 是编译期分支**，两个分支都写在同一个函数里；一侧 `return`、另一侧忘记 `return`
   时，编译能通过、但另一侧会 fall-through 到不属于它的代码 —— 这类 bug 很隐蔽。
   建议：每个 `IF/ELSE` 分支末尾都显式 `return` 或 `raise`。
2. **后端语义差异要在 helper 边界显式注释**。cuBLAS 的"转置技巧"是后端特例，
   `aclnnMatmul` 不需要；把差异写在函数 docstring 里可避免后续维护者踩坑。
3. **迁移时删除/隔离原后端专有代码**。这段 cuBLAS 转置逻辑对 ascend 毫无意义，
   用 `IF` 守卫隔离比"让它自然 fall-through"安全得多。
