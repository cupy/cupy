
## Done

### 1.1 Progress
- Oct 12: MVP for add, cos, matmul, benchmark 10-100X acceleration

- Oct 23: benchmark.py 经过xpu重构后 (重构代码在 xpu branch), NPU测试可以运行
  backends -> cupy.backends,  git mv + _features.py
  disable _preflight, so ignore Cutensor submodule

- Nov 08: reduction op such as `sum()` is working, 
    90% math ops ACLOP supported has been added into numpy-ascend
    UnitTest: `pytest tests/cupy_tests/logic_tests/test_truth.py `

- Nov 15: concatenate(), clip(), copy(), non-math/irregular ops initially supported
    + but `array()` seems not working properly 
    > reason is async_copy, two arrays created without print the first will have the same value as the second array
    + scalar 转化 not working, 可能是exp scalar op 没有注册   DONE

- Dec 06: 
    + creation apis:  `arrange()` added, but test failed
    + `concatenate` test passed
- Dec 13:  sorting API
    + `sort()/argosrt()` added, but no `partition()` alcop, 
    + sort also depending on `rollaxis()`
    + fill_kernel() -> aclnnop_FillScalar  GeneralOp 类型

- BitwiseAddScalar op register, _kernel.pyx need update (done)
- reduction kernel, replaced by aclnnop

pytest can run on ubuntu with NPU installed


## 1. Short-term TODO

1. creation/manipulation/indexing/linalg ops

2. statistics ops: passing string arg, it has issue
    it may need CANN 8.5 to construct aclScalar of string type

3. ~~matmul~~ **FIXED**: 
   - 根因: `matmul()` 的 ascend 分支调用 `_ascend_matmul(a, b, out)` 后**缺少 `return`**,
     fall-through 到 cuBLAS 专用代码块的 `a, b = b, a` 转置技巧, 又重新算了 `B @ A`,
     导致 `matmul(a,b)` 返回 `np.matmul(b,a)`。
   - 修复: ascend 分支提前 `return _ascend_matmul(a, b, out)`;
     并把下方 CUDA/cuBLAS 转置代码块包进 `IF CUPY_CANN_VERSION <= 0:` 守卫, 使其对 ascend 成为死代码。
   - 同时实现 `dot()`: 1-D·1-D -> `aclnnDot`(0-D 输出), 2-D@2-D -> `aclnnMatmul`;
     其余维度显式 `NotImplementedError`。
   - `_ascend_dot`/`_ascend_matmul` 统一用 `launch_general_func(..., [], {}, 0)`; 清理了 debug print。
   - `aclop_Matmul` 的 `math_type` 由 `uint8_t` 修正为 aclnn 签名要求的 `int8_t`。

4. concat/pad/reshape op
   numpy_to_acl_dtype ->  numpy_dtype_to_acl_dtype

6. triton-fusion (add data adaptor API)
 or once CANN 8.5 stable released, and pyPTO will be used to write customised kernel

7. aclBlas integration

8. test build on diff OS, currently focus on Ubuntu

10. inplace op :  _kernel.pyx need update


## 2. 核心op支持情况 ( see also Array API standard)
https://data-apis.org/array-api/latest/API_specification/index.html

### 2.2 math ops: 
+ 未注册  einsum, cbrt(cube root, not std api), fix (Trunc), rint (Round), round/around, convolve (?),
+ 自己实现: radians (deg2rad), degrees (rad2deg), deg2rad, rad2deg. lcm, divmod 
+ missing 数值计算: gradient, interp, trapezoid, diff
+ missing: frexp, ldexp ()
+ complex numpy ops: angle, conj,  缺少几个ops但是自己实现很简单,  real, complex
+ scan (numpy has no such op), true_divide
+ cupy.math_op(scalar, tensor), can aclop kernel broadcast deal with this?

### 2.3 indexing ops
- slicing ? working, but it does not use `Slice` aclop
- `math.scan()` is a dummy/empty func, no such aclop
- aclop has `take, put(InplacePut), slice`, but no `choose`

### 2.4 manipulation ops
可能有大量不兼容, 测试工作量不小
+ CUPY `reshape, split` does not need kernel, it is done in cython code on host (Reshape api)
+ ACLOP having: `roll, permute, flip, repeat` , while repeat/rollaxis() is written in cython, no kernel needed
+ cupy uses `concatenate` to impl vstack, stack, hstack without using CUDA kernel
+ `_manipulation/rearange.py`  slicing is used to flip, rotate
+ `squeeze`: Removes size-one axes from the shape of an array

### 2.5 logical/bitwise ops:  
+ ACLOP misses numpy op: `_left_shift`, `_left_right`
+ `cupy_is_close` should be used as `a.isclose(b)`
+ `is_nan()`: 

TODO   but why `aclnnEqual`has no tensor-scalar version?

### 2.6 statistics reduction ops: 
+ registered: median, var, mean, std,  bincount, histgram (histc), 主要是看nan怎么处理, 部分做了注册

+ missing: average, quantile,  percentile, vecter op实现难度应该不太大
+ ptp (Range of values (maximum - minimum) along an axis.) -> Aminmax

TODO: passing keyword args

### 2.7 set op
+ `is1d`
Array Std API support only:
+ unique_all
+ unique_counts
+ unique_inverse
+ unique_values

### 2.8 random and distribution

AsNumpy project has impl
https://gitcode.com/cann/asnumpy

### similar ops 需要验证numpy行为是否一致
1. fmin, nanmin, min, amin
2. remainder, fmod, modf
3. rint, round, around
4. dot, matmul, mm, gemm, inner
5. fabs(real number only), abs
