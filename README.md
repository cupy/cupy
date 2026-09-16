# numpy for Ascend NPU: forked from Cupy

By Qingfeng Xia

## 1. Status of numpy-ascend Array API suport

see  [Progress.md](./Progress.md)  91.5 % Array API（118/129），except for eigen；注册算子 174，
细节见自动生成的 [tools/cst_db.md](./tools/cst_db.md)

### completed

1. customed kernel (ascend c, triton-ascend python)
2. all cupy major features, except for random (can be done)

### limitation
1. float32 only for all array API, similarly, default dtype float32, instead of float64 on CPU
2. float64/int64 support add/substract/mul/div ops
3. sparse array/matrix not supported

#### pytest 上的 dtype 过滤 (减少假失败)

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

### torch_npu used as numpy array API
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

## 3. 安装指南


### 3.2 基本稳定阶段

```sh
# 第二阶段： 如果测试比较稳定， 可以直接git拉去代码， 编译二进制wheel
pip install  git+https://github.com/qingfengxia/cupy-ascend.git

# 第三阶段： 如果大规模测试通过， 已经有pip二进制包
pip install numpy-ascend
# pip install cupy-cuda12x
```

可以打包wheel简化安装, 不用用户编译

### 3.3 benchmark.py 代码

2025年开发了100小时, 达成MVP (最小功能单元), 测试了matmul, cos, add, 在910B实现了非常客观的加速, 几十到一百的加速. 
看benchmark.py 

但是还是有大量工作, 预计为1人年, 欢迎加入测试和开发. 

