# ScipyCompat — scipy array-API 兼容：numpy-ascend 的主要 scipy 兼容技术路径

## 结论

numpy-ascend 对 scipy 的兼容走 **scipy array-API 模式 + cupyx 委托** 这一条路：
scipy（>= 1.15，本仓库验证于 1.17.1）在 array-API 模式下检测到 cupy 数组时，
会把调用**原样委托给 `cupyx.scipy.<module>` 下的同名函数**。因此本 fork 不需要
在 scipy 侧做任何 patch，只需要保证：

1. `import cupy` 之后 `cupyx.scipy.*` 各模块**可以被导入**（即使部分功能不可用）；
2. `cupyx.scipy.*` 的函数实现本身在 Ascend 后端可用（绝大部分与 CUDA 共享同一份代码）。

## 委托机制（scipy 侧）

入口由环境变量控制：

```
SCIPY_ARRAY_API=1
```

scipy 内部链路（以 `scipy.ndimage` 为例，见
`scipy/ndimage/_support_alternative_backends.py`）：

1. `scipy._lib._array_api` 引入 `is_cupy`（来自 array-api-compat 的
   `is_cupy_namespace`），对 `cupy` 命名空间返回 `True`。
2. 每个公开函数都被 `delegate_xp(delegator, module_name)` 装饰：
   - `is_cupy(xp)` 且函数名不在 `CUPY_BLOCKLIST` 中 →
     `importlib.import_module(f"cupyx.scipy.{module_name}")`，取同名函数直接调用；
   - 否则走 numpy 原实现。
3. `scipy.signal` 有同样的委托层（`scipy/signal/_support_alternative_backends.py`）。

scipy 1.17.1 中声明 cupy 委托的模块只有两个：

| scipy 模块 | 委托目标 | BLOCKLIST（不委托，回退 numpy 原实现） |
|---|---|---|
| `scipy.ndimage` | `cupyx.scipy.ndimage` | `distance_transform_bf`, `distance_transform_cdt`, `find_objects`, `geometric_transform`, `vectorized_filter` |
| `scipy.signal` | `cupyx.scipy.signal` | （见其 `_support_alternative_backends.py`） |

其余子模块（linalg、sparse、fft、special、interpolate…）在 array-API 模式下
只要求输入数组遵循 Array API standard（dtype/device 一致性），不涉及 cupyx 委托。

## 为什么 texture 会阻断整条链路

委托是 **`importlib.import_module("cupyx.scipy.ndimage")` 的模块级导入**——
`cupyx/scipy/ndimage/_interpolation.py` 顶部 `from cupyx import _texture`，
而 `cupyx/_texture.py` 顶部 `from cupy.cuda import texture`。

- CUDA 构建中 `cupy/cuda/texture.pyx` 编译为扩展模块；
- **Ascend 构建** 中该 pyx 不参与编译，且 `cupy/cuda/__init__.py` 的
  `if not is_ascend:` 分支跳过了 `from cupy.cuda import texture`，
  `cupy/cuda/` 目录下也没有 texture 模块文件 → `ImportError`。

也就是说：即便用户从不用 texture memory（`affine_transform(texture_memory=True)`
是可选路径），**一次 import 失败就足以让整个 scipy.ndimage 在 array-API 模式下不可用**。
这是 import-time 依赖 vs runtime 依赖的典型问题。

## 修复：texture stub（2026-10-08）

### 1. `cupy/xpu/texture_stub.py`（新增）

纯 Python stub，镜像 `cupy/cuda/texture.pxd` 中的全部 6 个类：

```
ChannelFormatDescriptor, ResourceDescriptor, TextureDescriptor,
CUDAarray, TextureObject, SurfaceObject
```

- 构造函数签名与真实 `texture.pyx` 一致；
- 所有构造函数及 `CUDAarray.copy_from/copy_to` 抛
  `NotImplementedError('texture memory is not supported on this backend ...')`；
- 导入本身完全安全（不触碰任何 CUDA/CANN 符号）。

### 2. `cupy/cuda/__init__.py`（else 分支）

```python
if not is_ascend:
    from cupy.cuda import compiler  # NOQA
    from cupy.cuda import texture   # NOQA
else:
    import cupy.xpu.texture_stub as _texture_stub
    sys.modules.setdefault('cupy.cuda.texture', _texture_stub)
    texture = _texture_stub  # NOQA
```

与文件中已有的 `device/function/memory/...` 的 `sys.modules` 别名模式一致。

### 语义

- `import` / 属性访问：可用 ✓
- 实际构造 texture 对象（如 `cupyx._texture.affine_transformation` 的
  `texture_memory=True` 路径）：抛 `NotImplementedError`（Ascend 无 CUDA
  texture API，无对应 aclnn 算子）

## 验证

```bash
python -c "import cupy.cuda.texture as t; print(t.__file__)"
# -> .../cupy/xpu/texture_stub.py

python -c "import cupyx.scipy.ndimage"          # 委托目标模块导入
SCIPY_ARRAY_API=1 python -c "
import scipy.ndimage, cupy
x = cupy.random.rand(64, 64, dtype=cupy.float32)
print(scipy.ndimage.gaussian_filter(x, sigma=3).shape)
"
```

## 定位与后续

- **这条委托路径是 numpy-ascend 兼容 scipy 的主通道**：用户代码写
  `import scipy.ndimage`、传 cupy 数组即可，无需改为 `cupyx.scipy.ndimage`。
- 后续兼容工作应集中在**保证 `cupyx.scipy.*` 各子模块 import 干净 + 函数在
  Ascend 后端可运行**；凡属 import-time 即触发的 CUDA-only 依赖（texture 是
  第一例），一律按 stub / 延迟导入处理。
- 排查清单：对 scipy 声明委托的每个 `cupyx.scipy.<module>`，在 Ascend 环境
  `python -c "import cupyx.scipy.<module>"` 验证导入通过。
