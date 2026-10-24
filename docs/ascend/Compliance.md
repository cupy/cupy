# Compliance — 发布物里的 CUDA 资产：审计、政策与处置

> **状态**：活跃（2026-10-18）。政策 + 审计快照 + 已落地改动。
> 审计对象：本仓库源码树、`dist/wheelhouse/*.whl`、`dist/*.tar.gz`。
> 复现命令见 §6；数字来自 2026-10-18 实测，改动后请重跑。

---

## 0. 一句话

**仓库里唯一不能带的是 NVIDIA 专有的 CUDA Toolkit 头文件**（`cuda_fp16.h/.hpp`）；
它们已从版本控制移除，改为「CUDA 后端在构建/运行前从本机 toolkit 复制」，
并且 `setup.py` + `MANIFEST.in` 会把它从**所有** wheel/sdist 里过滤掉。
其余 CUDA 相关文件要么是**自研 shim（只引用）**，要么是**开源许可的第三方代码**（可再分发，需 LICENSE），
Ascend 侧额外把 CCCL / jitify / `*.cuh` / PBA 内核从发布物里剔除（纯减重，不影响功能）。

---

## 1. 政策（红线）

| # | 规则 | 说明 |
|---|---|---|
P1 | 仓库（git 跟踪内容）不得包含任何第三方 CUDA **专有**源码/头文件 | 只能引用（`#include <cuda_runtime.h>` 或 submodule 指针） |
P2 | sdist / wheel **不得**包含 CUDA Toolkit 头文件 | 即使是本地临时复制出来的，也必须在打包前过滤 |
P3 | 开源许可的 CUDA 代码（CCCL/jitify/PBA/`cupy/complex`）允许分发，但必须随附 LICENSE/NOTICE | Ascend 发布物**不需要**它们，直接剔除 |
P4 | 获取第三方 CUDA 资产的唯一方式是「本机已有 → 复制」或「submodule 指针」 | 禁止把厂商文件提交入库；脚本不联网下载 |

判定依据：NVIDIA CUDA Toolkit EULA 明确禁止未经书面同意的再分发；
而 CCCL/jitify/dlpack/PBA 分别以 Apache-2.0(+LLVM exception)/BSD-3/Apache-2.0/MIT 授权。

---

## 2. 审计结果：逐项定性

### 2.1 `cupy/backends/**` —— 合规（自研 shim，只引用）

| 文件 | 性质 | 判定 |
|---|---|---|
`cupy/backends/cuda/cupy_cuda.h:4` | 6 行，仅 `#include <cuda.h>  // for CUDA_VERSION` | ✅ 引用 |
`cupy/backends/cuda/cupy_cuda_runtime.h:4-5` | `#include <cuda.h>` / `<cuda_runtime.h>` | ✅ 引用 |
`cupy/backends/cuda/cupy_{cublas,cusolver,cusparse,nccl,cutensor,cuda_profiler_api}.h` | 同名 API 的转发/补丁层 | ✅ 自研 |
`cupy/backends/cupy_{blas,sparse,rand,lapack,tx,xccl,complex}.h` | 按 `CUPY_USE_HIP` / `CUPY_USE_ASCEND` / `CUPY_NO_CUDA` 分发 | ✅ 自研 |
`cupy/backends/stub/*.h` | 4167 行手写桩（`// This file is a stub header file of cuda for Read the Docs.`） | ✅ 自研 |
`cupy/backends/ascend/*.h` | 只 `#include "acl/acl.h"` 等 CANN 头 | ✅ 无关 CUDA |

Ascend 构建链里没有任何文件包含 CUDA 头（全仓仅 `cupy/cuda/jitify.pyx:138,146` 含 `<cuda_fp16.h>`，该文件在 Ascend 上不编译）。

### 2.2 `cupy/_core/include/cupy/**` —— 混合（关键区）

| 路径 | 版权/许可 | git 跟踪方式 | 判定 |
|---|---|---|---|
`_cuda/cuda-11/cuda_fp16.{h,hpp}`、`_cuda/cuda-12/cuda_fp16.{h,hpp}` | **NVIDIA 专有**（`Copyright 1993-2021 NVIDIA`，`PROPRIETARY AND CONFIDENTIAL`，禁止再分发） | **曾是实体文件** | ❌ **硬违规 → 已删除（2026-10-18）** |
`_cccl/{cub,libcudacxx,LICENSE}` | NVIDIA CCCL，Apache-2.0 + LLVM exception | **symlink → `third_party/cccl`**（submodule，只存指针） | ✅ 仓库内合规；Ascend 发布物剔除 |
`_jitify/jitify.hpp` + `LICENSE` | jitify，BSD-3 | symlink → `third_party/jitify` | ✅ 同上 |
`_xsf` → `third_party/xsf` | scipy xsf，Boost | symlink | ✅ 非 CUDA |
`_dlpack/dlpack.h` | DLPack，Apache-2.0 | 实体副本（LICENSE 为 symlink） | ✅ 非 CUDA |
`complex/*.h{,pp}` | `Copyright 2008-2013 NVIDIA`，**Apache-2.0**（CuPy 维护的派生） | 实体文件 | ✅ 可再分发（建议补 NOTICE） |

`_cuda/README.md`（上游）自称 *"copied from CUDA Toolkit Distribution and redistributed under
the NVIDIA CUDA Toolkit License Agreement"* —— 这是上游的自我解释，**不满足 P1/P2**，已重写为
「不 vendored、用 `tools/fetch_cuda_fp16_headers.py` 本地复制」。

### 2.3 其他 CUDA 源码（项目自有，授权上无问题）

| 文件 | 许可 | 说明 |
|---|---|---|
`cupy/cuda/cupy_{thrust,cub,cufftXt}.{h,cu}`、`cupy/cuda/cupy_thrust.h` | 项目 Apache-2.0 | 自研 CUDA 包装（`thrust.h` 本就是 `.gitignore` 的生成物） |
`cupy/random/cupy_distributions.{cu,cuh}` | 项目 Apache-2.0 | 自研 device 代码 |
`cupyx/scipy/ndimage/cuda/pba_kernels_{2,3}d.h` + `LICENSE` | MIT（NUS PBA） | 已带 LICENSE |

按 P1 只针对**第三方专有**资产，这些不需要删除；若政策升级为"零 CUDA 源码"，见 §5 的可选清理。

### 2.4 发布物实测（改动前）

| 产物 | 头文件数 | 体积（未压缩） | 其中 CUDA 相关 |
|---|---|---|---|
`dist/wheelhouse/numpy_ascend_cann90-14.0.0a1-...whl`（Ascend） | 1593 | 13.85 MB（压缩后 2.79 / 7.05 MB ≈ **40 %**） | `_cccl` 1442 个 11.19 MB、`_cuda` 4 个 0.47 MB、`_jitify` 0.17 MB、`complex` 0.15 MB、PBA 0.03 MB |
`dist/cupy-14.0.0a1.tar.gz`（sdist） | 1672 | — | 含 `cuda_fp16.h` 本体（**违反 P2**） |

LICENSE 归属：`_cccl/LICENSE`、`_dlpack/LICENSE`、`_jitify/LICENSE`、`cuda/LICENSE`(PBA) 均在 wheel 内；
`xsf` 的 Boost LICENSE 缺失（非 CUDA，低优先，见 §5）。

---

## 3. 已落地改动（2026-10-18）

| # | 改动 | 文件 |
|---|---|---|
1 | **删除** 4 个 NVIDIA 专有头（`git rm`），保留并重写目录 README | `cupy/_core/include/cupy/_cuda/cuda-{11,12}/cuda_fp16.{h,hpp}` |
2 | 复制目录加入 `.gitignore` | `.gitignore` |
3 | **本地获取脚本**（只复制、不联网） | `tools/fetch_cuda_fp16_headers.py` |
4 | **打包过滤**（唯一权威实现）：`is_cuda_toolkit_header()` / `is_cuda_only_file()` / `strip_cuda_headers()`；CUDA 专有头**所有后端都剔除**，CCCL/jitify/`*.cuh`/PBA **仅 Ascend 剔除** | `install/cupy_builder/cupy_setup_build.py` |
5 | `package_data` 应用上述过滤（wheel） | `setup.py` |
6 | 新增 `sdist` cmdclass：Ascend 打 sdist 时过滤 CUDA-only 源 | `setup.py` |
7 | `prune _cuda/cuda-11|cuda-12`（sdist 兜底，跨后端无条件） | `MANIFEST.in` |
8 | NVRTC 不再假定 bundled 目录存在；目录缺失时回退 toolkit/wheel 头，否则给带指引的 `RuntimeError` | `cupy/_core/_gpu/_compile_with_cache.pyx:194-211` |
9 | `build_py` 子类：打包后 prune `build/lib` 里历史残留的 CUDA-only 文件与 Ascend 剔除包（`build_py` 只增不删，否则过滤会被旧产物绕过） | `setup.py` |

### 3.1 CUDA 后端的获取路径（P4）

```
cupy/_core/_gpu/_compile_with_cache.pyx 查找顺序
  ① cupy/_core/include/cupy/_cuda/cuda-{11,12}/     ← 本机复制的副本（gitignored）
  ② $CUDA_PATH/include（或自动探测的 toolkit）
  ③ conda / nvidia-cuda-runtime-* wheel 提供的 include
  ④ 都没有 → RuntimeError（提示跑 tools/fetch_cuda_fp16_headers.py）
```

按 NVRTC 版本决定是否需要 bundled 副本（与上游一致）：CUDA 11.x → `cuda-11`；12.0–12.1 → `cuda-12`；
**12.2+ 不使用 bundled 副本**（`cuda_fp16.h` 依赖无法打包的 `vector_types.h`）。

脚本行为：`--list` 列候选与版本；`--include-dir` 指定；`--target` 强制目录；`--dry-run` 试跑；
`--force` 覆盖；**不发起任何网络请求**。

### 3.2 Ascend 不受影响（已核）

- `cupy/_core/_gpu/_compile_with_cache.pyx`、`cupy/cuda/jitify.pyx`、`cufft` 在
  `install/cupy_builder/features/ascend.py` 的编译清单里都不存在；
- Ascend 头文件只 include CANN（`acl/acl.h`）；
- `_CUPYX_EXCLUDED_ON_ASCEND`（Ascend wheel 剔除的 `cupyx` 子包）与本改动互不重叠。

---

## 4. 验收标准

| # | 检查 | 期望 |
|---|---|---|
A1 | `git ls-files cupy/_core/include/cupy/_cuda/` | 只剩 `README.md` |
A2 | `git ls-files \| grep -i cuda_fp16` | 空 |
A3 | `CUPY_INSTALL_USE_ASCEND=1 python setup.py sdist` 后扫 tarball | 无 `_cuda/`、`_cccl/`、`_jitify/`、`*.cuh`、`pba_kernels_*` |
A4 | 普通（CUDA）`python setup.py sdist` | 无 `_cuda/cuda-1*`；`_cccl`/`_jitify` 保留（NVRTC JIT 需要） |
A5 | Ascend wheel | 头文件体积从 13.85 MB → ~1.8 MB（仅 `xsf`） |
A6 | 本地跑过 `fetch_cuda_fp16_headers.py` 后再打包 | 依然不含 `_cuda/*`（过滤是双保险） |
A7 | `python tools/fetch_cuda_fp16_headers.py --list`（无 CUDA 机器） | 打印"未找到"+ 安装提示，exit 1，不报 traceback |

---

## 5. 后续（可选，按政策强度决定）

| 优先级 | 项 | 说明 |
|---|---|---|
P1 | ✅ `cupy/_core/include/cupy/complex/*` 补 NOTICE | 已完成（2026-10-24）：根目录 `NOTICE` 统一列出上游 MIT 部分与 Apache-2.0/BSD-3/BSL-1.0 第三方的版权行、许可证位置、修改声明与商标声明；`pyproject.toml` 的 `license-files` 与 `MANIFEST.in` 保证其进入 wheel/sdist |
P2 | `xsf` 的 Boost LICENSE 未进 wheel | 加 `_core/include/cupy/xsf/LICENSE` symlink 到 `third_party/xsf/LICENSE` |
P2 | Ascend 发布物连 `xsf` 也剔除 | 无 `cupyx.scipy.special` 的 C++ 实现需求（kernel 路径在 Ascend 不可用），可再省 1.79 MB |
P3 | 若"零 CUDA 源码"成为硬政策 | `cupy/cuda/*.cu{c,h}`、`cupy_distributions.cu{c,h}`、`pba_kernels_*.h` 也改为构建期从 cccl / 独立仓库获取 |
P3 | sdist 全面 Ascend-only 化 | 需要 CI 分两条流水线（CUDA sdist 与 Ascend sdist 各自发布） |

---

## 6. 复现命令

```sh
# 1) 仓库里还有没有厂商头
git ls-files | grep -iE 'cuda_fp16|_cuda/' 

# 2) 本地获取脚本（不联网）
python tools/fetch_cuda_fp16_headers.py --list
python tools/fetch_cuda_fp16_headers.py --dry-run

# 3) 打 Ascend sdist 并检查
CUPY_INSTALL_USE_ASCEND=1 python setup.py sdist --dist-dir /tmp/sd
tar tzf /tmp/sd/*.tar.gz | grep -E '_cuda/|_cccl/|_jitify/|\.cuh$|pba_kernels'   # 期望空

# 4) 检查已有 wheel
python - <<'EOF'
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1])
h = [n for n in z.namelist() if n.endswith(('.h', '.hpp', '.cuh'))]
print(len(h), 'headers', sum(z.getinfo(n).file_size for n in h) / 1e6, 'MB')
print([n for n in h if '_cuda/' in n or '_cccl/' in n or '_jitify/' in n])
EOF

# 5) 过滤器单测（无需构建）
python - <<'EOF'
import sys; sys.path.insert(0, 'install')
from cupy_builder.cupy_setup_build import strip_cuda_headers
xs = ['cupy/_core/include/cupy/_cuda/cuda-11/cuda_fp16.h',
      'cupy/_core/include/cupy/_cccl/cub/cub/a.cuh',
      'cupy/_core/include/cupy/xsf/gamma.h',
      'cupyx/scipy/ndimage/cuda/pba_kernels_2d.h']
print(strip_cuda_headers(xs, include_cccl=False))  # 只掉 cuda_fp16
print(strip_cuda_headers(xs, include_cccl=True))   # 掉 3 个
EOF
```

---

## 7. 参考

- 打包权威文档：[Package.md](Package.md)、[GithubPackagingAndPypi.md](GithubPackagingAndPypi.md)
- CUDA 编译 + Ascend wheel 端到端验证：[../CudaBuild.md](../CudaBuild.md)
- 上游合并口径：[UpstreamTrackingStrategy.md](UpstreamTrackingStrategy.md)（本组改动用 `_ascend_build` / 环境变量门控，保持 diff 最小）
- 代码：`install/cupy_builder/cupy_setup_build.py`（过滤器）、`setup.py`（package_data + sdist cmdclass）、
  `MANIFEST.in`（prune 兜底）、`tools/fetch_cuda_fp16_headers.py`、
  `cupy/_core/_gpu/_compile_with_cache.pyx`（运行时查找顺序）

---

## 8. 命名与商标（`numpy_xpu` / `cupy`）

**版权与名称无关，商标才与名称有关。** 版权保护代码表达，不保护目录名/包名；
CuPy 根许可是 MIT（另含 Apache-2.0/BSD-3/BSL-1.0 的第三方部分），义务是
"保留版权与许可声明" —— 与目录是否叫 `cupy` 无关。

| 项 | 风险 | 现状 |
|---|---|---|
| 目录名 `cupy` | 无（版权不覆盖名称；改不改名义务相同） | 保留原目录名，避免改上千个 `.pyx/.pxd` 中的 `cimport cupy` 与扩展模块路径 |
| 分发名 | 中：不能以 "CuPy" 为产品名 | `numpy-ascend`（`ASCEND_DISTRIBUTION_NAME`），PyPI 与 wheel 名一致 |
| 导入名 | 中：`import cupy` 占坑易被误认为官方包 | 对外主名 `numpy_xpu`（`numpy_xpu/__init__.py` 经 `sys.modules` 别名到 `cupy`，同一模块对象）；`cupy` 保留为兼容层 |
| 元数据 | 高：上游维护者邮箱/姓名写成包作者属不实陈述 | 已改为本 fork 维护者；上游作者仅在 `NOTICE` 以版权行署名 |
| 声明 | 高：缺 NOTICE → Apache-2.0 §4 / MIT 义务不完整 | 根目录 `NOTICE`（衍生作品、修改声明、商标声明），`license-files` + `MANIFEST.in` 保证随包分发 |

商标合规的判定标准是**是否造成来源混淆**，而不是目录名。因此：
不要使用 CuPy 的 logo/品牌做产品名；README 顶部保留 "Unofficial fork / 非官方衍生版本"
声明；"CuPy-compatible" 属描述性使用（nominative use），可以继续用。

长期（2.0 目标）：发布物只提供 `numpy_xpu`，`cupy` 通过可选 extra 或
`.pth`/别名包反向提供，彻底消除 `import cupy` 占坑问题。
