# docs/ascend — 开发文档索引

> numpy-ascend（CuPy v14 fork → Ascend NPU）的全部开发文档目录。
> **新 session / 新人请从 §1 阅读路线开始**；查事实看 §0 速查表。
>
> 文档约定：每篇标注**状态**——`活跃`（持续维护）、`快照(日期)`（当日记录，
> 数字不要当现状）、`方案`（未落地/待确认）、`已修复`（问题记录）、
> `分析`/`检视`（一次性调研）。过时的事实（命名、平台标签、版本号等）
> 以 §0 和 [Package.md](Package.md) §2.2、[GithubPackagingAndPypi.md](GithubPackagingAndPypi.md) §7 为准。

## 0. 当前关键事实（速查，2026-10-17）

| 项 | 值 |
|---|---|
| 仓库 / 工作分支 | `git@github.com:qingfengxia/numpy-ascend.git` · **`ascend`**（不是 main） |
| 基线 | CuPy v14 alpha fork，NumPy 2.x；**import 包恒为 `cupy`** |
| 本地环境 | CANN 9.0.1（conda env，**无 NPU**）→ 本地最高验证等级 **L3**（编译+import+注册表） |
| 增量构建 | `export CUPY_INSTALL_USE_ASCEND=1 && python setup.py build_ext --inplace && python -c "import cupy._core"` |
| 验证等级 | L1 编译 / L2 链接 / L3 import+注册表 / L4 真机数值（需 910B） |
| **发行版名** | 单一 **`numpy-ascend`**（所有 CANN train 共用；`numpy-ascend-cann85/90` 是 2026-10-17 前的历史名） |
| **wheel 命名** | 主力 train（`setup.py::_PYPI_CANN_TRAIN`，当前 cann9.0）→ 裸版本 `numpy_ascend-14.0.0a1-cp311-cp311-manylinux_2_34_x86_64.whl`，传 PyPI；其他 train → `numpy_ascend-14.0.0a1+cann8.5-...whl`（本地版本段），**只上 GitHub Release** |
| **平台标签** | `manylinux_2_34_<arch>`（auditwheel 按实测符号基线自动判定；PyPI 拒绝 `linux_*`，也拒绝本地版本段） |
| 安装命令 | 9.0：`pip install numpy-ascend==14.0.0a1`（PyPI）；8.5：`pip install numpy-ascend==14.0.0a1+cann8.5 --find-links https://github.com/qingfengxia/numpy-ascend/releases/expanded_assets/<tag>` |
| commit 前缀 | `ASCEND [AI]:` 或 `ASCEND [AI+Human]:` |
| 算子覆盖统计 | `python tools/scan_ops.py`（tree-sitter 静态扫描，勿手数） |
| 项目总纲 | 仓库根 [AGENTS.md](../../AGENTS.md)（AI 协作指南）+ [README.md](../../README.md)（用户向） |

## 1. 阅读路线

**新人 / 新 session（按顺序）**

1. [Memory.md](Memory.md) — 交接备忘：环境、常用命令、坑、会话历史（**操作手册**）
2. [DeveloperNotes.md](DeveloperNotes.md) — 架构总览（自上而下五层）、环境搭建、无 NPU 开发、排错
3. [Progress.md](Progress.md) — 进度快照（注意是 2026-09-16 的日期快照，覆盖率用 `tools/scan_ops.py` 重新统计）
4. [AscendSpecialization.md](AscendSpecialization.md) — Ascend 与 CUDA 的行为差异/适配手段汇总（改后端前必读）

**要打包 / 发布** → [Package.md](Package.md) → [GithubPackagingAndPypi.md](GithubPackagingAndPypi.md) →（conda）[CondaPackage.md](CondaPackage.md)

**要改派发/算子层** → [code_review_ascend_backend.md](code_review_ascend_backend.md) → [code_review_ascend_core.md](code_review_ascend_core.md) → 对应专题（§2.3）

**要跟上游 cupy 合并** → [UpstreamTrackingStrategy.md](UpstreamTrackingStrategy.md) → [UpstreamMergePlan.md](UpstreamMergePlan.md)

## 2. 分类索引

### 2.1 入门与状态

| 文档 | 状态 | 内容 |
|---|---|---|
| [Memory.md](Memory.md) | 活跃（2026-10-07） | 交接备忘/操作手册：环境、命令、坑、会话历史 |
| [DeveloperNotes.md](DeveloperNotes.md) | 活跃 | 开发笔记：ascend-numpy 五层架构、环境搭建、无 NPU 开发、新平台适配、排错 |
| [Progress.md](Progress.md) | 快照（2026-09-16） | 进度数字快照（覆盖率/注册算子数/缺口）；当日记录，勿当现状 |
| [Roadmap.md](Roadmap.md) | 活跃 | 功能补全路线图（blas/sparse/solver/fft…） |


### 2.2 构建、打包与发布

| 文档 | 状态 | 内容 |
|---|---|---|
| [Package.md](Package.md) | 活跃（2026-10-17 更新命名） | **打包权威文档**：发行版命名/版本策略、wheel 结构、运行时前置（libstdc++/CANN）、安装排错 |
| [Compliance.md](Compliance.md) | 活跃（2026-10-18） | **发布物合规**：仓库/wheel/sdist 里的 CUDA 资产审计、红线（不 vendor NVIDIA 专有头）、本地获取脚本、打包过滤器与验收命令 |
| [../CudaBuild.md](../CudaBuild.md) | 活跃（2026-10-18） | **跨后端构建**：Ascend 的 xpu 抽象（4 条通道）、合规的 CUDA 编译步骤（fp16 头，12.2+ 不需要）、Ascend wheel 端到端构建与验收 A1-A5 |
| [GithubPackagingAndPypi.md](GithubPackagingAndPypi.md) | 活跃（v3，2026-10-17） | GitHub Release + PyPI 发布全流程：workflow 矩阵、OIDC Trusted Publishing、auditwheel（实测 manylinux_2_34 基线）、本地版本段方案、`--find-links` 安装 |
| [CondaPackage.md](CondaPackage.md) | 实战总结（2026-10-07） | conda 打包 CANN 9.0 全流程与坑；**conda 保留 per-train 包名**（`numpy-ascend-cann90`），已与 wheel 命名分家 |
| [InstallCannOpsPackages.md](install_cann_ops_packages.md) | 操作笔记 | 自编 CANN 额外算子包（ops-sparse / ops-fft / ops-solver…） |
| [CustomKernel.md](CustomKernel.md) | 方案（待确认） | 自定义 AscendC Kernel（对标 `cupy.RawKernel`）基础设施设计 |
| [Profiler.md](Profiler.md) | 活跃（2026-10-17） | `cudaProfilerStart/Stop` → ACL profiler（aclprof）映射：语义差异、环境变量、`msprof` 导出、NPU 测试 |

### 2.3 库级兼容（cupyx / scipy）

| 文档 | 状态 | 内容 |
|---|---|---|
| [ScipyArrayApiCompat.md](ScipyArrayApiCompat.md) | 分析（2026-10-06） |  |
| [CupyxScipyMigration.md](CupyxScipyMigration.md) | 分析（2026-10-18） | `cupyx` / `cupyx.scipy` 缺口盘点：4 个子系统级阻塞 + 248 个内联 kernel 分布、pytest 红区、覆盖率三档估算（~31 % → ~88-90 %）、落地顺序 R0-R5；含 scipy array-API 是否覆盖 sparse 的结论 |
| [DistributedMigration.md](DistributedMigration.md) | 分析+方案（2026-10-18） | `cupyx.distributed` 原理拆解（`DistributedArray` 不是拆分 `_ndarray_base`、单进程多卡）、NCCL↔HCCL API/dtype 对照、四个结构性差异（root info 4108B、store 256B、无 group、P2P 不可假设）、Tier0/1/2 移植路线与风险 |

## 3. 文档维护约定

1. **状态标注**：新文档头部注明状态与日期；`快照` 类文档只追加、不改旧数字
   （要新数字就新开一节并注明日期）。
2. **交叉引用**：一律相对路径（`[Package.md](Package.md)`）；改了文件名先
   `grep -rn "旧名" docs/` 再提交。
3. **易变事实集中维护**：发行版命名 / wheel 命名 / 平台标签 / 版本策略只在
   [Package.md](Package.md) §2.2 与 [GithubPackagingAndPypi.md](GithubPackagingAndPypi.md) §7
   描述；其它文档需要时**链接过去**，不要复制数值（复制必过期）。
4. **操作类笔记**（命令、坑）进 [Memory.md](Memory.md) 或 [DeveloperNotes.md](DeveloperNotes.md)，
   不要散落在缺陷记录里。
5. 本索引随新增/重命名/废弃文档同步更新。
