# `_core` 共享 pyx 的后端拆分方案（_routines_indexing.pyx 等）

> 状态：**方案，未执行**（2026-09-20）。触发问题：
> 「`_routines_indexing.pyx` 这样的 pyx 能否拆分为多个 pyx，每个 backend 一个，
> 只把共享的放在 `_core/` 下面？」
>
> 相关文档：`dtype_promotion.md`（M-D1 死分支的教训，§6 风险 1 的先例）、
> `arg_passing_plan.md` §2.2.1（操作数与参数分区——拆分不得破坏）。

---

## 0. 结论

1. **「每个 backend 一个 pyx」的机制已经存在**：`install/cupy_builder/features/
   {ascend,cuda,rocm}.py` 各自维护一张「模块名 → 源文件」映射表，构建时按所选
   backend 把对应源文件编译到**同一个模块名**下。`cupy._core._kernel`
   （← `_ascend/_kernel.pyx`）、`_routines_linalg`、`_routines_sorting`、`raw`
   已经走了这条路。**拆分 = 在映射表加一行，不需要发明任何新机制。**
2. 但 **`_routines_indexing.pyx` 不应该整文件复制成双份**：实测 1245 行中
   后端相关内容仅 23 处（<2%，且多为本次 scatter 工作加的 dtype 门槛），它的
   "后端性"已经通过**按名派发**解决（`cupy_scatter_*` kernel → `ascend_scatter_*`
   注册表 → aclnn）。双份拷贝是负收益（见 §6 风险 1 的真实先例）。
3. **修正（2026-09-20 二稿）**：初稿曾判断「IF 倒挂」，**错误**——
   `_command.py:141` 的 `CUPY_CANN_VERSION = 0` 只用于 `use_stub`（RTD）分支；
   真实构建由 backend descriptor 探测：`backends/ascend.py:264`
   `('CUPY_CANN_VERSION', str(self.get_version()))` → `build.get_cann_version()`。
   所以 62 个 IF 在 Ascend 构建下语义正确：56 个 `<=0`（CUDA-only）编译掉、
   6 个 `>0`（Ascend 分支）正常编译。IF 的问题降级为**可维护性**（§1/§5），不是正确性。
4. **真实发生的事故（导出 scatter 方法的缩进错误）**：手工删除
   `IF CUPY_CANN_VERSION <= 0:` 导出 `scatter_add/max/min` 时，`_scatter_op`
   留在原深层缩进上变成 `scatter_min` 的**嵌套函数**——三个方法与
   `cupy.add.at` 一调用就 AttributeError（已修）。这是 §1 把 IF 列为
   「可维护性差」的实证：**删 IF 比留 IF 更容易出错**（内部所有行都要反缩进）。

---

## 1. 现状：三种后端选择机制并存

| 机制 | 用法 | 现状评价 |
|---|---|---|
| **A. 构建期「模块名 → 源文件」映射** | `features/ascend.py`：`('cupy._core._kernel', ['cupy/_core/_ascend/_kernel.pyx'])`；单源可简写 `'cupy._core._routines_creation'` | ✅ **目标机制**。已用于 `_kernel`/`_routines_linalg`/`_routines_sorting`/`raw`；同一模块名保证上游 API 不变 |
| **B. 编译期 `IF CUPY_CANN_VERSION`** | 62 处 / 10+ 个 pyx（56 个 `<=0`、6 个 `>0`） | ✅ 语义正确：版本由 backend descriptor 构建期探测（`backends/ascend.py:264`；`_command.py:141` 的 0 仅用于 use_stub/RTD）。扣分项是**可维护性**——手工删/改 IF 极易出缩进错误（scatter 导出事故，§0.4），且 IF 另一侧永远无法被当前构建的回归覆盖 |
| **C. 运行期 `is_ascend()`** | `is_ascend()` 读 backend descriptor（`runtime.pyx:125` `_is_ascend = ascend_environment`）；本会话新增的 `_ascend_runtime()`（`_routines_indexing.pyx`，带缓存） | ✅ 适合行为分叉（dtype 白名单、组合实现选择），不适合"设备 API 存在性" |

`cupy/_core/__init__.py` 顶层还有一处运行期分支（`if not runtime.is_ascend():`
选 `_gpu/_accelerator`+`_gpu/fusion`，否则 `fusion_stub`）——与 A 等价的运行期版本，
只适合纯 python 层的选择。

---

## 2. 判据：什么拆、什么不拆

| 代码内容 | 归属 | 理由 |
|---|---|---|
| 形状/索引/broadcast 数学、public API（take/put/choose/diagonal/nonzero/...） | `_core/` 共享 | 后端无关 |
| 按名派发的 ElementwiseKernel 定义（`cupy_scatter_add` 等，CUDA body 只是**字符串**） | `_core/` 共享 | Ascend 按 name 查 `ascend_scatter_*` 注册表，CUDA body 在 Ascend 上从不执行 |
| 行为策略分叉（dtype 白名单、组合实现选择） | `_core/` 共享 + 运行期 `is_ascend()` | 分叉通常只有几行 |
| 直接调 aclnn 的 C++ wrapper | `cupy/backends/ascend/` | 已就位，不随本次改动 |
| ufunc/Kernel 派发机器 | `_core/_ascend/_kernel.pyx`（CUDA 版在 `_gpu/`） | ✅ 已按 A 机制拆好 |
| 真正无法共享（cuda runtime/driver 调用、texture、NVTX、graph、cuBLAS 句柄） | per-backend 源文件（`_gpu/` / `_ascend/`） | 唯一值得整段拆的类别 |

**新增后端分叉的门槛**：只有当「共享版必须写 `if is_ascend()` 且分叉超过 ~20 行、
或牵涉设备 API 存在性」时才拆出 per-backend 源文件；否则一律共享 + 运行期判据。

---

## 3. 对 `_routines_indexing.pyx`（1245 行）的具体裁决

| 内容块 | 行数占比 | 裁决 |
|---|---|---|
| public API（take/put/choose/diagonal/nonzero/roll...） | 大头 | 共享，不动 |
| `_scatter_op`/`_scatter_op_single` 控制流、mask 索引准备 | 中 | 共享，不动 |
| `cupy_scatter_*` / `cupy_getitem_*` kernel 定义 | 中 | 共享（按名派发），不动 |
| dtype 门槛（`_ascend_runtime()` 收窄白名单） | ~30 行 | 共享 + 运行期判据（本次已实现），不动 |
| 直接调 aclnn 的逻辑 | **0** | 无 |

⇒ **维持单文件共享，不拆**。它已经是「共享为主、按名派发、运行期策略」三件套的
标准形态，反而是其他文件的模板。

若未来真出现 >20 行的 Ascend 分叉（例如想把 scatter 的组合逻辑下沉到 C++ 之外的
pyx 层），按 §4 checklist 拆出 `cupy/_core/_ascend/_routines_indexing.pyx` 并在
`features/ascend.py` 加
`('cupy._core._routines_indexing', ['cupy/_core/_ascend/_routines_indexing.pyx'])`——
机制零改动。

---

## 4. 拆分某个文件时的 checklist

1. **`.pxd` 拆分**：`_ndarray_base`/`shape_t` 等 cimport 链以共享 `.pxd` 为锚
   （`_core/_kernel.pxd` 现在在 `_gpu/` 下——共享化时要挪回 `_core/`）。
2. **循环导入审计**：`core.pyx ↔ _routines_*.pyx` 双向（`import cupy._core.core
   as core` 这类）——per-backend 文件只允许单向依赖共享层。
3. **features 映射表**：加 `('cupy._core.<name>', ['cupy/_core/_ascend/<name>.pyx'])`；
   CUDA 侧同步确认默认映射（`features/cuda.py`）。机制本身已支持，无需改 Extension 逻辑。
4. **旧 `.so` 残留**：模块源文件路径变了，旧的 `cupy/_core/<name>*.so` 会遮蔽新构建
   ——拆分提交里必须带 `build clean` 说明或删除旧产物。
5. **re-export 面不变**：`cupy/_core/__init__.py` 的 `from cupy._core.<name> import ...`
   一行不改（模块名不变的意义就在这里）。
6. **两侧对照测试**：凡共享逻辑上提到共享层之后，给 per-backend 文件加
   「与 `_gpu/` 版本行为对照」的测试或注释锚点（防漂移，见 §6.1）。

---

## 5. 里程碑

| 步骤 | 内容 | 工作量 | 验证 |
|---|---|---|---|
| ~~M-S0 / M-S1~~ | ~~IF 倒挂审计、修版本号取值~~ **撤销**（初稿误判：`CUPY_CANN_VERSION` 由 backend descriptor 构建期探测，见 §0.3 修正） | — | — |
| M-S2（P2，可选） | IF 可维护性清理：「行为分叉」类 IF 换成运行期 `is_ascend()`（参照 `_scatter_op_single` 的 `_ascend_runtime()` 缓存范式）；「设备 API 存在性」类保留编译期，可选换成语义明确的宏（如 `IF CUPY_BACKEND_CUDA`） | 1-2d | 分批重编 + pytest |
| M-S3 | （可选）真分叉的文件级拆分：只对出现 >20 行 Ascend 分叉的文件执行 §4 checklist | 按需 | import 图无环 + pytest |
| M-S4 | backend 归属表文档：`cupy/**` 每个模块的「共享 / _gpu / _ascend / backends」归属 + 选择机制说明，作为 porting 指南 | 0.5d | 文档评审 |

---

## 6. 风险

| # | 风险 | 说明 | 对策 |
|---|---|---|---|
| 1 | **双实现漂移**（最大风险） | 真实先例：`_kernel.pyx` 的 CUDA 版（`_gpu/_kernel.pyx:1374`）是 `from_numpy_scalar_with_dtype(x, t)`，`_ascend/` 版抄错成死分支——dtype_promotion M-D1 修的就是它。整文件复制必然重演 | 共享逻辑尽量上提，per-backend 文件尽量薄；拆分同时加对照锚点（§4.6） |
| 2 | **手工删 IF 的缩进错误**（已发生，2026-09-20） | `IF ...:` 块整体删除时内部所有行都要反缩进；scatter 导出时 `_scatter_op` 漏调，嵌进 `scatter_min` 成局部函数 → 三个 `scatter_*` 方法与 `cupy.add.at` 运行期 AttributeError，且恰好被撤销 commit 的操作扫进另一笔提交 | 删 IF 用机械步骤（只删 IF 行、方法体整体反缩进一级）；或先把 IF 换成运行期判定再删（M-S2） |
| 3 | IF 两侧无法互相回归 | 编译期另一侧在本构建里不存在：CUDA 侧无 GPU、Ascend 侧无 NPU 时，回归各只能覆盖一侧 | 共享逻辑上提（§2 判据）；per-backend 文件保持薄，分叉点集中 |
| 4 | 构建时间 | per-backend 文件成对编译会拉长构建 | 拆分单位是「函数」而非「文件」，多数文件不新增编译单元 |
| 5 | upstream merge | 共享单文件 = merge 友好 ✓；per-backend 文件是 fork 特有，merge 时只需处理共享文件 | 坚持 §2 判据，克制拆分冲动 |

---

## 7. 参考

* 选择机制：`install/cupy_builder/features/ascend.py:62-110`（映射表）、
  `features/cuda.py`（默认映射）；
  `CUPY_CANN_VERSION` 取值：`backends/ascend.py:264,281`（探测的真实版本）、
  `_command.py:137-146`（仅 use_stub/RTD 置 0）
* 已按 A 机制拆分的模块：`cupy._core._kernel`、`_routines_linalg`、
  `_routines_sorting`、`raw`（ascend.py:76-90）
* IF 统计（2026-09-20）：62 处（56 `<=0` / 6 `>0`），分布文件见 §0.3
* 先例与教训：`docs/ascend/dtype_promotion.md` §2 问题 1（双实现死分支）、
  `arg_passing_plan.md` §2.2.1（拆分不得破坏操作数/参数分区）
* 现状 runtime 分叉范式：`cupy/_core/_routines_indexing.pyx::_ascend_runtime()`
  （带缓存的惰性判定）、`cupy/_core/__init__.py`（fusion/accelerator 选择）
