# Profiler（aclprof）

> 状态：**活跃**（2026-10-17）。实现见
> [cupy/backends/ascend/cupy_ascend_profiler.h](../../cupy/backends/ascend/cupy_ascend_profiler.h)，
> 测试见 [tests/ascend/test_profiler.py](../../tests/ascend/test_profiler.py)。

`cudaProfilerStart/Stop` 已映射到 CANN 的 ACL profiler（`aclprof*`，见
`$ASCEND_HOME_PATH/include/acl/acl_prof.h`），因此以下三条路径在 Ascend 上可用：

- `cupy.cuda.runtime.profilerStart()` / `profilerStop()`
- `cupyx.profiler.profile()` 上下文管理器
- `cupyx.profiler.benchmark()`（内部走同一对调用）

## 1. 语义：CUDA 开关 vs CANN 会话

| | CUDA | CANN |
|---|---|---|
| 采集开关 | `cudaProfilerStart/Stop` 仅 toggle，输出目录由 nsys/nvprof 持有 | 需显式会话 `aclprofInit → CreateConfig → Start → Stop → DestroyConfig → Finalize` |
| 输出目录 | 工具侧指定 | 代码侧 `aclprofInit(path)` 指定 |
| 无 profiler 时 | no-op | 库/设备缺失时本实现退化为 no-op |

实现要点（避免踩坑）：

1. `aclprofStart(NULL)` **无效**，返回 `100000`（`ACL_ERROR_INVALID_PARAM`）。旧 stub 里
   `//return aclprofStart();  // this lead to runtime error?` 的根因是缺 `aclprofConfig`，不是缺 profiler。
2. CANN 9.x 的 `libascendcl.so` **不导出** `aclprof*`（只在 `DT_NEEDED` 里带 `libmsprofiler.so`），
   所以 `-lascendcl` 链接会报 `undefined reference to aclprofInit`；加 `-lmsprofiler` 又会让构建
   依赖某些 CANN 版本可能缺席的库。故用 `dlopen("libmsprofiler.so") + dlsym` 延迟解析
   （失败时按 soname → `$ASCEND_HOME_PATH/{lib64,x86_64-linux/lib64,aarch64-linux/lib64}` 依次尝试）。
3. `profilerStop()` 里必定 `aclprofFinalize()`，保证本次会话数据落盘；下次 `profilerStart()`
   重新 `aclprofInit`，因此可以反复 start/stop（每次一个会话）。
4. 两个调用都**幂等**，与 CUDA 一致；`aclprofCreateConfig` 返回 NULL（典型原因：没有可用 device）
   时打印一次 `[cupy ascend profiler] ...` 告警并整体禁用，不抛异常——无卡机器上也能安全调用。

## 2. 使用

环境变量（**每次 `profilerStart()` 重新读取**，可在 Python 运行时修改）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CUPY_ASCEND_PROFILER` | `1` | `0` 关闭（纯 no-op，适合无 NPU 的 CI） |
| `CUPY_ASCEND_PROFILER_PATH` | `./cupy_ascend_prof` | 结果目录（相对启动目录） |
| `CUPY_ASCEND_PROFILER_DATA_TYPE` | `ACL_PROF_ACL_API \| ACL_PROF_TASK_TIME \| ACL_PROF_AICORE_METRICS \| ACL_PROF_RUNTIME_API` | `aclprof` 的 `dataTypeConfig`，支持 `0x..` |
| `CUPY_ASCEND_PROFILER_AICORE_METRICS` | `0`（`ACL_AICORE_ARITHMETIC_UTILIZATION`） | 0..8，或 `0xFF`（`ACL_AICORE_NONE`）；非法值回落默认 |

```python
import os
import cupy as cp
import cupyx.profiler

os.environ['CUPY_ASCEND_PROFILER_PATH'] = './npu_prof'   # 必须在 start 前设
os.environ['CUPY_ASCEND_PROFILER_AICORE_METRICS'] = '1'  # 1 = PipeUtilization

with cupyx.profiler.profile():        # 进入 = aclprofStart，退出 = aclprofStop
    a = cp.random.rand(4096, 4096, dtype=cp.float32)
    for _ in range(5):
        cp.matmul(a, a)
cp.cuda.Stream.null.synchronize()
```

手动配对（与 `cudaProfilerStart/Stop` 一一对应）：

```python
cp.cuda.runtime.profilerStart()   # Init + CreateConfig + Start
... kernels ...
cp.cuda.runtime.profilerStop()    # Stop + DestroyConfig + Finalize（落盘）
```

多卡：先用 `cp.cuda.Device(n).use()` 选卡，`profilerStart()` 按当前 device
（`aclrtGetDevice`）建 config。

## 3. 查看与导出结果

```bash
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/runtime/lib64:$LD_LIBRARY_PATH

find ./npu_prof -maxdepth 3 | head          # PROF_* / device_0 / host 原始数据
$ASCEND_HOME_PATH/tools/profiler/bin/msprof --export=on --output=./npu_prof
ls ./npu_prof/mindstudio_profiler_output     # op_summary_*.csv / op_statistic_*.csv 等
```

也可用 MindStudio Insight 直接打开结果目录。`msprof` 的具体开关以本机
`$ASCEND_HOME_PATH/tools/profiler/bin/msprof --help` 为准
（9.0.1 二进制中存在 `--application/--output/--export/--parse/--ai-core/--aic-metrics/
--task-time/--ascendcl/--runtime-api/--sys-hardware-mem`）。

## 4. 不改代码的采集方式

整程序采集，再用 Insight/`--export` 解析：

```bash
msprof --application="python bench.py" --output=./msprof_out \
       --ai-core=on --aic-metrics=PipeUtilization --task-time=on \
       --ascendcl=on --runtime-api=on
msprof --export=on --output=./msprof_out
```

## 5. 在 NPU 上测试

```bash
cd /home/qingfeng/numpy-ascend
python -m pytest tests/ascend/test_profiler.py -v \
  2>&1 | tee tests/logs/$(date +%F)-logs/test_profiler.log
```

| 用例 | 是否需要 NPU | 断言 |
|---|---|---|
| `test_start_stop_without_device_is_noop` | 否（有卡则 skip） | 不抛异常、不产生文件（含幂等 Start/Stop） |
| `test_disabled_by_env` | 否 | `CUPY_ASCEND_PROFILER=0` 时纯 no-op |
| `test_profile_context_manager` | **是** | 跑 `cp.matmul` 后结果目录内出现文件（唯一能证明“真的采到”的用例） |
| `test_aicore_metrics_env_is_validated` | 否 | 非法 metrics 回落不报错 |

排错顺序：

1. 无卡机器上出现一次 `aclprofCreateConfig failed (no usable device?)` 属预期；
   `CUPY_ASCEND_PROFILER=0` 可静音。
2. 有卡但目录为空 → 查 `libmsprofiler.so` 是否可 `dlopen`（`$ASCEND_HOME_PATH/lib64`）、
   结果路径可写、`ASCEND_RT_VISIBLE_DEVICES` 是否指向有效卡。
3. `aclprofInit` 失败会返回错误码（交给 `check_status` 抛异常），告警文字见 stderr。

## 6. TODO

- `cupy.cuda.nvtx`（mark / range）→ `aclprofMarkEx` / `aclprofRangeStart` / `aclprofRangeStop`；
  新接口可考虑 `mspti`（`$ASCEND_HOME_PATH/tools/mspti`，CUPTI 风格）。
- 与 msprof/mindstudio 结果目录联动（同一 `--output` 下多次会话的目录命名）。
- 真机验证：截至 2026-10-17 仅完成 L1/L2/L3（编译、链接、无卡 no-op），
  「真卡采集 + Finalize 落盘」待 910B 上由 `test_profile_context_manager` 确认。
