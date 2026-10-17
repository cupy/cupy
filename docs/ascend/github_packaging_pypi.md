# github release and pypi publish
下面给一套“人工触发 → 多 wheel 构建 → 打 GitHub Release → 自动传 PyPI”的可落地计划，按你用 `quay.io/ascend/cann` 给 numpy-ascend 打包来设计。核心原则：**构建用 CANN 容器、发布用 OIDC Trusted Publishing、人工触发只出包/可只发 TestPyPI、正式发 PyPI 再走 release 或二次确认**。

[ascend/cann · Quay](https://quay.io/repository/ascend/cann)

```bash
export CANN_REPO=quay.io/ascend/cann
export CANN_TAG=9.2.0-beta.2-a3-ubuntu22.04-py3.12

docker run \
    --name cann_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it $CANN_REPO:$CANN_TAG bash
```

### trigger a publish

```
git tag v14.0.0a1        # tag 名与包版本 14.0.0a1 对齐（PEP 440）
(aigent) qingfeng@JinzePC:~/numpy-ascend$ git push origin v14.0.0a1
```


## 一、总体流水线

1. 人工在 GitHub Actions 点 

   ```
   workflow_dispatch
   ```

   ，填参数：

   - `cann_tags`：如 `9.1.0-910b-ubuntu22.04-py3.12,9.0.1-a3-ubuntu22.04-py3.11,8.2.rc1-910b-ubuntu22.04-py3.11`
   - `pythons`：如 `3.10,3.11,3.12`
   - `archs`：`x86_64,aarch64`
   - `target`：`build-only / testpypi / pypi`

2. 矩阵展开：每个 

   ```
   (cann_tag × python × arch)
   ```

    起一个 job

   - 拉 `quay.io/ascend/cann:<tag>`
   - 容器内 source CANN、设 Ascend 环境变量、编译 wheel
   - 无 NPU 只保证“编译+链接+import 成功（L1–L3）”，不跑真实数值

3. 收集所有 wheel（+1 个 sdist job）为 artifact

4. `build-only`：只上传 artifact，不发布 
   `testpypi`：传 TestPyPI 做安装冒烟 
   `pypi` 或 `release: published`：传正式 PyPI + 建/更新 GitHub Release 附 wheel

> 建议 wheel 命名带 CANN train，如 `numpy_ascend-14.0.0a1+cann90-cp311-cp311-manylinux_2_28_x86_64.whl`（PyTorch 方案：train 编码为 PEP 440 **本地版本段** `+cann90`，平台标签保持标准 manylinux，由 CI 的 auditwheel repair 产出）；sdist 不绑 CANN 版本，全 Python/CANN 通用 。

## 二、CANN 镜像与矩阵选择

`quay.io/ascend/cann` 常见 tag 模式：`<CANN>-<soc>-<os>-py<ver>`，soc 有 `910b`、`a3`、`950` 等，os 有 `ubuntu22.04/24.04`、`openeuler24.03`，类型有 runtime/devel/toolkit 等。

- 只编译 wheel：用含 toolkit 的 devel 镜像（如 `9.1.0-950-ubuntu22.04-py3.12-devel`、`9.0.1-910b-ubuntu22.04-py3.11` 一类），里面带 compiler/ops/headers
- 旧 8.x：公开镜像多为 `arm64` 910b（8.0/8.1/8.2 示例都是 aarch64） ，x86 要自己确认 tag 或用自托管臂量机
- 架构对应 runner：
  - x86_64 → GitHub `ubuntu-latest`（标准分钟计费/公库免费）
  - aarch64 → `ubuntu-24.04-arm`（若可用）或自托管 910B/Atlas 臂量机；别用 QEMU 编大 CANN 项目，极慢
- 多 CANN 同发：单一 PyPI 项目 `numpy-ascend`，train 用本地版本段区分（`+cann90`/`+cann8.5`），文件名不冲突，且 `pip install numpy-ascend==14.0.0a1+cann90` 可精确选 train

## 三、容器内构建环境变量（按 numpy-ascend 调整）

如果是 cupy 系 numpy-ascend：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 容器里通常在装路径
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CUPY_INSTALL_USE_ASCEND=1
# 若 AOT AscendC 内核要预编某 SoC（有 NPU/或离线指定）
export CUPY_ASCEND_KERNEL_SOCS=Ascend910B4,Ascend910C
# 依赖有特定的版本, see AGENTS.md
python -m pip install -U pip build cython
python -m build --wheel --no-isolation -o /out
```



```
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
cmake … && python -m build --wheel
```

无 NPU 机器：编译/链接完做 `python -c "import cupy; backends.ascend.check_cann_version()"` 这种非执行校验即可；真跑 sum/matmul 要 910B runner 。

## 四、GitHub Actions 示例（精简可改）

> **已落地**: `.github/workflows/build-cann-wheels.yml`（wheel 构建专用，无 NPU 不跑测试，
> 容器内做 `import cupy._core` 冒烟检查）。
>
> **触发时机**（普通 push / PR 不触发）：
>
> | 触发方式 | 行为 |
> |---|---|
> | 人工 `workflow_dispatch`，`target=build-only`（默认） | 只构建，产物存 artifact，不发布 |
> | 人工 `workflow_dispatch`，`target=testpypi` | 只构建（冒烟装包自行下载 artifact 验证），不发布 |
> | 人工 `workflow_dispatch`，`target=pypi` | 构建 + 发布到正式 PyPI |
> | `git tag vX.Y.Z` 后 push tag（`v*`） | 自动构建 + 发布 PyPI + 自动创建 Release 挂 wheel/sdist |
> | GitHub 上发布 Release（`release: published`） | 自动构建 + 发布 PyPI + 把 wheel/sdist 挂到该 Release |
>
> 两步发布建议：先 dispatch `build-only` 拿 wheel 在本地 910B 冒烟，确认后
> `git tag vX.Y.Z && git push --tags` 自动出正式包。
>
> 相对下面示例修正的问题：
>
> 1. 矩阵展开的 `fromJson(format(..., replace(...)))` 不可用 —— GitHub 表达式没有
>    `replace()`，改为前置 `gen-matrix` job 用 `jq` 把 CSV 输入转成 JSON 数组。
> 2. `update-alternatives --set python3 /usr/bin/python$py` 不可靠 —— 改为容器内用
>    `uv python install <ver>` 装 CPython 再建 venv 构建（镜像自带的 Python 版本
>    由镜像 tag 固定，如 `...-py3.11`）。
> 3. `--no-isolation` 需要预装构建依赖：`setuptools>=77, wheel, Cython>=3,<3.2,
>    fastrlock, numpy`（与 `pyproject.toml [build-system].requires` 保持一致）。
> 4. 容器以 root 运行，`dist/` 属主为 root，构建后需 `chown` 才能 upload-artifact。
> 5. import 冒烟检查必须先安装 wheel 并在 `/src` 之外执行，否则源码树里的
>    `cupy` 包（无编译产物）会遮蔽安装好的包。
> 6. publish 的 `if`/`environment` 对 `release`/`push tag` 事件（无 `inputs.*`）的
>    取值做了兼容：这两个事件下 `inputs.target` 为 null，`inputs.target !=
>    'build-only'` 恒为 true，必须按事件名显式判断。
> 7. sdist job 曾报 `Your Cupy building environment is invalid`：`python -m
>    build --sdist` 仍会执行 `setup.py`，旧代码只在 `dist_info`/`egg_info` 时
>    跳过 ext_modules 构造，sdist 会走 `get_ext_modules` → 用默认 CUDA 后端
>    探测工具链（runner 上无 nvcc/CANN）而失败。已在 `setup.py` 把 `'sdist'`
>    加进跳过列表（sdist 只按 `MANIFEST.in` 枚举 `.pyx` 源文件，不需要
>    extensions）。
>
> **fork 里其它 workflow 的触发情况**（继承自上游 cupy，见 `.github/workflows/`）：
>
> | workflow | 触发 | fork 中是否需要 |
> |---|---|---|
> | `build-cann-wheels.yml` | dispatch / push tag `v*` / release | 本文档所述，Ascend 发布主流程 |
> | `pretest.yml`（"Pre-review Tests"） | 每次 push / pull_request | 只保留 `static-checks`（pre-commit + mypy）；`build-cuda`/`build-rocm` 是上游残留，已加 `if: github.repository_owner == 'cupy'` 守卫，在 fork 中显示为 skipped |
> | `publish.yml`（"Publish to PyPI"） | 仅 workflow_dispatch，且面向上游 cupy/cupy-release-tools | 与 Ascend 无关，不触发、可忽略（或禁用） |
> | `docker.yml` | release / PR 改 docker/** / dispatch | 上游 cupy Docker 镜像，与 Ascend 无关 |
> | `flexci.yml`、`self-hosted-ci.yml`、`pr-*.yml` 等 | PR 评论 `/test`、PR 状态等 | 上游 CI 联动，fork 中不会实际执行 |
>
> 所以 push 后看到 `build-cuda`/`build-rocm` 在跑，不是 `publish.yml`，而是
> `pretest.yml` 的 `on: [push, pull_request]` 触发的。



要点：

- 人工 dispatch 默认 `build-only` 不出 PyPI；发正式版走“打 tag → Draft Release → Publish”，`release: published` 自动传 PyPI 并挂附件。
- 用 OIDC 不需要把 PyPI token 放 secrets；PyPI 后台加 Trusted Publisher，填 owner/repo/workflow=release-cann.yml、可绑定 environment `pypi` 。
- TestPyPI 单独 environment `testpypi`，dispatch 选 `testpypi` 先冒烟。

## 五、发布门控与防错

- 版本号单一来源：只在 `pyproject.toml` 改，`git tag vX.Y.Z` 与版本一致；CI 可加一步校验 tag≠已存在 PyPI 版本。
- 人工两步发布更稳：dispatch 出 wheel → 人下载装测试 → 打 release 自动传 PyPI。比“一次 dispatch 直接传生产”安全。
- 多 CANN 同版 Python 不覆盖：wheel 名带本地版本段 `+cann9.0/+cann8.5`；sdist 只一个、不写 CANN 后缀 。
- 无 NPU runner 只做“构建+import 自检”；L4 数值测试放自托管 910B job，且不在 dispatch 默认矩阵里跑，避免耗免费额度。
- quay 拉取：公共 tag 一般可匿名拉；若频繁 429，CI 里 `docker login quay.io -u robot -p …` 用 robot account，或把基础镜像缓存到 GHCR。
- 大镜像缓存：用 `docker/build-push-action` 或 registry cache，别每次全拉 4–13GB 的 CANN 镜像。

## 六、最小落地顺序

1. 本地用目标 CANN 容器跑通 `python -m build --wheel`，确认环境变量和 wheel 名。
2. 写上面 workflow，dispatch 只 `build-only`，验证 x86 一个 CANN tag + 一个 Python 出 wheel。
3. PyPI 建项目→Trusted Publishing→environment `pypi`/`testpypi`。
4. dispatch `testpypi` 冒烟安装；再打 tag 发 release 走生产。
5. 扩 aarch64：GitHub 公开仓库免费提供 `ubuntu-24.04-arm` runner（arm64 原生，非 QEMU），
   dispatch 时 `archs=aarch64` + aarch64 可用的 CANN tag 即可；注意同一 tag 会同时被
   x86 和 arm job 拉取，需该 tag 在 quay 上有多架构 manifest（或另传 arm 专用 tag）。

## 七、平台标签与 CANN train：manylinux_2_28 + 本地版本段（2026-10 定稿）

**当前产物命名**（CI + `setup.py`）：

```
numpy_ascend-14.0.0a1+cann9.0-cp311-cp311-manylinux_2_28_x86_64.whl
numpy_ascend-14.0.0a1+cann8.5-cp311-cp311-manylinux_2_28_x86_64.whl   # 同一 PyPI 项目，另一 train
```

两个设计决策，都是被 PyPI/ pip 的现实约束逼出来的：

1. **CANN train = PEP 440 本地版本段（PyTorch 方案）**，早期方案是把它拼进
   平台标签（`manylinux_2_17_x86_64.cann9.0`）。拼标签版有两个问题：
   - pip 从索引解析时**无法按 train 选择**（两个 train 的 wheel 都被判"兼容"，
     任意选一个），用户没有命令行手段指定；
   - 只有直链/本地文件安装时 pip 才接受带点标签，行为不一致。
   改为本地版本段后：
   - `pip install numpy-ascend==14.0.0a1+cann9.0` 精确选 train；
   - `pip install numpy-ascend==14.0.0a1` 匹配任意本地段，pip 取本地段最大者
     （`+cann9.0` > `+cann8.5`，确定性默认）；
   - 平台标签回归标准 manylinux，wheel 完全合规。
   实现在 `setup.py::_make_cmdclass` 的 `bdist_wheel` 子类：在
   `finalize_options()` **最早**（`super()` 之前，因为 wheel 的
   `finalize_options` 会预先 finalize `egg_info` 并快照版本）把
   `+cannX.Y` 追加到 `distribution.metadata.version`。只作用于 wheel；
   sdist 与运行时 `cupy.__version__` 保持裸版本（tarball 与 CANN 无关）。
   同一版本两个 train 的 wheel 文件名不同，可共存于单一 PyPI 项目
   `numpy-ascend`，不再需要 per-train 项目名（`numpy-ascend-cann90` 方案已废弃）。

2. **平台标签 = manylinux_2_28**：`bdist_wheel` 默认按构建机打
   `linux_x86_64` / `linux_aarch64`，**不做 glibc 兼容检测**，而 PyPI
   （Warehouse）**拒绝** `linux_*` 平台标签的 wheel。因此在 CI 的容器构建
   之后增加 `auditwheel repair` 一步（见 `build-cann-wheels.yml`）：

   ```bash
   uv pip install auditwheel
   auditwheel repair --plat manylinux_2_28_$TARGET_ARCH \
       --exclude '/usr/local/Ascend/*' --exclude '*/Ascend/*' \
       dist/*.whl -w /src/wheelhouse
   ```

   要点：

   - **CANN 运行时库绝不 vendor 进 wheel**：它们必须来自目标机的 CANN 安装
     （ABI 与 toolkit 匹配）。auditwheel 的 `--exclude` 同时匹配 SONAME 与
     **解析后的真实路径**（fnmatch），所以按安装路径 `/usr/local/Ascend/*`
     整体排除即可覆盖 `libascendcl`/`libopapi`/`liboptiling`/`libcann_*` 及
     其全部传递依赖，无需维护一长串库名名单。
   - `libstdc++`/`libgcc_s` 等系统库不在排除名单里，auditwheel 会按标准
     manylinux 行为 vendor 进 wheel——这同时缓解了 Package.md 里
     "目标机 libstdc++ 过旧导致 `GLIBCXX_3.4.32 not found`" 那类问题。
   - **repair 失败 = wheel 的符号基线高于 glibc 2.28**（CANN 官方镜像基于
     Ubuntu 22.04，glibc 2.35；若扩展引用了 GLIBC_2.29+ 符号就会失败）。
     CI 此时**不强行改标签**：保留未 repair 的 wheel 供 GitHub Release 使用
     并打 `::warning::`，但 PyPI 上传会拒绝它。要修只有两条路：换更老的
     构建基础镜像，或在 `quay.io/pypa/manylinux_2_28` 容器里自行安装 CANN
     （CANN 官方只发 ubuntu/openeuler 镜像，需自行验证）。
   - 发布前可本地预检：`auditwheel show dist/*.whl` 看当前标签与最高 glibc
     符号要求；repair 后确认 wheel 内没有 CANN 库泄漏
     （`unzip -l x.whl | grep -iE 'ascend|opapi|cann'` 应为空）。
   - aarch64 侧同理产出 `manylinux_2_28_aarch64`。

**wheel 命名总览**（单一项目 `numpy-ascend`，import 包恒为 `cupy`）：

| 构件 | 命名 | 说明 |
|---|---|---|
| wheel | `numpy_ascend-<ver>+cann<XY>-cp<py>-cp<py>-manylinux_2_28_<arch>.whl` | train 在本地版本段；`setup.py` 的 bdist_wheel 子类注入 |
| sdist | `numpy_ascend-<ver>.tar.gz` | 与 CPython/CANN 均无关，只发一份 |
| CUDA/HIP 构建 | `cupy-<ver>-…` | 同一份代码的上游命名，不受影响 |

安装命令（README 同步维护）：`pip install numpy-ascend==14.0.0a1+cann9.0`。


不太适合cann, 这种基于ubuntu的镜像的, 所以希望是pip install auditwheel 然后 fix