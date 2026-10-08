

# Package.md — numpy-ascend conda 打包经验（CANN 9.0 实战总结）

> **用途**: 记录 2026-10-07 conda 打包 CANN 9.0 的完整流程、踩坑与验证结论，
> 供后续打包（8.5/9.1 train、CI 自动化）复用。
> **产物**: `conda/meta.yaml` + `conda/build.sh`（已提交），
> `dist/conda/linux-64/numpy-ascend-cann90-14.0.0a1-py311_0.tar.bz2`（首个可安装产物）。

---

## 1. 包命名与版本策略

- 包名 **`numpy-ascend-cann90`**，与 wheel 命名（`setup.py` 的
  `_ascend_distribution_name()` → `numpy-ascend-cann85/cann90`）对齐。
  **CANN release train 之间 ABI 不兼容**（库集合按 train 重新选择，见
  `install/cupy_builder/features/ascend.py`），所以按 train 拆包名，
  而不是只放 build string。
- import 包名保持 `cupy`/`cupyx`/`cupy_backends`（与 wheel 的 distclass
  改名技巧一致），用户代码 `import cupy` 无感。
- conda 包版本 jinja 从 `cupy/_version.py` 提取：

```jinja
{% set version = load_file_regex(load_file="cupy/_version.py",
       regex_pattern="__version__ = ['\"]([^'\"]+)['\"]").group(1) %}
```

## 2. CANN 的 conda channel 事实（2026-10-07 核实）

| Channel | 包名 | 有 9.0.1? |
|---|---|---|
| `conda.anaconda.org/ascend` | `cann-toolkit` / `cann-950-ops` | ✅（9.0.0 / 9.0.1）|
| `repo.huaweicloud.com/ascend/repos/conda` | `ascend-cann-toolkit` / `ascend-cann-950-ops` | toolkit ✅ / **950-ops 只有 9.1+** ❌ |

**坑 1**：两个 channel 的名字都叫 "ascend"，`-c ascend` 有歧义；
`conda search`/`conda list` 的 channel 列还会用短名显示，误导排查。
**一律用完整 URL 显式传 channel。**

**坑 2**：conda 包本体是 **6 个文件的包装器**（activate 脚本 + 引导），
真正的 CANN 树（`$CONDA_PREFIX/Ascend/cann-9.0.1/`）不在包文件清单里，
`conda-meta/*.json` 查不到 lib64 归属——别用 conda-meta 判断某个 .so 属于
toolkit 还是 ops 包。

## 3. 版本约束的预发布语义坑

`cann-toolkit 9.1.0.beta.3` 在 conda 版本序里**满足 `<9.1`**（beta 是
prerelease）。9.0 train 构建的包若写 `<9.1`，dry-run 会拉进 9.1 beta：

```yaml
- cann-toolkit >=9.0.1,<9.1.0a0     # <9.1.0a0 才能排除 9.1 train 的 beta
- cann-950-ops  >=9.0.1,<9.1.0a0
```

## 4. 配方（conda/meta.yaml + build.sh）要点

- `build.script_env` 透传 `CUPY_INSTALL_USE_ASCEND=1` 与外部
  `ASCEND_HOME_PATH`（外部值优先于 host env 的 activate 脚本）。
- `build.sh` 里对 `ASCEND_HOME_PATH` 做 fallback：cann-toolkit 的
  activate.d 没导出时，探测 `$CONDA_PREFIX/Ascend/cann-*`，并校验
  `version.cfg` / `compiler/version.info` 存在（与
  `install_build.check_cann_version` 的探测文件一致）。
- host 依赖 = `cython >=3,<3.2`、`numpy >=2.0,<2.6`、`fastrlock`、
  `setuptools >=77`、两个 cann 包；run 依赖同 pyproject
  （`numpy >=1.24,<2.6`）+ `pin_compatible(..., max_pin='x.x')`。
- 安装用 `pip install . --no-deps --no-build-isolation`（
  `setup.py develop` 在 setuptools 80 已废，见 Memory.md §1）。

构建命令（**显式双 channel**）：

```sh
conda build conda/ --output-folder dist/conda \
    -c https://conda.anaconda.org/ascend -c defaults
```

## 5. 验证结论（2026-10-07 实测）

conda-build 流程走到 **build 阶段全绿**：

1. meta 渲染 → host env 求解（真实拉下 cann-toolkit/cann-950-ops 9.0.1）
2. Cython + C++ 全量编译通过（~4 min，含 AOT Ascend910B4 内核 staging）
3. wheel 构建成功装入 build env
   （`numpy_ascend_cann90-14.0.0a1-cp310-...-linux_x86_64.cann9.0.whl`）

**未走通**：build 成功后的打包/清理阶段，conda 删除自己的 lock 文件
（`~/miniconda3/locks/*`）会触发 IDE 批量删除保护把进程杀掉（复现 3 次，
无一例 traceback，日志戛然而止）。**在 IDE 外的普通终端跑同一命令即可
走通全自动路径**；IDE 内会话用下面的手工路径。

## 6. 手工路径：wheel → conda 包（绕开 conda-build 清理）

```sh
# 1) 构建 wheel（复用既有 .cpp，增量快）
export CUPY_INSTALL_USE_ASCEND=1
export ASCEND_HOME_PATH=$CONDA_PREFIX/Ascend/cann-9.0.1
python -m pip wheel . --no-deps --no-build-isolation -w dist/wheelhouse

# 2) 按 conda 布局解包
STAGE=/tmp/condapkg_stage
mkdir -p $STAGE/lib/python3.11/site-packages
pip install --no-deps --target $STAGE/lib/python3.11/site-packages dist/wheelhouse/*.whl

# 3) 写 info/（index.json 的 depends 与 meta.yaml 保持一致）
#    files 列表用 os.walk 生成（排除 info/ 自身）

# 4) 打 legacy .tar.bz2 —— 注意 tar 项不能带 './' 前缀！
cd $STAGE && tar --owner=0 --group=0 -cjf <out>.tar.bz2 info lib

# 5) 生成 channel 索引并验证
conda index dist/conda
conda create --dry-run --prefix /tmp/t -c file://$PWD/dist/conda \
    -c https://conda.anaconda.org/ascend -c defaults numpy-ascend-cann90
```

**坑 3**：`tar -cjf out.tar.bz2 .` 产生的 `./info/index.json` 前缀会让
`conda index` **静默产出空 repodata**（无任何报错）。必须 `tar ... info lib`
这种无前缀形式。检查方法：直接读 `repodata.json` 的 `packages` 是否非空。

**坑 4**：wheel 若从 `build/bdist.linux-x86_64/wheel/` 残载荷目录手工
`python -m wheel pack .` 恢复，platform tag 顺序可能与 `bdist_wheel`
子类的规范输出（`linux_x86_64.cann9.0`）不同（得到 `0.cann9.linux_x86_64`）。
pip 按点分段解析 tag，直接文件安装不受影响，但发 PyPI 前应走正常
bdist_wheel 路径。

**坑 5**：wheel 构建前必须清理陈旧的 `build/` 目录——setuptools 的
build_py/bdist 会对其中几千个旧文件做删除重写，触发保护或失败
（`rm -rf build`）。

## 7. 运行时行为注意（发布包 vs 本地开发）

- 发布包内 `runtime.pyx` 的 **`initialize_backend(0)` 是激活的**：
  import 时即初始化 Ascend runtime——**无 NPU 的机器上 `import cupy` 直接
  抛 `EL0003 init soc version failed`，属预期**。数值/导入验证需在有
  910B 的机器上做。
- 本地无 NPU 开发用未提交的本地 patch（注释掉该行）做开关；
  打包前必须 `git stash` 收起它，打完 `git stash pop` 恢复。

## 8. 用户安装方式

```sh
conda install \
    -c file:///path/to/numpy-ascend/dist/conda \
    -c https://conda.anaconda.org/ascend \
    numpy-ascend-cann90
```

（发布到公网后，把本地 channel 换成托管 channel；ascend channel 必须保留，
因为 cann-toolkit/cann-950-ops 运行时依赖来自那里。）

## 9. 待办

- [ ] CI 上用无删除拦截的环境跑通全自动 `conda build`（含 test 阶段的
      import 冒烟——需 NPU runner）
- [ ] `conda index` 后把 `dist/conda` 发布到可公网访问的 channel
- [ ] 8.5 train 复用本配方：复制 meta.yaml 改名 `numpy-ascend-cann85` 并
      核对 8.5 的 cann 包在哪个 channel 有 9.0.1 等价物
- [ ] sdist（`dist/cupy-14.0.0a1.tar.gz`，PEP 625 命名尚是 `cupy-*` 而非
      `numpy-ascend-*`）与 conda source 的对接（`source: url:` 指向 sdist，
      避免 CI 拷贝整个 work tree）
