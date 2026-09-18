# Packaging numpy-ascend (`cupy`) for Ascend / CANN

This document covers three things that are routinely confused with each other:

1. **How the package cooperates with NumPy 2.x** and what "version compatible"
   means for it (spoiler: NumPy is a *runtime Python dependency only* — there is
   **no** NumPy C-ABI coupling, so the wheel is *not* NumPy-version-locked).
2. **The wheel packaging strategy** used to ship two separate wheels for
   CANN 8.5 and CANN 9.0 from one source tree, plus the wheel naming rule and
   the platform × CANN support matrix (§2.8).
3. **The hard runtime prerequisites** — C/C++ toolchain, C runtime (glibc),
   CANN installation, and `LD_LIBRARY_PATH` / `LD_PRELOAD` — together with the
   verification commands for each, and a troubleshooting table for an installed
   wheel (§3.6).

This is the **developer/packaging** document behind the short install guide in
[`README.md`](../../README.md) §3 — user-facing installation steps live there.

Related documents:
* `README.md` §3 — install guide (wheel / from source); links here for §3 (hard
  runtime prerequisites) and §2.8 (naming/support matrix).
* `DeveloperNotes.md` §1–§3 — CANN installation, including a machine without an
  NPU, toolchain/dependency setup, and build-time troubleshooting.
* `install/README.md` §4.4 — the RPATH policy in build-system terms.
* `Progress.md` — which array-API ops are ported.

---

## 1. How `cupy` / `numpy-ascend` cooperates with NumPy 2.x

### 1.1 The interaction is purely at the Python level

The port keeps CuPy's original design: **NumPy is a runtime dependency, consumed
through the Python API only.**

```
cupy/                      import numpy           -> numpy.dtype, numpy.can_cast
cupy/_core/*.pyx           (Python-level)         -> from numpy import ...
cupy/backends/ascend/*.h   (no NumPy at all)      -> only aclnn / CANN headers
```

Verification (run from the repo root):

```sh
# No .pyx/.pxd/.h file pulls in the NumPy C API:
grep -rn "cimport numpy\|arrayobject.h\|import_array\|PyArray_\|NPY_" \
     --include='*.pyx' --include='*.pxd' --include='*.h' cupy/
# Expected: no matches (the only `npy_*` symbols live in the copied
# helper `cupy/_core/halffloat.h`, which contains float16 bit-twiddling
# routines lifted from NumPy source, not the NumPy C API).
```

Concretely:

| Concern | Where | Mechanism |
|---|---|---|
| dtype objects | `cupy/_core/_dtype.pyx:33` | `cdef _dtype = numpy.dtype`, cached in `_dtype_dict` |
| casting / promotion rules | `cupy/_core/_dtype.pyx:96,118` | delegates to `numpy.can_cast` — so NEP 50 promotion semantics are inherited, not reimplemented |
| public re-exports | `cupy/__init__.py` | `from numpy import dtype, iinfo, finfo, ...` |
| NumPy-2 shims | `cupy/__init__.py:989-1049` | see §1.3 |
| zero-copy CPU bridge | `cupy/_core/dlpack.pyx` | DLPack, not NumPy C API |
| dtype → aclnn | `cupy/backends/ascend/api/acl_utils.pyx:58` | maps on `dtype.char`, Python-level only |

**Why this matters for packaging:** because there is no NumPy C-ABI surface,
the wheel does **not** have to be rebuilt per NumPy minor version, and it is not
subject to the "built against NumPy 1.x, breaks on NumPy 2.x" class of problems
that C-extension packages normally have. Whether the wheel is usable is decided
by (a) the CPU Python API present at runtime and (b) the CANN release — *not* by
`numpy.__version__`'s ABI.

### 1.2 The array protocols (`__array_*__`)

`cupy.ndarray` deliberately refuses implicit conversion to NumPy:

| Protocol | Location | Behaviour |
|---|---|---|
| `__array__` | `cupy/_core/core.pyx:1528` | **raises `TypeError`** — forces the user to call `.get()` |
| `__array_priority__` | `cupy/_core/core.pyx:477` | `100`, so `numpy_arr + cupy_arr` dispatches to CuPy |
| `__array_ufunc__` | `cupy/_core/core.pyx:1700` | present, but **only compiled when `CUPY_CANN_VERSION <= 0`** |
| `__array_function__` | `cupy/_core/core.pyx:1760` | same guard |
| `__cuda_array_interface__` | `cupy/_core/core.pyx:199` | the real zero-copy interop channel |
| `__array_interface__` / `__array_wrap__` | `core.pyx:482,1542` | TODO, not implemented |

> **Ascend-specific caveat.** In the Ascend build `CUPY_CANN_VERSION > 0`, so the
> `__array_ufunc__` / `__array_function__` implementations are *not* compiled.
> Mixing a NumPy `ndarray` and a CuPy `ndarray` in one expression therefore does
> not transparently dispatch — use explicit `.get()` / `cupy.asarray()` at the
> boundary.

### 1.3 NumPy 2.0 compatibility shims

All NumPy-2 name/behaviour differences are handled in one place,
`cupy/__init__.py:989-1049`:

```python
# 989: names that *changed* in NumPy 2.0
if _numpy.__version__ < "2":
    from numpy import bool_ as bool
    from numpy import int_  as long
    from numpy import uint  as ulong
else:
    from numpy import bool, long, ulong

# 1001: names that *moved* in NumPy 2.0
if _numpy.__version__ < "2":
    from numpy import format_parser, DataSource
else:
    from numpy.rec import format_parser
    from numpy.lib.npyio import DataSource

# 1009: names *removed* in NumPy 2.0 -> clear RuntimeError with a suggestion
```

Other, smaller adaptations: `broadcast_shapes` (NumPy ≥ 1.20, guarded with
`hasattr`), the `_deprecated_apis = ['int0', 'uint0', 'bool8']` fallback via
module `__getattr__`, and `cupy/_core/core.pyx:60` (`NUMPY_1x`).

Note that `cupy/array_api/_dtypes.py` intentionally defines its *own* promotion
table and does **not** follow NumPy's rules — that is the Array API standard's
requirement, unrelated to NEP 50.

### 1.4 Version policy

| Item | Value | Source |
|---|---|---|
| NumPy pin | `numpy>=1.24,<2.6` | `pyproject.toml:34` |
| Python | `>=3.9` (`3.9`–`3.13` classifiers) | `pyproject.toml:32` |
| fastrlock | `>=0.5` | `pyproject.toml:35` |
| CuPy version | `14.0.0a1` | `cupy/_version.py` |

The NumPy pin is an **installation-time** pin (`dependencies`), satisfied by
pip. It is *not* an ABI pin: a wheel built in an environment with NumPy 1.24
runs correctly in an environment with NumPy 2.4 and vice versa, as long as the
Python-level API used above exists. The `,<2.6` upper bound exists only to
guard against future NumPy removals of names that `cupy/__init__.py` re-exports.

**Practical consequence for releasing wheels:** ship **one wheel per CANN
release**, not one per NumPy release.

---

## 2. Packaging strategy: two wheels, one source tree

### 2.1 Goal

> One wheel for CANN 8.5, one wheel for CANN 9.0, installable side by side,
> neither silently running against the wrong CANN.

This is necessary because:

* `aclnn` operator **signatures change** between CANN releases;
* `libop_common.so` is coupled to a matching `liboptiling.so` per release;
* CANN shuffles its shared-library names between releases
  (see `install/cupy_builder/features/ascend.py`, `_CANN_LIBS_82` vs `_CANN_LIBS_85`).

A single "universal" wheel would fail at *load time* with an opaque
`undefined symbol`, or worse, segfault.

### 2.2 The wheel platform tag

`AscendBackend.get_wheel_platform_tag()` (`install/cupy_builder/backends/ascend.py:122`)
derives a tag from the detected CANN version, where the version is encoded as
`major*100 + minor*10 + patch` (CANN 8.5.1 → `851`):

```python
major, rest = divmod(version, 100)
minor = rest // 10
return f'cann{major}.{minor}'          # 851 -> 'cann8.5'
```

`setup.py:104-124` installs a `bdist_wheel` cmdclass that appends this tag to
the platform tag, producing:

```
numpy_ascend_cann85-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl
numpy_ascend_cann90-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
```

Because the platform tags differ, the two wheels are distinct artifacts to pip
and cannot silently overwrite each other.

### 2.3 Build-time metadata

`write_wheel_metadata()` (`install/cupy_builder/cupy_setup_build.py:569`)
records the build environment into `cupy/.data/_wheel.json`, which is bundled
into the wheel:

```json
{
  "backend": "ascend",
  "cann_build_path": "/home/qingfeng/Ascend/cann-8.5.1",
  "cann_version": 851,
  "cann_version_str": "8.5.1",
  "cupy_backend": "ascend",
  "cupy_version": "14.0.0a1",
  "packaging": "pip"
}
```

The file is written to a `tempfile` first and copied into `cupy/.data/` by
`prepare_wheel_libs()`, because that function wipes `cupy/.data/` before
populating it (writing directly there caused a `SameFileError`).

### 2.4 Import-time version validation

`cupy/backends/ascend/__init__.py` re-reads the metadata at import and compares
it with the CANN actually installed (`$ASCEND_HOME_PATH`):

* candidates probed, kept in sync with the build-time probe:
  `<cann>/version.cfg` (≤ 8.2), `<cann>/compiler/version.info` (8.5+),
  `<cann>/opp/version.info` (fallback);
* comparison is on the **release train** (`built // 10 == installed // 10`), so
  patch releases within a train are accepted;
* mismatch ⇒ `RuntimeWarning` (never an exception — a false positive must not
  make the package unimportable);
* `CUPY_ASCEND_SKIP_VERSION_CHECK=1` silences it.

Public API:

```python
import cupy
cupy.backends.ascend.check_cann_version()   # -> bool
cupy.backends.ascend.get_wheel_metadata()   # -> dict | None
```

### 2.5 The RPATH policy (the actual bug that was fixed)

A wheel must import on a machine that is **not** the build machine. Originally
the build embedded the *absolute* build-time CANN path
(`/home/qingfeng/Ascend/cann-8.5.1/lib64`) into every `.so` as `DT_RPATH`. Since
`DT_RPATH` **takes precedence over `LD_LIBRARY_PATH`**, a user could not point
the wheel at their own CANN install — the wheel only ever worked on the build
host.

Two orthogonal fixes:

**(a) Do not embed the SDK path.** `Backend.embed_sdk_in_rpath`
(`install/cupy_builder/backends/_base.py:110`):

| Backend | Value | Rationale |
|---|---|---|
| CUDA / ROCm | `True` | SDK lives at a stable system path (`/usr/local/cuda/lib64`) that also exists at runtime |
| **Ascend / CANN** | **`False`** (`backends/ascend.py:49`) | CANN is a multi-GB, *user-installed*, relocatable toolkit |

When `False`, the `-L` flags are still passed to the linker (so `-lascendcl` et
al. resolve at build time) but no `-rpath` entry is emitted for them
(`cupy_setup_build.py:489-508`).

**(b) Use `DT_RUNPATH`, not `DT_RPATH`.** The default is now the modern
`DT_RUNPATH` (the `--disable-new-dtags` flag is omitted), so `LD_LIBRARY_PATH`
and `source set_env.sh` take precedence as expected
(`cupy_setup_build.py:521-533`). The legacy behaviour is still reachable via
`CUPY_INSTALL_LEGACY_RPATH=1`.

The `$ORIGIN`-relative entry for bundled libs (`cupy/.data/lib`) is unaffected
and still emitted.

### 2.6 How to build the two wheels

```sh
# ---------------- CANN 8.5 ----------------
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest   # CANN 8.5.x
export CUPY_INSTALL_USE_ASCEND=1
python -m build --wheel
# -> numpy_ascend_cann85-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl

# ---------------- CANN 9.0 ----------------
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/9.0      # CANN 9.0.x
export CUPY_INSTALL_USE_ASCEND=1
python -m build --wheel
# -> numpy_ascend_cann90-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
```

The CANN release train appears **twice** in the file name, on purpose: in the
distribution name (`numpy_ascend_cann90`, what `pip` records and uninstalls) and
in the platform tag (`.cann9.0`, what makes two trains unable to install over
each other). The one artefact that is *not* per-train is the source tarball,
see §2.9.

In-tree development (editable, faster iteration):

```sh
export CUPY_INSTALL_USE_ASCEND=1
python setup.py develop --inplace
python -c "import cupy._core"
```

> Build **inside a `manylinux_2_17` container** to keep glibc/GCC consistent
> with CANN's C++ ABI (see §3.2). CANN's own glibc symbol requirement is only
> `GLIBC_2.14`, so glibc is *not* the binding constraint — `libstdc++` is.

### 2.7 Wheel checklist

| Check | Command |
|---|---|
| No SDK path leaked into any `.so` | `for f in $(find cupy cupyx -name '*.so'); do readelf -d "$f" \| grep -E 'RPATH\|RUNPATH' \| grep -i cann && echo "LEAK: $f"; done` |
| Uses `DT_RUNPATH` (not `DT_RPATH`) | `readelf -d cupy/_core/core.*.so \| grep -E 'RPATH\|RUNPATH'` → must say `RUNPATH` |
| Metadata bundled | `unzip -p dist/*.whl 'cupy/.data/_wheel.json'` |
| Platform tag correct | `ls dist/` → must end in `.cann8.5.whl` / `.cann9.0.whl` |
| Imports in a clean env | `python -c "import cupy; print(cupy.__version__)"` |
| Version guard works | install the `cann9.0` wheel against a CANN 8.5 install → expect a `RuntimeWarning` |

### 2.8 Wheel naming rule and support matrix

A wheel name encodes everything that must match on the target machine:

```
numpy_ascend_cann90-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
```

| Field | Example | Meaning |
|---|---|---|
| distribution | `numpy_ascend_cann90` | backend-dependent, see the table below; PEP 427-escaped `-` → `_`. The *import* package is `cupy` on every backend |
| version | `14.0.0a1` | `cupy.__version__` |
| python tag | `cp311` | CPython minor (3.11, not interchangeable) |
| ABI tag | `cp311` | same value as the python tag |
| platform | `manylinux_2_17_x86_64` | CPU arch + glibc floor (`{x86_64,aarch64}`) |
| SDK tag | `cann9.0` | CANN release train (8.5.x → `cann8.5`, 9.0.x → `cann9.0`) |

| Platform | glibc | Python | CANN | Availability |
|---|---|---|---|---|
| linux **x86_64** | ≥ 2.17 (`manylinux_2_17`) | 3.9–3.13 | 8.5.x / 9.0.x | prebuilt wheels, one per release |
| linux **aarch64** | ≥ 2.17 | 3.9–3.13 | 8.5.x / 9.0.x | **build from source** (§2.6) |
| OpenEuler / ModelArts (910B test box) | system glibc | 3.9 | 8.2 | build from source |

* Build floor: CANN **8.2** (`AscendBackend.minimum_version = 820`).
* One wheel per **CANN release train**, not per NumPy version (§1.4); the tag
  only has to *match the train*, patch releases within it are fine (§2.4).
* The distribution name is **backend-dependent** and, for Ascend, carries the
  **CANN release train** — the same scheme as upstream's
  `cupy-cuda11x` / `cupy-cuda12x`, so `pip list`, `pip freeze` and
  `pip uninstall` all say which train you have:

  | Build | Distribution name | Where it comes from |
  |---|---|---|
  | CUDA / HIP / CPU stub | `cupy` | `[project].name` in `pyproject.toml` |
  | Ascend, CANN 8.5.x | `numpy-ascend-cann85` | `setup.py::_BackendAwareDistribution` |
  | Ascend, CANN 9.0.x | `numpy-ascend-cann90` | 〃 |
  | Ascend, CANN version unknown | `numpy-ascend` | 〃 (no tag to derive the suffix from) |
  | Ascend, source tarball | `numpy-ascend` | 〃 (`sdist` is exempt — §2.9) |

  The suffix is derived from `Backend.get_wheel_platform_tag()` (§2.2) with the
  dot removed (`cann8.5` → `cann85`). Mechanism: PEP 621 forbids a dynamic
  `name`, and a static `[project].name` silently wins over `setup(name=...)`,
  so the rename is applied to `dist.metadata.name` right after
  `Distribution.parse_config_files()` — which is where setuptools applies
  `[project]` (doing it in `__init__` gets overwritten).
* The import package is **`cupy`** on every backend
  (`[tool.setuptools.packages.find]` includes `cupy*`/`cupyx*`/
  `cupy_backends*`), so `import cupy` is unaffected by the rename.
* Because every one of these distributions ships the same top-level `cupy`
  package, two of them must never be installed side by side — **including two
  different CANN trains**: `pip uninstall numpy-ascend-cann85` before installing
  `numpy-ascend-cann90`. `cupy._environment._detect_duplicate_installation()`
  knows all of the names above and warns as soon as it finds more than one.
* Release page (wheel downloads):
  <https://github.com/qingfengxia/numpy-ascend/releases>.

### 2.9 How to build the source distribution (sdist)

The wheels above are *per CPython* and *per CANN train*. The sdist is the
opposite: **one tarball for every CPython 3.9–3.13 and every CANN release
train**, because it contains no compiled artefact at all.

```sh
export CUPY_INSTALL_USE_ASCEND=1
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
python -m build --sdist            # add --no-isolation if deps are installed
# -> dist/numpy_ascend-14.0.0a1.tar.gz
```

| Artefact | File name | Encodes |
|---|---|---|
| wheel | `numpy_ascend_cann90-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl` | distribution + CPython + platform + CANN |
| sdist | `numpy_ascend-14.0.0a1.tar.gz` | distribution only |

Why the sdist drops **both** tags:

* **No `cp3XX` tag.** The tarball ships Cython sources and headers only
  (`MANIFEST.in`: `recursive-include cupy *.pyx *.pxd *.pxi`, `*.h *.hpp`), and
  it explicitly *excludes* the generated C++ (`recursive-exclude cupy *.cpp` —
  "fail-safe to avoid including Cythonized sources in sdist"). `Cython>=3,<3.2`
  is declared in `[build-system].requires`, so the consumer's `pip` re-runs
  Cython for *its* interpreter. The same tarball therefore serves 3.9 … 3.13.
* **No `cannX.Y` tag.** `CUPY_CANN_VERSION` and the `.cannX.Y` platform tag are
  produced when the *wheel* is built from the tarball, not when the tarball is
  packed. `_BackendAwareDistribution` deliberately returns early for the `sdist`
  command (keeping the bare `numpy-ascend`) because the tarball is
  CANN-agnostic. `CUPY_INSTALL_USE_ASCEND=1` must still be set, otherwise the
  sdist is named after the CUDA/HIP default (`cupy-14.0.0a1.tar.gz`).

> ⚠️ **Packing** the tarball still needs a toolchain on the build machine:
> unlike `dist_info` / `egg_info` (which `setup.py` skips extensions for), the
> `sdist` command *does* configure the extension modules, which requires the
> CANN SDK plus a host compiler. No NPU is needed — as always, the 910B is only
> required to *run* (§3).

Consuming the tarball — `pip` picks the CPython and the CANN of whatever machine
it runs on:

```sh
pip install numpy_ascend-14.0.0a1.tar.gz
# CANN 9.0.x host -> builds + installs numpy-ascend-cann90
# CANN 8.5.x host -> builds + installs numpy-ascend-cann85
```

One source artefact therefore covers the whole matrix:

| Built from | CPython | CANN | Result |
|---|---|---|---|
| `numpy_ascend-14.0.0a1.tar.gz` | 3.9–3.13 | 8.5.x | `numpy_ascend_cann85-14.0.0a1-cp3XX-…-manylinux_2_17_{x86_64,aarch64}.cann8.5.whl` |
| 〃 | 3.9–3.13 | 9.0.x | `numpy_ascend_cann90-14.0.0a1-cp3XX-…-manylinux_2_17_{x86_64,aarch64}.cann9.0.whl` |
| 〃 | 3.9 | 8.2 | in-place build only (`build_ext --inplace`, §2.6) |

The installed distribution is reported under the CANN-suffixed name, so
`pip freeze` / `pip uninstall` are unambiguous:

```sh
pip freeze | grep numpy-ascend        # numpy-ascend-cann90==14.0.0a1
pip uninstall numpy-ascend-cann90
```

Checklist:

| Check | Command |
|---|---|
| Name has no CPython/CANN tag | `ls dist/` → `numpy_ascend-<ver>.tar.gz` (no `cp3`, no `cann`) |
| No Cythonised C++ inside | `tar tzf dist/numpy_ascend-*.tar.gz '*.cpp'` → empty |
| Sources + headers present | `tar tzf dist/numpy_ascend-*.tar.gz '*.pyx'` → non-empty |
| Build system can fetch Cython | `grep Cython pyproject.toml` |

---

## 3. Hard runtime prerequisites

A wheel produced by §2.6 does **not** bundle the SDK. The target machine must
satisfy all of the following.

### 3.1 CANN installation

* CANN **≥ 8.2** (`AscendBackend.minimum_version = 820`), toolkit **and**
  operator kernel packages. The version must match the wheel's tag
  (`cann8.5` wheel ⇔ a CANN 8.5.x install).
* The wheel no longer depends on `ASCEND_HOME_PATH` being set at runtime for
  *loading*, because no path is baked in. It is still read for the version
  check, and for `LD_PRELOAD` resolution — so set it.

### 3.2 C++ / C runtime requirements

| Requirement | Detail |
|---|---|
| glibc | CANN needs only `GLIBC_2.14`; `manylinux_2_17` is more than sufficient. **Not the constraint.** |
| **libstdc++** | **The real constraint.** CANN mixes `__cxx11` and legacy `_ZNSs` C++ symbols and links `libstdc++.so.6` dynamically → the target machine's libstdc++ must be at least as new as the build machine's. |
| Python | 3.9–3.13, same minor as the wheel's `cp3XX` tag |
| NumPy | `>=1.24,<2.6` (runtime, Python-level only — see §1.4) |
| fastrlock | `>=0.5` (pulled in by pip) |

> **conda pitfall (observed on Ubuntu 24.04).** conda ships its own older
> `libstdc++.so.6` which shadows the system one:
> `ImportError: libstdc++.so.6: version 'GLIBCXX_3.4.32' not found`.
> Fix by pointing conda at the system library:
> ```sh
> ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 \
>        "$CONDA_PREFIX/lib/libstdc++.so.6"
> ```

### 3.3 `LD_LIBRARY_PATH`

Because the SDK path is deliberately **not** in the RPATH, the CANN libraries
must be found at runtime via the environment:

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # exports LD_LIBRARY_PATH
```

This works precisely *because* the wheel now uses `DT_RUNPATH` — with the old
`DT_RPATH` this would have had no effect. Verify:

```sh
ldd cupy/_core/core.*.so | grep -i -E 'ascendcl|aclnn|opapi'   # must resolve
```

### 3.4 `LD_PRELOAD` (CANN 8.5.1 defect workaround)

`libop_common.so` in CANN 8.5.1 has an **undefined symbol**

```
_ZN2ge19GetViewErrorCodeStrENS_13ViewErrorCodeE
```

that only `liboptiling.so` provides. This is a CANN packaging defect, **not** a
wheel defect — it cannot be fixed by re-packaging (bundling `liboptiling.so`
would violate CANN's layout contract and bloat the wheel).

**Current status:** `cupy/__init__.py` pre-loads `liboptiling.so` with
`ctypes.CDLL(..., RTLD_LAZY | RTLD_GLOBAL)` before importing the runtime (silently
skipped off Ascend), so the manual `LD_PRELOAD` is **no longer required**. Keep it
as a fallback if a newer CANN patch reintroduces the symptom:

```sh
export LD_PRELOAD=<cann>/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so
```

### 3.5 Minimal working runtime

```sh
# 1. install the matching wheel
pip install numpy_ascend_cann85-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl

# 2. bring CANN into the environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 3. verify
python -c "import cupy; print(cupy.__version__); \
           print(cupy.backends.ascend.check_cann_version())"
```

### 3.6 Troubleshooting an installed wheel

| Symptom | Cause | Fix |
|---|---|---|
| `libop_common.so: undefined symbol: _ZN2ge19GetViewErrorCodeStr...` | CANN 8.5.1 packaging defect | already worked around in `cupy/__init__.py`; if it reappears, use the `LD_PRELOAD` of §3.4 |
| `libstdc++.so.6: version 'GLIBCXX_3.4.32' not found` | older `libstdc++` on the target machine, typically conda | §3.2 (`ln -sf` the system `libstdc++.so.6` over `$CONDA_PREFIX/lib/`) |
| `RuntimeWarning: installed CANN version differs from ...` | wheel tag and installed CANN are different release trains | install the matching wheel; `CUPY_ASCEND_SKIP_VERSION_CHECK=1` if the train differs only in patch level |
| `cannot open shared object file: libcann_ops_fft.so` | `ops-fft` is an optional operator package, not part of the CANN SDK | install `ops-fft`, or set `ASCEND_OPS_FFT_PATH` + `LD_LIBRARY_PATH`; harmless when FFT is unused (`DeveloperNotes.md` §3.5b) |
| `NotImplementedError: Ascend backend: no implementation registered for 'ascend_xxx'` | that operator is not ported; dispatch is by name (`cupy_x` → `ascend_x`) | see the gap list in `tools/cst_db.md` / `Progress.md` §4 |
| `cupy.show_config()` → `AttributeError: '_UnavailableModule' object has no attribute 'get_build_version'` | `cupyx/_runtime.py` still collects CUDA-only build info (CUB, …) | known gap; use `get_wheel_metadata()` / `check_cann_version()` / `py_list_acl_ufuncs()` instead |
| `EL0003` or other device errors | no NPU on the machine (or the driver is missing) | compiling and importing do not need an NPU; real computation does |

---

## 4. Environment-variable reference (build time)

| Variable | Effect | Where |
|---|---|---|
| `CUPY_INSTALL_USE_ASCEND=1` | select the Ascend backend | `_context.py` |
| `ASCEND_HOME_PATH` | CANN root, used to detect version and libraries | `backends/ascend.py:38` |
| `CUPY_INSTALL_LEGACY_RPATH=1` | emit legacy `DT_RPATH` (ignores `LD_LIBRARY_PATH`) | `_context.py:86` |
| `CUPY_INSTALL_NO_RPATH=1` | disable all default RPATH entries | `_context.py:84` |
| `CUPY_INSTALL_WHEEL_METADATA=<path>` | supply `_wheel.json` explicitly | `_context.py:82` |

Runtime-only:

| Variable | Effect |
|---|---|
| `CUPY_ASCEND_SKIP_VERSION_CHECK=1` | silence the CANN-version mismatch warning |
| `LD_LIBRARY_PATH` | must include CANN `lib64` (via `set_env.sh`) |
| `LD_PRELOAD` | `liboptiling.so` workaround for CANN 8.5.1 — **fallback only**, `cupy/__init__.py` now pre-loads it (§3.4) |
| `ASCEND_OPS_FFT_PATH` | location of a source-built `libcann_ops_fft.so` (optional FFT) |
| `CUPY_ASCEND_SOC` | target SoC for the custom AscendC kernel JIT (default `Ascend910B4`) |
| `CUPY_ASCEND_DISABLE_CUSTOM_KERNELS=1` | skip the AscendC JIT/registration at import |
| `CUPY_ASCEND_LENIENT_ARGS=1` | accept-and-drop unsupported operator args (migration only) |

---

## 5. Summary

* **NumPy 2.x**: a Python-level runtime dependency only. No NumPy C ABI is
  used (`_dtype.pyx` wraps `numpy.dtype`/`numpy.can_cast`; DLPack is the
  zero-copy bridge). NumPy-2 differences are shimmed in
  `cupy/__init__.py:989-1049`. ⇒ **one wheel per CANN release, not per NumPy
  release.**
* **Two wheels**: the CANN version is encoded in the wheel platform tag
  (`cann8.5` / `cann9.0`, §2.2) and in `cupy/.data/_wheel.json` (§2.3), and is
  re-validated at import (§2.4).
* **RPATH**: the SDK path is never embedded for Ascend
  (`embed_sdk_in_rpath = False`) and `DT_RUNPATH` is used, so the user's
  `LD_LIBRARY_PATH` / `set_env.sh` wins (§2.5).
* **Which wheel / is there one**: naming rule and platform × CANN availability
  in §2.8; the summary check is `cupy.backends.ascend.get_wheel_metadata()`.
* **When it misbehaves**: §3.6 tabulates the common install/run failures; build
  failures (bisheng, Cython, missing headers) are covered by
  `DeveloperNotes.md` §3.
* **Runtime**: CANN must be installed and sourced; libstdc++ must be new
  enough; CANN 8.5.1 additionally needs `LD_PRELOAD=liboptiling.so` (§3).
