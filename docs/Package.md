# Packaging numpy-ascend (`cupy`) for Ascend / CANN

This document covers three things that are routinely confused with each other:

1. **How the package cooperates with NumPy 2.x** and what "version compatible"
   means for it (spoiler: NumPy is a *runtime Python dependency only* — there is
   **no** NumPy C-ABI coupling, so the wheel is *not* NumPy-version-locked).
2. **The wheel packaging strategy** used to ship two separate wheels for
   CANN 8.5 and CANN 9.0 from one source tree.
3. **The hard runtime prerequisites** — C/C++ toolchain, C runtime (glibc),
   CANN installation, and `LD_LIBRARY_PATH` / `LD_PRELOAD` — together with the
   verification commands for each.

Related documents:
* `install/README.md` §4.4 — the RPATH policy in build-system terms.
* `README.md` §2 — CANN installation on a machine without an NPU.
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

`setup.py:75-98` installs a `bdist_wheel` cmdclass that appends this tag to the
platform tag, producing:

```
cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl
cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
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
# -> cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl

# ---------------- CANN 9.0 ----------------
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/9.0      # CANN 9.0.x
export CUPY_INSTALL_USE_ASCEND=1
python -m build --wheel
# -> cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
```

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

that only `liboptiling.so` provides. Until CANN fixes this, users must
preload it:

```sh
export LD_PRELOAD=<cann>/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so
```

This is a CANN packaging defect, **not** a wheel defect — it cannot be fixed by
re-packaging (bundling `liboptiling.so` would violate CANN's layout contract
and bloat the wheel).

### 3.5 Minimal working runtime

```sh
# 1. install the matching wheel
pip install cupy-14.0.0a1-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl

# 2. bring CANN into the environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_PRELOAD=$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so

# 3. verify
python -c "import cupy; print(cupy.__version__); \
           print(cupy.backends.ascend.check_cann_version())"
```

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
| `LD_PRELOAD` | `liboptiling.so` workaround for CANN 8.5.1 |

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
* **Runtime**: CANN must be installed and sourced; libstdc++ must be new
  enough; CANN 8.5.1 additionally needs `LD_PRELOAD=liboptiling.so` (§3).
