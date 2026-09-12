# CuPy Build System (`install/`)

This directory holds the XPU-neutral build system used to compile CuPy for
different accelerator backends (NVIDIA CUDA, AMD ROCm/HIP, Huawei Ascend/CANN).
It is intentionally structured so that **adding a backend does not require
editing shared code** — all backend-specific knowledge is confined to two
pluggable packages.

---

## 1. Architecture Overview

```
install/cupy_builder/
├── __init__.py            # bootstrap: Context / get_context() / registry
├── _context.py            # Context dataclass: CLI/env -> build configuration
├── _command.py            # setuptools build_ext command; Cythonizes modules
├── _compiler.py           # device compiler driver (thin; delegates to backend)
├── _features.py           # feature aggregation entry point (get_features)
├── _environment.py        # platform/environment helpers (Windows NVTX etc.)
├── _preflight.py          # pre-build sanity checks
├── cupy_setup_build.py    # orchestrates configuration + module list assembly
├── install_build.py       # low-level helpers: SDK path probes, version checks
├── install_utils.py       # generic utilities (warnings, path search)
│
├── features/              # ── "WHAT to build" ──────────────────────────
│   ├── __init__.py
│   ├── _base.py           #   Feature base class + from_dict()
│   ├── cuda.py            #   CUDA modules + optional library dicts
│   ├── rocm.py            #   ROCm/HIP modules + library dicts
│   └── ascend.py          #   Ascend/CANN modules + version-aware libraries
│
└── backends/              # ── "HOW to build" ───────────────────────────
    ├── __init__.py        #   registry: name -> Backend instance
    ├── _base.py           #   Backend abstract base class
    ├── cuda.py            #   CUDA: nvcc, gencode flags, CUDA_VERSION probe
    ├── rocm.py            #   ROCm: hipcc, include layout, HIP version probe
    └── ascend.py          #   CANN: bisheng, aclnn includes, CANN version probe
```

### The two-package split

| Package | Question answered | Contains |
|---------|-------------------|----------|
| `features/` | **What do I build?** | Cython module list, libraries to link, `Feature` objects, optional sub-libraries (CUB, NCCL, cuTENSOR, …) |
| `backends/` | **How do I build it?** | SDK path discovery, device compiler, include/library dirs, compile flags, preprocessor macros, compile-time constants, version detection |

Keeping these separate means the shared build logic
(`_command.py`, `cupy_setup_build.py`, `install_build.py`) only ever calls
*abstract* methods — it never contains `if backend == 'ascend'` branches.

### Data flow

```
$ CUPY_INSTALL_USE_ASCEND=1 python setup.py build_ext --inplace
        │
        ▼
_context.Context()                       # reads env -> use_ascend=True
        │
        ├─ cupy_builder.get_features(ctx)  # _features.py -> CUPY_ascend
        │        └─ features/ascend.py     # modules + version-aware libs
        │
        ├─ backends.get_backend(ctx)       # backends/__init__ -> AscendBackend
        │        └─ backends/ascend.py     # SDK path, includes, flags, macros
        │
        ▼
_command.py   → _compiler.py  → backend.get_device_compile_args()
cupy_setup_build.py           → backend.get_define_macros()
                              → backend.get_compile_time_env()
install_build.py              → backend.get_include_dirs()/get_library_dirs()
```

---

## 2. The `Backend` Interface

Defined in `backends/_base.py`. A backend subclass must implement the abstract
methods and may override any of the optional hooks.

### Identity attributes

| Attribute | Purpose | Example (`AscendBackend`) |
|-----------|---------|---------------------------|
| `name` | Canonical short name | `'ascend'` |
| `env_flag` | Env var that enables it | `'CUPY_INSTALL_USE_ASCEND'` |
| `version_macro` | Cython compile-time constant | `'CUPY_CANN_VERSION'` |
| `sdk_env_var` | SDK root env var (for summary) | `'ASCEND_HOME_PATH'` |
| `compiler_env_var` | Device compiler override env var | `''` |
| `compiler_name` | Human-readable compiler name | `'bisheng (ascendcc)'` |
| `needs_cub_headers` | Whether CUB/Thrust headers are needed | `False` |

### Abstract methods (required)

```python
def get_sdk_path(self) -> str | None: ...
def get_device_compiler(self) -> list[str] | None: ...
def get_device_compile_args(self, ctx, src) -> list[str]: ...
def check_version(self, compiler, settings) -> bool: ...
def get_version(self) -> int: ...
```

### Optional hooks (sensible defaults)

```python
def get_include_dirs(self, ctx) -> list[str]:          # default []
def get_library_dirs(self, ctx) -> list[str]:          # default []
def get_extra_compile_args(self) -> list[str]:         # default []
def get_extra_link_args(self) -> list[str]:            # default []
def get_define_macros(self) -> list[tuple[str, str]]:  # default []
def get_compile_time_env(self, ctx) -> dict[str, Any]: # default {}
def supports_platform(self, platform: str) -> bool:    # default: linux only

# wheel packaging (see section 4.4)
embed_sdk_in_rpath: bool = True                        # False for relocatable SDKs
def get_wheel_platform_tag(self) -> str | None:        # default None, e.g. 'cann8.5'
def get_wheel_metadata(self) -> dict[str, Any]:        # default {}
```

---

## 3. How To Add a New Backend

Suppose you are adding support for a hypothetical accelerator **"Foo"**.

### Step 1 — Create the `Backend` descriptor

Create `install/cupy_builder/backends/foo.py`:

```python
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import cupy_builder.install_build as build
from cupy_builder.backends._base import Backend

if TYPE_CHECKING:
    from cupy_builder._context import Context


class FooBackend(Backend):
    name = 'foo'
    env_flag = 'CUPY_INSTALL_USE_FOO'
    version_macro = 'CUPY_FOO_VERSION'
    sdk_env_var = 'FOO_HOME'
    compiler_env_var = 'FOOCC'
    compiler_name = 'foocc'
    needs_cub_headers = False

    def get_sdk_path(self):
        return os.environ.get('FOO_HOME') or None

    def get_device_compiler(self):
        sdk = self.get_sdk_path()
        if not sdk:
            return None
        cc = os.path.join(sdk, 'bin', 'foocc')
        return [cc] if os.path.exists(cc) else None

    def get_include_dirs(self, ctx):
        sdk = self.get_sdk_path()
        return [os.path.join(sdk, 'include')] if sdk else []

    def get_library_dirs(self, ctx):
        sdk = self.get_sdk_path()
        return [os.path.join(sdk, 'lib64')] if sdk else []

    def get_extra_compile_args(self):
        return ['-std=c++17']

    def get_define_macros(self):
        return [
            ('CUPY_USE_FOO', '1'),
            (self.version_macro, str(self.get_version())),
        ]

    def get_device_compile_args(self, ctx, src):
        compiler = self.get_device_compiler()
        if compiler is None:
            raise RuntimeError('foocc not found under FOO_HOME')
        base = build.get_compiler_base_options(compiler)
        return compiler + base + ['-O2', '-fPIC', '--std=c++17']

    def get_compile_time_env(self, ctx):
        return {
            'CUPY_CUDA_VERSION': 0,
            'CUPY_HIP_VERSION': 0,
            'CUPY_CANN_VERSION': 0,
            'CUPY_FOO_VERSION': self.get_version(),
        }

    def supports_platform(self, platform):
        return platform == 'linux'

    def check_version(self, compiler, settings):
        return build.check_foo_version(compiler, settings)

    def get_version(self):
        return build.get_foo_version()
```

### Step 2 — Register the backend

Edit `install/cupy_builder/backends/__init__.py`:

```python
from cupy_builder.backends.foo import FooBackend

BACKEND_CLASSES: dict[str, type[Backend]] = {
    CudaBackend.name: CudaBackend,
    RocmBackend.name: RocmBackend,
    AscendBackend.name: AscendBackend,
    FooBackend.name: FooBackend,          # <-- add this
}
```

and extend the resolver in `get_backend(ctx)`:

```python
if getattr(ctx, 'use_foo', False):
    return get_backend_by_name(FooBackend.name)
```

### Step 3 — Add the `Feature` (what to compile/link)

Create `install/cupy_builder/features/foo.py` with the module list and
libraries, following `features/ascend.py` as a template:

```python
foo_files = [ 'cupy.backends.foo.api....', ... ]

class CUPY_foo(Feature):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.name = 'foo'
        self.required = True
        self.modules = foo_files
        self.libraries = ['foort', ...]
        self.configure = build.check_foo_version
        build.check_foo_version(None, None)
        self._version = build.get_foo_version()
```

Then hook it into the `get_features()` dispatcher in `_features.py`
(the single aggregation entry point):

### Step 4 — Add the build-context flag

In `_context.py`, add the environment switch and to `get_backend_name()`:

```python
self.use_foo = _get_env_bool('CUPY_INSTALL_USE_FOO', _env)
...
def get_backend_name(self):
    if self.use_stub:   return "stub"
    if self.use_hip:    return "hip"
    if self.use_ascend: return "ascend"
    if self.use_foo:    return "foo"      # <-- add this
    return "cuda"
```

> **Note**: the *name* returned here is what gets passed to
> `backends.get_backend_by_name()`. If it differs from `Backend.name`
> (as `'hip'` does for ROCm), register an alias in `_BACKEND_ALIASES`.

### Step 5 — Version probe helpers

Add `check_foo_version()` / `get_foo_version()` in `install_build.py`
alongside the existing `check_cann_version()` etc. Follow the pattern:
cache in a module-global, parse a version file, return `bool` from `check`,
`int` from `get`.

### Step 6 — Verify

```sh
export CUPY_INSTALL_USE_FOO=1
python setup.py build_ext --inplace
python -c "import cupy; print(cupy.__version__)"
```

Expected output includes a configuration summary line:
```
  Backend            : foo
  SDK path           : /path/to/foo
  Device compiler    : ['/path/to/foo/bin/foocc']
```

---

## 4. Porting Checklist (Cython / XPU code)

The build system is only half the story. When porting backend-specific Cython
sources, follow the established conventions:

### 4.1 Source substitution

Backend-specific implementations of a module are listed in the `Feature`'s
`modules` as a tuple `(module_name, [source_path])`:

```python
('cupy._core._routines_linalg', ['cupy/_ascend/_core/_routines_linalg.pyx']),
```

The build replaces the default source with the given path. Edit the file under
the backend directory (e.g. `cupy/_ascend/_core/`), **not** `cupy/_core/`
(which may not even contain a `.pyx` anymore).

### 4.2 Compile-time constants

Use Cython's **compile-time** `IF` with the backend's version macro:

```cython
IF CUPY_CANN_VERSION > 0:
    # Ascend path
ELSE:
    # CUDA path
```

The macro value is injected by `backend.get_compile_time_env()`. **Always end
each `IF`/`ELSE` branch with an explicit `return` or `raise`** — otherwise the
CUDA branch can fall through into code that assumes a different memory layout
(this caused a real transpose bug in `matmul`).

### 4.3 One source of truth for versions

Do **not** hardcode a version anywhere. The same value must flow to:
- the C preprocessor macro (`backend.get_define_macros()`),
- the Cython compile-time env (`backend.get_compile_time_env()`).

Both read from the backend's cached version, so they cannot drift.

---

## 4.4 Building redistributable wheels (RPATH policy)

A wheel must import on a machine that is **not** the build machine. Two things
must hold:

1. no absolute SDK path is baked into any `.so`, and
2. the runtime library search path can still be overridden by the user.

### The `embed_sdk_in_rpath` flag

`Backend.embed_sdk_in_rpath` controls whether `get_library_dirs()` results are
copied into the extension modules' runtime search path:

| Backend | Value | Rationale |
|---------|-------|-----------|
| CUDA / ROCm | `True` | SDK normally lives at a stable system location (`/usr/local/cuda/lib64`) that also exists at runtime. |
| **Ascend / CANN** | **`False`** | CANN is a multi-GB, *user-installed* toolkit whose absolute path differs per machine. Embedding it produces a wheel that only imports where it was built. |

When `False`, the `-L` flags are still passed to the linker (so it can resolve
`-lascendcl` etc. at build time) but no `-rpath` entry is emitted for them.

### DT_RUNPATH vs DT_RPATH

`CUPY_INSTALL_LEGACY_RPATH=1` restores the legacy `--disable-new-dtags` flag,
which produces `DT_RPATH`. **`DT_RPATH` takes precedence over
`LD_LIBRARY_PATH`**, so a user can never point the wheel at a relocated CANN.
The default is now `DT_RUNPATH` (i.e. the flag is omitted), so
`LD_LIBRARY_PATH` and `source set_env.sh` work as expected.

### Verifying a build

```sh
# No SDK path may appear anywhere:
for f in $(find cupy cupyx -name "*.so"); do
  readelf -d "$f" | grep -E 'RPATH|RUNPATH' | grep -i cann && echo "LEAK: $f"
done
```

### Wheel naming and version safety

`Backend.get_wheel_platform_tag()` appends an SDK identifier to the wheel's
platform tag, and `Backend.get_wheel_metadata()` records the exact SDK version
into `cupy/.data/_wheel.json`. Both are consumed by `setup.py`
(`bdist_wheel` cmdclass) and `cupy/backends/ascend/__init__.py` respectively.

Producing the two Ascend wheels:

```sh
# ---- CANN 8.5 ----
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest   # CANN 8.5.x
export CUPY_INSTALL_USE_ASCEND=1
python -m build --wheel
# -> cupy-<ver>-cp311-cp311-manylinux_2_17_x86_64.cann8.5.whl

# ---- CANN 9.0 ----
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/9.0      # CANN 9.0.x
export CUPY_INSTALL_USE_ASCEND=1
python -m build --wheel
# -> cupy-<ver>-cp311-cp311-manylinux_2_17_x86_64.cann9.0.whl
```

Both wheels install into the same environment without clashing (different
platform tags), and each refuses to silently run against the wrong CANN:

```python
import cupy                      # warns if CANN release train differs
cupy.backends.ascend.check_cann_version()          # -> False on mismatch
cupy.backends.ascend.get_wheel_metadata()          # build-time record
```

Set `CUPY_ASCEND_SKIP_VERSION_CHECK=1` to silence the warning (e.g. in CI where
a patch-level difference is intentional).

### Runtime requirements of the wheel

The wheel still requires an external CANN installation. Users must:

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # exports LD_LIBRARY_PATH
export LD_PRELOAD=<cann>/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so
```

The `LD_PRELOAD` works around a defect in CANN 8.5.1: `libop_common.so` has an
undefined symbol `_ZN2ge19GetViewErrorCodeStrENS_13ViewErrorCodeE` that only
`liboptiling.so` provides. This is a CANN problem, not a wheel problem, so it
cannot be fixed by packaging — document it as a prerequisite.

### Why not a single universal manylinux wheel

* aclnn operator signatures change between CANN releases (hence the
  version-specific tags);
* `libop_common.so` is coupled to a matching `liboptiling.so` per release;
* CANN mixes `__cxx11` and legacy `_ZNSs` C++ symbols and links `libstdc++.so.6`
  dynamically, so the target machine's libstdc++ must be new enough. Build
  inside a `manylinux_2_17` container (CANN's own glibc symbol requirement is
  only `GLIBC_2.14`, so glibc itself is not the constraint).

Recommended: build for a **CANN version + platform** pair, not for "any Linux".

---

## 5. Reference: Existing Backends

| Backend | SDK env | Compiler | Version source | Library selection |
|---------|---------|----------|----------------|-------------------|
| `CudaBackend` | `CUDA_PATH` | `nvcc` | `CUDA_VERSION` probe (feature `configure`) | fixed list |
| `RocmBackend` | `ROCM_HOME` | `hipcc` | `check_hip_version()` | canonicalized per HIP version |
| `AscendBackend` | `ASCEND_HOME_PATH` | `bisheng` | `check_cann_version()` (multi-path probe) | **version-aware** (8.2 vs 8.5+), filtered by on-disk existence |

### Ascend library selection (`features/ascend.py`)

CANN renames its shared libraries between releases, so the library set is
chosen by version and then filtered against the filesystem:

```python
_CANN_LIBS_82 = [... 'aclnn_ops_train', 'aclnn_ops_infer', 'aclnn_math' ...]
_CANN_LIBS_85 = [... 'opapi_nn', 'opapi', 'op_common', 'ge_compiler' ...]

libs = select_cann_libraries(version)   # 8.2 / 8.5+ sets
libs = _filter_existing_libs(libs)      # drop anything not on disk
```

This lets newer CANN releases (e.g. 9.0) work without code changes as long as
they reuse the known library names.

---

## 6. Troubleshooting

### `Compile-time name 'X' not defined`
Calling `cythonize()` manually bypasses `_command.py`, which injects
`compile_time_env`. Use `python setup.py build_ext` instead, or pass
`compile_time_env=backend.get_compile_time_env(ctx)` yourself.

### Headers changed but nothing recompiles
Editing a `.h` file does not trigger Cython recompilation. Run
`bash clean_cpp_so_files.sh` to force a full rebuild.

### Stale `.cpp` / `.so`
The generated `.cpp` and `.so` live next to the `.pyx`. Delete them (or use the
clean script) after changing `.pxd`/`.h` files.

### `Unknown backend: '...'`
`ctx.get_backend_name()` returned a name not present in `BACKEND_CLASSES`.
Check for a spelling mismatch and add an entry to `_BACKEND_ALIASES` if the
historical name differs from `Backend.name`.
