# `cupy_backends/` — CUDA build compatibility shim

This directory exists **only** so the Ascend fork's sources keep compiling
against the upstream layout they were written for. It is a *build-time* shim:
it holds no implementation, no `.pyx` and no `.so`, and nothing in it is
imported at runtime (legacy runtime module paths are served by `sys.modules`
aliases registered in `cupy/__init__.py` and `cupy/cuda/__init__.py`).

Unless noted otherwise, "upstream" means CuPy at
`c18c31443a7f4ca8f2c8dbe7ac183b746a67ffec`.

## Background

Upstream CuPy keeps the backend-abstract layer in a top-level package
`cupy_backends/`. The Ascend fork moved it into the main package and also
renamed the CUDA device API to a backend-neutral `cupy.xpu`:

| upstream (`cupy_backends/…`) | this fork (`cupy/…`) |
|---|---|
| `cuda/api/runtime.{pyx,pxd}` | `backends/backend/api/runtime.{pyx,pxd}` |
| `cuda/api/driver.{pyx,pxd}` | `backends/backend/api/driver.{pyx,pxd}` |
| `cuda/api/_runtime_enum.*`, `_driver_enum.*` | `backends/cuda/api/_runtime_enum.*` (kept there) |
| `cuda/libs/<lib>` | `backends/cuda/libs/<lib>` |
| `cuda/stream.{pyx,pxd}` | `backends/backend/stream.{pyx,pxd}` |
| `cuda/_softlink.*` | `backends/backend/_softlink.*` |
| `cupy_*.h` | `backends/cupy_*.h` |
| device API `cupy/cuda/{device,memory,stream,pinned_memory,…}` | `cupy/xpu/{…}` |

The fork's sources still spell the upstream names (`cupy_backends.*` cimports and
header includes, `cupy.cuda.<device module>` cimports). Satisfying those
references from here is deliberately cheaper than rewriting ~30 files, because
**files added under this directory cannot conflict when merging upstream —
whereas every line changed in a file upstream also maintains can.**

## What is here: 8 symlinks + 5 tiny files

```
cupy_backends/
├── __init__.pxd                       # empty: makes this a cimport-able package
├── cupy_backend_common.h -> ../cupy/backends/cupy_backend_common.h
├── cupy_backend_runtime.h -> ../cupy/backends/cupy_backend_runtime.h
├── cupy_complex.h       -> ../cupy/backends/cupy_complex.h
├── cupy_lapack.h        -> ../cupy/backends/cupy_lapack.h
└── cuda/
    ├── __init__.pxd                    # from cupy.backends.backend cimport stream
    ├── _softlink.pxd                   # from cupy.backends.backend._softlink cimport *
    ├── cupy_cuda.h         -> ../../cupy/backends/cuda/cupy_cuda.h
    ├── cupy_cuda_runtime.h -> ../../cupy/backends/cuda/cupy_cuda_runtime.h
    ├── cupy_cusolver.h     -> ../../cupy/backends/cuda/cupy_cusolver.h
    ├── libs -> ../../cupy/backends/cuda/libs
    └── api/
        ├── __init__.pxd                # from cupy.backends.backend.api cimport driver, runtime
        └── runtime.pxd                 # from cupy.backends.backend.api.runtime cimport *
```

## Why the headers are symlinks (and why 3 extra live under `cuda/`)

`cdef extern from '<path>'` is emitted as a plain `#include` in the generated
C++, resolved by the compiler **relative to that `.cpp`**, which sits next to the
`.pyx`. The fork writes e.g.

```cython
# cupy/_core/_gpu/_routines_linalg.pyx
cdef extern from '../../cupy_backends/cupy_complex.h':
```

so `<repo-root>/cupy_backends/cupy_complex.h` must exist. A symlink is enough:
headers carry no Cython-level symbol names, so nothing can be mangled.

The three `cuda/cupy_*.h` symlinks are needed because the referenced top-level
headers contain `#include "cuda/cupy_cuda.h"` (also `cuda_runtime.h`,
`cuda_cusolver.h`), and **quoted nested includes are resolved relative to the
directory of the including file as written** (`cupy_backends/`) — not relative to
the symlink target's real directory. The layout therefore has to be reproduced at
this level too.

Only headers actually reached through this path are symlinked. Headers reached
through the real tree (`cupy_blas.h`, `cupy_rand.h`, `cupy_cusparselt.h`, …) and
the `hip/`, `stub/`, `ascend/` include branches (non-CUDA builds) need nothing
here: 16 surplus symlinks were removed after verifying zero references
repo-wide plus a forced rebuild.

## Why `cuda/libs` is a directory symlink

Everything cimported through that path is declared as `cdef extern from *:`
(inline C) — see `cupy/backends/cuda/libs/{nvrtc,cutensor,cusolver}.pxd`. For
such declarations the C symbol names come from the inline C code, **not** from
Cython's module-name mangling, so reading the very same `.pxd` under the old
module name (`cupy_backends.cuda.libs.cusolver`) yields a *compatible*
declaration. One directory symlink then satisfies both import forms:

* package form — `from cupy_backends.cuda.libs cimport nvrtc`
  (Cython resolves `libs/nvrtc.pxd` through the link)
* module form — `from cupy_backends.cuda.libs.cusolver cimport (…)`

The only mangling-sensitive file in that tree, `cusparse.pxd`
(`cdef class SpVecAttributes` …), is never cimported through the old path — it is
only *imported* at runtime through the `sys.modules` alias — so it is harmless.
**If `cupy_backends.cuda.libs.cusparse` ever gets cimported, replace this symlink
with explicit `cimport` re-export files** (see the next section).

## Why the `.pxd` files are one-line `cimport` re-exports, not symlinks

Cython mangles C names **per module name** (`__pyx_obj_<module>_<class>`,
`__pyx_opt_args_<module>_<func>…`). A symlinked `.pxd` is read under the *old*
module name, so a consumer would reference
`__pyx_opt_args_4cupy_4cuda_6stream_*` while the real modules
(`cupy.backends.backend.stream` / `cupy.xpu.stream`) export
`__pyx_opt_args_4cupy_3xpu_6stream_*`. This is not a link error one can ignore —
it fails when the module is initialised:

```
TypeError: C function cupy.xpu.stream.get_current_stream has wrong signature
(expected ...__pyx_opt_args_4cupy_4cuda_6stream_*, got ...__pyx_opt_args_4cupy_3xpu_6stream_*)
```

A one-line re-export avoids the mismatch because `cimport` binds **aliases to the
original declarations** (same C names, same classes, same module identity):

```cython
# cupy_backends/cuda/__init__.pxd
from cupy.backends.backend cimport stream
# cupy_backends/cuda/api/__init__.pxd
from cupy.backends.backend.api cimport driver, runtime
# cupy_backends/cuda/api/runtime.pxd
from cupy.backends.backend.api.runtime cimport *
# cupy_backends/cuda/_softlink.pxd
from cupy.backends.backend._softlink cimport *
```

Rule of thumb for adding anything here:

| the referenced `.pxd` contains | mechanism to use |
|---|---|
| only `cdef extern from …` / inline `cdef extern from *:` | symlink (file or directory) |
| `cdef class` or `cdef`/`cpdef` function definitions | `cimport` re-export — never a symlink |
| module form `from X.Y cimport name` | a real `Y.pxd` is required (a package `__init__.pxd` re-export is not enough) |
| package form `from X cimport name` | one line in the package's `__init__.pxd` is enough |

`cupy/cuda/__init__.pxd` uses the same trick for
`from cupy.cuda cimport device|memory|pinned_memory|stream`.

## The runtime side (deliberately not in this directory)

Python-level imports such as `from cupy_backends.cuda.libs import cublas` or
`import cupy.cuda.device` are served by `sys.modules` aliases registered in
`cupy/__init__.py` (for `cupy_backends.*`) and `cupy/cuda/__init__.py`
(for `cupy.cuda.*` -> `cupy.xpu.*`). They map a legacy path to the **same module
object**, so aliasing/`isinstance`/identity checks keep working and no duplicate
extension module is created.

## Fixes that could not be shimmed here (source edits, already applied)

* `cupy/_core/_gpu/_routines_linalg.pyx` — relative header path is one `../` short
  (`../../cupy/backends/…` resolves to the non-existent `cupy/cupy/backends/…`).
* `cupy/cuda/texture.pyx`, `cupyx/cusolver.pyx` — same class of wrong relative
  header path; they now point at `cupy/backends/…` directly, which is also why
  fewer header symlinks are needed here.
* `cupy/fft/_callback.pyx`, `cupyx/cutensor.pyx` — module-form cimports
  (`from cupy.cuda.device cimport …`, `from cupy.cuda.pinned_memory cimport …`)
  now import from `cupy.xpu.*`; re-creating `cupy/cuda/<module>.pxd` was avoided
  because those exact paths exist upstream and would collide on merge.
* `cupy/cuda/__init__.py` — the `sys.modules` alias block must run *before* the
  first `cupy.cuda.<submodule>` import, because `cupy.cuda.texture` imports
  `cupy.cuda.stream` while it is being initialised.
* `cupy/xpu/device.pyx`, `cupy/_core/_routines_{creation,math,statistics}.pyx` —
  7 references to the non-existent `cupy.backends.backend.libs` corrected to
  `cupy.backends.cuda.libs`.
* `cupy/__init__.py` — `cupy.fuse = _core.fusion.fuse` re-enabled (fixes
  `import cupyx.scipy.linalg`).

## Removal plan / verification

When upstream lands the same refactor, or the fork's sources are updated to the
new paths, delete this directory together with the source edits listed above.

```sh
python setup.py build_ext --inplace           # expect rc=0 and 64 extensions
python -c "import cupy, cupyx.scipy.linalg"   # plus alias-identity checks
```

The wheel does **not** need this directory: it ships compiled `.so` files and
resolves legacy module paths through the `sys.modules` aliases, which is why only
a stray entry or two from here ends up inside the wheel.
