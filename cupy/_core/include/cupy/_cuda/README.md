These headers are **not vendored** in this repository.

`cuda_fp16.h` / `cuda_fp16.hpp` are NVIDIA proprietary files from the CUDA
Toolkit; every copy previously kept here (`cuda-11/`, `cuda-12/`) has been
removed from version control. See [docs/ascend/Compliance.md](../../../../../../docs/ascend/Compliance.md).

They are only needed at run time by the **CUDA** backend, for NVRTC JIT on
CUDA 11.x / 12.0–12.1 (CUDA Enhanced Compatibility: the headers of the running
toolkit must match what NVRTC was built against). CUDA 12.2+ must not use a
bundled copy — `cuda_fp16.h` there depends on headers that cannot be bundled
(e.g. `vector_types.h`).

How to obtain them locally (never committed, never packaged)
------------------------------------------------------------

```sh
python tools/fetch_cuda_fp16_headers.py            # auto-detect toolkit/conda/pip wheel
python tools/fetch_cuda_fp16_headers.py --list     # show what was found
python tools/fetch_cuda_fp16_headers.py --include-dir /usr/local/cuda/include
```

The script only copies files that already exist on this machine (no network
access). The resulting `cuda-11/` and `cuda-12/` are gitignored and are
filtered out of every wheel and sdist by
`cupy_setup_build.strip_cuda_headers()` + `MANIFEST.in`.

Runtime lookup order (`cupy/_core/_gpu/_compile_with_cache.pyx`):

1. this directory (a local copy from the script),
2. `$CUDA_PATH/include` (or the auto-detected toolkit),
3. the `include` dir shipped by conda / `nvidia-cuda-runtime-*` wheels.

If none is available, compilation raises a `RuntimeError` pointing at the
script. Ascend/CANN builds never touch this code path.
