# Shim around `_cusolver.pyx`, the whole purpose of this is to ensure
# we load `from cupy_backends.cuda.libs import sucolver` as a `from`
# import, which ensure cuda-pathfinder use.
# This should become unnecessary with `CUPY_USE_CUDA_PYTHON`.
from __future__ import annotations

from cupy_backends.cuda.libs import cusolver as _libs_cusolver  # noqa: F401
from cupyx._cusolver import (  # noqa: F401
    _available_compute_capability,
    _available_cuda_version,
    _available_hip_version,
    _geqrf_orgqr_batched,
    _gesvd_batched,
    _gesvdj_batched,
    _syevj_batched,
    check_availability,
    csrlsvqr,
    gels,
    gesv,
    gesvda,
    gesvdj,
    syevj,
)
