#!/usr/bin/env python3
"""Validate and benchmark ``cupyx`` sparse (cuSPARSE / ops-sparse) on Ascend.

What it answers on a 910B box
-----------------------------
1. **build**: is the optional ``ops-sparse`` binding compiled in and loadable?
   (``libops_sparse.so`` must be reachable by the loader, e.g. via
   ``LD_LIBRARY_PATH`` or an install into the CANN tree.)
2. **feature**: which cuSPARSE entry points does *this* ops-sparse build
   actually provide?  ops-sparse implements a different operator set per SoC
   (``arch22`` = 910B, ``arch35`` = 950) and can be built with a subset of
   operators (``build.sh --ops=...``), so this list is not a constant -- see
   ``docs/ascend/ascend_sparse.md``.
3. **numbers**: does each available op agree with the SciPy/NumPy reference?
4. **speed**: median/min latency, and GFLOP/s where the op count is known.

Usage
-----
    python tools/benchmark_sparse.py                  # full run
    python tools/benchmark_sparse.py --list           # env + capabilities only
    python tools/benchmark_sparse.py --n 4096 --nnz-per-row 32
    python tools/benchmark_sparse.py --dtype float64  # if ops-sparse has FP64
    python tools/benchmark_sparse.py --no-bench       # correctness only
    python tools/benchmark_sparse.py --only spmv,spmm --json out.json
    python tools/benchmark_sparse.py --device 1

Notes
-----
* Expected ``SKIP`` list on a 910B (``arch22``) ops-sparse build:
  ``csr2cscEx2``, ``csrgeam2``, ``csrsort``, ``coosort``, ``coo2csr``,
  ``csr2coo``, ``sparseToDense`` -- those operator families are only
  implemented for ``arch35`` (950) there, so the high level cases that need
  them (``tocsc``, ``tocsr``, ``toarray``) skip as well.  ``spmv``/``spmm``/
  ``spgemm``/``spsm``/``cscsort``/``denseToSparse`` must pass.
* ``--list`` never touches the device, so it also works on a machine without
  an NPU (it is the quickest way to check a build).
* Cases whose backing op is missing from ``libops_sparse.so`` are reported as
  ``SKIP`` (with the missing entry point), *not* as failures.
* Exit status: ``0`` all executed cases passed, ``1`` a case failed,
  ``2`` no usable device / backend.

Validation is L4 only when the numbers are actually checked against SciPy on a
910B; ``--list`` is L3 (import / symbol / capability level).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy

import cupy


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------
#: Every entry point ``cupyx.cusparse`` may ask ops-sparse for.
KNOWN_OPS = (
    'spmv', 'spmm', 'spgemm', 'spsm', 'csrmvEx',
    'denseToSparse', 'sparseToDense',
    'csrsort', 'cscsort', 'coosort',
    'coo2csr', 'csr2coo', 'csr2cscEx2', 'csc2csrEx2',
    'csrgeam2', 'csr2csr_compress',
)


def _availability(name: str) -> bool | None:
    """``cupyx.cusparse.check_availability`` without importing errors."""
    try:
        from cupyx import cusparse
        return bool(cusparse.check_availability(name))
    except Exception:
        return None


def _cusparse_module():
    """The module serving the cuSPARSE API for the active backend."""
    from cupyx import _cusparse_backend
    return _cusparse_backend.get_cusparse()


def env_report(device_id: int = 0) -> dict[str, Any]:
    """Collect build/feature information (never allocates device memory)."""
    info: dict[str, Any] = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'cupy': cupy.__version__,
        'numpy': numpy.__version__,
    }

    from cupy.backends import backend
    if backend.is_ascend:
        info['backend'] = 'ascend'
    elif backend.is_hip:
        info['backend'] = 'hip'
    else:
        info['backend'] = 'cuda'

    try:
        import scipy
        info['scipy'] = scipy.__version__
    except Exception:
        info['scipy'] = None

    if info['backend'] == 'ascend':
        from cupy.backends import ascend
        metadata = ascend.get_wheel_metadata() or {}
        info['cann_build_version'] = metadata.get('cann_version')
        try:
            installed = ascend._detect_installed_version()
        except Exception:
            installed = None
        info['cann_installed_version'] = installed
        info['ld_library_path_has_cann'] = _loader_hint()

    # The sparse binding itself.
    try:
        module = _cusparse_module()
        info['cusparse_module'] = module.__name__
        missing = getattr(module, '_missing_optional', None)
        info['ops_sparse_missing'] = sorted(missing) if missing else []
        info['ops_sparse_library'] = _sparse_library_path()
    except Exception as exc:  # ops-sparse absent / not loadable
        info['cusparse_module'] = None
        info['cusparse_unavailable_reason'] = f'{type(exc).__name__}: {exc}'
        info['ops_sparse_missing'] = []
        info['ops_sparse_library'] = None

    info['availability'] = {name: _availability(name) for name in KNOWN_OPS}
    info['device_id'] = device_id
    return info


def _sparse_library_path() -> str | None:
    """Best-effort `ldd`-free lookup of the loaded libops_sparse.so."""
    try:
        with open('/proc/self/maps') as f:
            for line in f:
                if ('libops_sparse.so' in line
                        or 'libcann_ops_sparse.so' in line):
                    return line.split()[-1]
    except OSError:
        pass
    return None


def _loader_hint() -> bool:
    import os
    path = os.environ.get('LD_LIBRARY_PATH', '')
    return any('cann' in p for p in path.split(':'))


def _device_name(device_id: int) -> str:
    try:
        from cupy.cuda import runtime
        props = runtime.getDeviceProperties(device_id)
        for key in ('name', 'Name'):
            if isinstance(props, dict) and key in props:
                return str(props[key])
    except Exception:
        pass
    return '<unknown>'


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
def _sync() -> None:
    try:
        from cupy.cuda import runtime
        runtime.deviceSynchronize()
        return
    except Exception:
        pass
    try:
        cupy.cuda.Stream.null.synchronize()
    except Exception:
        pass


def _timeit(fn: Callable[[], Any], warmup: int,
            repeat: int) -> tuple[float, float]:
    """Return ``(median_ms, min_ms)`` including a sync per iteration."""
    for _ in range(max(1, warmup)):
        fn()
    _sync()
    samples = []
    for _ in range(max(1, repeat)):
        _sync()
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), min(samples)


# ---------------------------------------------------------------------------
# problem data
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    """Shape / timing configuration shared by all cases."""
    n: int = 20000
    nnz_per_row: int = 20
    dtype: str = 'float32'
    seed: int = 0
    repeat: int = 20
    warmup: int = 5
    bench: bool = True
    check: bool = True

    @property
    def n_small(self) -> int:
        """Shape for cases whose output is dense (kept small on purpose)."""
        return min(self.n, 2048)

    @property
    def np_dtype(self) -> numpy.dtype:
        return numpy.dtype(self.dtype)

    @property
    def n_dense_cols(self) -> int:
        return 64


def _csr_numpy(ctx: Ctx, n: int) -> tuple[numpy.ndarray, numpy.ndarray,
                                          numpy.ndarray]:
    """Random CSR arrays ``(data, indices, indptr)``, unique indices/row."""
    rng = numpy.random.default_rng(ctx.seed)
    k = max(1, min(ctx.nnz_per_row, n))
    indptr = numpy.zeros(n + 1, dtype=numpy.int32)
    rows = []
    for i in range(n):
        cols = numpy.sort(rng.choice(n, size=k, replace=False))
        rows.append(cols)
        indptr[i + 1] = indptr[i] + k
    flat = (numpy.concatenate(rows) if rows
            else numpy.empty(0, numpy.int64))
    indices = numpy.ascontiguousarray(flat, dtype=numpy.int32)
    data = rng.random(indices.size).astype(ctx.dtype)
    return data, indices, indptr


def _scipy_csr(ctx: Ctx, n: int):
    """CPU reference matrix (``None`` when SciPy is unavailable)."""
    try:
        import scipy.sparse
    except Exception:
        return None
    data, indices, indptr = _csr_numpy(ctx, n)
    return scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(n, n), dtype=ctx.dtype)


def _cupy_csr(ctx: Ctx, n: int):
    """Device matrix mirroring :func:`_scipy_csr` (same arrays)."""
    import cupyx.scipy.sparse as sparse
    data, indices, indptr = _csr_numpy(ctx, n)
    return sparse.csr_matrix(
        (cupy.asarray(data), cupy.asarray(indices), cupy.asarray(indptr)),
        shape=(n, n))


def _tol(dtype: str) -> float:
    if dtype == 'float64':
        return 1e-10
    if dtype == 'float32':
        return 2e-4
    return 1e-2


def _rel_err(got: Any, expected: numpy.ndarray) -> float:
    """Relative L2 error between a device result and a host reference."""
    got_np = cupy.asnumpy(got) if isinstance(got, cupy.ndarray) else got
    got_np = numpy.asarray(got_np, dtype=numpy.float64)
    expected = numpy.asarray(expected, dtype=numpy.float64)
    denom = numpy.linalg.norm(expected)
    if denom == 0:
        return float(numpy.linalg.norm(got_np))
    return float(numpy.linalg.norm(got_np - expected) / denom)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    status: str                      # OK / SKIP / FAIL
    reason: str = ''
    shape: str = ''
    nnz: int | None = None
    median_ms: float | None = None
    min_ms: float | None = None
    gflops: float | None = None
    rel_err: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Case:
    name: str
    requires: tuple[str, ...]
    fn: Callable[[Ctx], Result]


CASES: list[Case] = []


def _case(name: str, requires: tuple[str, ...] | None = None):
    def decorator(fn):
        CASES.append(Case(name=name,
                          requires=requires or (name,),
                          fn=fn))
        return fn
    return decorator


def _flops_per_sec(count: float, median_ms: float | None) -> float | None:
    if not median_ms:
        return None
    return count / (median_ms * 1e-3) / 1e9


@_case('spmv')
def _spmv(ctx: Ctx) -> Result:
    from cupyx import cusparse
    ref = _scipy_csr(ctx, ctx.n)
    A = _cupy_csr(ctx, ctx.n)
    rng = numpy.random.default_rng(ctx.seed + 1)
    x_np = rng.random(ctx.n).astype(ctx.dtype)
    x = cupy.asarray(x_np)

    out = cusparse.spmv(A, x)
    err = _rel_err(out, ref @ x_np) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.spmv(A, x),
                                  ctx.warmup, ctx.repeat)
    return Result('spmv', 'OK', shape=f'{ctx.n}x{ctx.n}', nnz=A.nnz,
                  median_ms=median, min_ms=minimum,
                  gflops=_flops_per_sec(2 * A.nnz, median), rel_err=err)


@_case('spmm')
def _spmm(ctx: Ctx) -> Result:
    from cupyx import cusparse
    ref = _scipy_csr(ctx, ctx.n)
    A = _cupy_csr(ctx, ctx.n)
    cols = ctx.n_dense_cols
    rng = numpy.random.default_rng(ctx.seed + 2)
    b_np = rng.random((ctx.n, cols)).astype(ctx.dtype)
    b = cupy.asarray(b_np)

    c = cusparse.spmm(A, b)
    err = _rel_err(c, ref @ b_np) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.spmm(A, b),
                                  ctx.warmup, ctx.repeat)
    return Result('spmm', 'OK', shape=f'{ctx.n}x{ctx.n} @ {ctx.n}x{cols}',
                  nnz=A.nnz, median_ms=median, min_ms=minimum,
                  gflops=_flops_per_sec(2 * A.nnz * cols, median),
                  rel_err=err)


@_case('spgemm')
def _spgemm(ctx: Ctx) -> Result:
    from cupyx import cusparse
    n = ctx.n_small
    ref = _scipy_csr(ctx, n)
    A = _cupy_csr(ctx, n)

    c = cusparse.spgemm(A, A)
    # ``C`` is sparse and potentially dense-ish: compare through a matvec
    # instead of materialising (n, n) on the host.
    rng = numpy.random.default_rng(ctx.seed + 3)
    v_np = rng.random(n).astype(ctx.dtype)
    v = cupy.asarray(v_np)
    err = _rel_err(c @ v, ref @ (ref @ v_np)) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.spgemm(A, A),
                                  ctx.warmup, ctx.repeat)
    return Result('spgemm', 'OK', shape=f'{n}x{n} @ {n}x{n}',
                  nnz=A.nnz, median_ms=median, min_ms=minimum,
                  gflops=None, rel_err=err,
                  extra={'nnz_C': int(c.nnz)})


@_case('spsm')
def _spsm(ctx: Ctx) -> Result:
    from cupyx import cusparse
    import cupyx.scipy.sparse as sparse

    n = ctx.n_small
    data, indices, indptr = _csr_numpy(ctx, n)
    # Keep the strictly-lower triangle + a dominant diagonal => non-singular.
    import scipy.sparse as sp
    A0 = sp.csr_matrix((data, indices, indptr), shape=(n, n),
                       dtype=ctx.dtype)
    eye = sp.eye(n, format='csr', dtype=ctx.dtype) * ctx.np_dtype.type(n)
    L = (sp.tril(A0, k=-1) + eye).tocsr()
    L.sort_indices()

    A = sparse.csr_matrix((cupy.asarray(L.data), cupy.asarray(L.indices),
                           cupy.asarray(L.indptr)), shape=L.shape)
    rng = numpy.random.default_rng(ctx.seed + 4)
    b_np = rng.random(n).astype(ctx.dtype)
    b = cupy.asarray(b_np)

    x = cusparse.spsm(A, b, lower=True)
    # Residual check, plus a cross-check against scipy when available.
    residual = _rel_err(A @ x, b_np)
    err = residual
    try:
        from scipy.sparse.linalg import spsolve_triangular
        err = _rel_err(x, spsolve_triangular(L.tocsc(), b_np, lower=True))
    except Exception:
        pass

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.spsm(A, b, lower=True),
                                  ctx.warmup, ctx.repeat)
    return Result('spsm', 'OK', shape=f'{n}x{n} triangular',
                  nnz=A.nnz, median_ms=median, min_ms=minimum,
                  gflops=_flops_per_sec(2 * A.nnz, median), rel_err=err,
                  extra={'residual': residual})


@_case('denseToSparse')
def _dense_to_sparse(ctx: Ctx) -> Result:
    from cupyx import cusparse
    n = ctx.n_small
    rng = numpy.random.default_rng(ctx.seed + 5)
    dense_np = rng.random((n, n)).astype(ctx.dtype)
    dense_np[dense_np < 0.9] = 0.0     # ~10% dense, keeps nnz reasonable
    dense = cupy.asarray(dense_np)

    x = cusparse.denseToSparse(dense, format='csr')
    try:
        import scipy.sparse as sp
        ref = sp.csr_matrix(dense_np)
        err = _rel_err(x.toarray(), ref.toarray())
    except Exception:
        err = None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(
            lambda: cusparse.denseToSparse(dense, format='csr'),
            ctx.warmup, ctx.repeat)
    return Result('denseToSparse', 'OK', shape=f'{n}x{n} dense',
                  nnz=int(x.nnz), median_ms=median, min_ms=minimum,
                  rel_err=err)


@_case('sparseToDense')
def _sparse_to_dense(ctx: Ctx) -> Result:
    from cupyx import cusparse
    n = ctx.n_small
    ref = _scipy_csr(ctx, n)
    A = _cupy_csr(ctx, n)

    out = cusparse.sparseToDense(A)
    err = _rel_err(out, ref.toarray()) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.sparseToDense(A),
                                  ctx.warmup, ctx.repeat)
    return Result('sparseToDense', 'OK', shape=f'{n}x{n}',
                  nnz=A.nnz, median_ms=median, min_ms=minimum, rel_err=err)


def _sorted_check(ctx: Ctx, kind: str, sort_by: str = 'r') -> Result:
    """Shared body for csrsort/cscsort/coosort (in-place ops)."""
    from cupyx import cusparse
    import cupyx.scipy.sparse as sparse

    n = ctx.n_small
    k = max(1, min(ctx.nnz_per_row, n))
    rng = numpy.random.default_rng(ctx.seed + 6)
    # Deliberately *unsorted* rows: the sort has to do real work.
    indptr = numpy.zeros(n + 1, dtype=numpy.int32)
    indptr[1:] = numpy.cumsum(numpy.full(n, k, dtype=numpy.int32))
    per_row = [rng.permutation(rng.choice(n, size=k, replace=False))
               for _ in range(n)]
    indices = numpy.ascontiguousarray(numpy.concatenate(per_row),
                                      dtype=numpy.int32)
    data = rng.random(indices.size).astype(ctx.dtype)

    if kind == 'csr':
        A = sparse.csr_matrix((cupy.asarray(data), cupy.asarray(indices),
                               cupy.asarray(indptr)), shape=(n, n))
        call = lambda: cusparse.csrsort(A)  # noqa: E731
        checker = lambda: _blocks_sorted(A.indptr, A.indices)  # noqa: E731
        nnz = A.nnz
    elif kind == 'csc':
        A = sparse.csc_matrix((cupy.asarray(data), cupy.asarray(indices),
                               cupy.asarray(indptr)), shape=(n, n))
        call = lambda: cusparse.cscsort(A)  # noqa: E731
        checker = lambda: _blocks_sorted(A.indptr, A.indices)  # noqa: E731
        nnz = A.nnz
    else:
        rows = numpy.repeat(numpy.arange(n, dtype=numpy.int32), k)
        A = sparse.coo_matrix((cupy.asarray(data),
                               (cupy.asarray(rows), cupy.asarray(indices))),
                              shape=(n, n))
        call = lambda: cusparse.coosort(A, sort_by)  # noqa: E731
        checker = lambda: _coo_sorted(A.row, A.col)  # noqa: E731
        nnz = A.nnz

    call()
    err = 0.0 if checker() else 1.0

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(call, ctx.warmup, ctx.repeat)
    return Result(kind + 'sort', 'OK', shape=f'{n}x{n}', nnz=nnz,
                  median_ms=median, min_ms=minimum, rel_err=err)


def _blocks_sorted(indptr, indices) -> bool:
    """``True`` when every row (column) of a CSR (CSC) matrix is sorted."""
    ptr = cupy.asnumpy(indptr)
    idx = cupy.asnumpy(indices)
    for start, stop in zip(ptr[:-1], ptr[1:]):
        block = idx[start:stop]
        if block.size > 1 and bool((numpy.diff(block) < 0).any()):
            return False
    return True


def _coo_sorted(rows, cols) -> bool:
    """``True`` when a COO matrix is ordered by (row, column)."""
    r = cupy.asnumpy(rows)
    c = cupy.asnumpy(cols)
    if r.size < 2:
        return True
    dr = numpy.diff(r)
    if bool((dr < 0).any()):
        return False
    dc = numpy.diff(c)
    same_row = dr == 0
    if bool(same_row.any()) and bool((dc[same_row] < 0).any()):
        return False
    return True


@_case('csrsort')
def _csrsort(ctx: Ctx) -> Result:
    return _sorted_check(ctx, 'csr')


@_case('cscsort')
def _cscsort(ctx: Ctx) -> Result:
    return _sorted_check(ctx, 'csc')


@_case('coosort', requires=('coosort',))
def _coosort(ctx: Ctx) -> Result:
    return _sorted_check(ctx, 'coo')


@_case('coo2csr')
def _coo2csr(ctx: Ctx) -> Result:
    from cupyx import cusparse
    import cupyx.scipy.sparse as sparse
    import scipy.sparse as sp

    n = ctx.n_small
    data, indices, indptr = _csr_numpy(ctx, n)
    A = sparse.csr_matrix((cupy.asarray(data), cupy.asarray(indices),
                           cupy.asarray(indptr)), shape=(n, n))
    coo = A.tocoo()

    out = cusparse.coo2csr(coo)
    ref = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    err = _rel_err(out.toarray(), ref.toarray())

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.coo2csr(coo),
                                  ctx.warmup, ctx.repeat)
    return Result('coo2csr', 'OK', shape=f'{n}x{n}', nnz=out.nnz,
                  median_ms=median, min_ms=minimum, rel_err=err)


@_case('csr2cscEx2')
def _csr2csc(ctx: Ctx) -> Result:
    from cupyx import cusparse
    n = ctx.n_small
    ref = _scipy_csr(ctx, n)
    A = _cupy_csr(ctx, n)

    out = cusparse.csr2cscEx2(A)
    err = _rel_err(out.toarray(), ref.toarray()) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: cusparse.csr2cscEx2(A),
                                  ctx.warmup, ctx.repeat)
    return Result('csr2cscEx2', 'OK', shape=f'{n}x{n}', nnz=A.nnz,
                  median_ms=median, min_ms=minimum, rel_err=err)


@_case('tocsr', requires=('coo2csr',))
def _tocsr(ctx: Ctx) -> Result:
    """High-level COO -> CSR (exercises coosort + coo2csr)."""
    import cupyx.scipy.sparse as sparse
    import scipy.sparse as sp

    n = ctx.n_small
    data, indices, indptr = _csr_numpy(ctx, n)
    ref = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    rows = numpy.repeat(numpy.arange(n, dtype=numpy.int32),
                        numpy.diff(indptr))
    coo = sparse.coo_matrix((cupy.asarray(data),
                             (cupy.asarray(rows), cupy.asarray(indices))),
                            shape=(n, n))
    out = coo.tocsr()
    err = _rel_err(out.toarray(), ref.toarray())

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: coo.tocsr(), ctx.warmup, ctx.repeat)
    return Result('tocsr', 'OK', shape=f'{n}x{n}', nnz=out.nnz,
                  median_ms=median, min_ms=minimum, rel_err=err)


@_case('tocsc', requires=('csr2cscEx2',))
def _tocsc(ctx: Ctx) -> Result:
    """High-level CSR -> CSC (exercises csr2cscEx2)."""
    n = ctx.n_small
    ref = _scipy_csr(ctx, n)
    A = _cupy_csr(ctx, n)
    out = A.tocsc()
    err = _rel_err(out.toarray(), ref.toarray()) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: A.tocsc(), ctx.warmup, ctx.repeat)
    return Result('tocsc', 'OK', shape=f'{n}x{n}', nnz=A.nnz,
                  median_ms=median, min_ms=minimum, rel_err=err)


@_case('toarray', requires=('sparseToDense',))
def _toarray(ctx: Ctx) -> Result:
    n = ctx.n_small
    ref = _scipy_csr(ctx, n)
    A = _cupy_csr(ctx, n)
    out = A.toarray()
    err = _rel_err(out, ref.toarray()) if ref is not None else None

    median, minimum = (None, None)
    if ctx.bench:
        median, minimum = _timeit(lambda: A.toarray(), ctx.warmup, ctx.repeat)
    return Result('toarray', 'OK', shape=f'{n}x{n}', nnz=A.nnz,
                  median_ms=median, min_ms=minimum, rel_err=err)


# ---------------------------------------------------------------------------
# runner / reporting
# ---------------------------------------------------------------------------
def run_case(case: Case, ctx: Ctx) -> Result:
    missing = [op for op in case.requires if _availability(op) is False]
    if missing:
        return Result(case.name, 'SKIP',
                      reason=f'not provided by libops_sparse.so: '
                             f'{", ".join(missing)}')
    try:
        result = case.fn(ctx)
    except NotImplementedError as exc:
        return Result(case.name, 'SKIP', reason=str(exc).split('\n')[0])
    except Exception as exc:
        return Result(case.name, 'FAIL',
                      reason=f'{type(exc).__name__}: {exc}',
                      extra={'traceback': traceback.format_exc()})

    if ctx.check and result.rel_err is not None:
        tol = _tol(ctx.dtype)
        if result.rel_err > tol:
            result.status = 'FAIL'
            result.reason = f'rel.err {result.rel_err:.3e} > tol {tol:.1e}'
    return result


def print_env(info: dict[str, Any]) -> None:
    print('=' * 78)
    print('environment')
    print('=' * 78)
    for key in ('cupy', 'numpy', 'scipy', 'backend', 'machine',
                'cann_build_version', 'cann_installed_version',
                'cusparse_module', 'ops_sparse_library', 'device_id'):
        if key in info:
            print(f'  {key:24s}: {info[key]}')
    if info.get('cusparse_unavailable_reason'):
        print(f'  {"cusparse error":24s}: '
              f'{info["cusparse_unavailable_reason"]}')
    if info.get('ops_sparse_missing'):
        print(f'  {"ops-sparse missing":24s}: '
              f'{", ".join(info["ops_sparse_missing"])}')

    print('-' * 78)
    print('capability (cupyx.cusparse.check_availability)')
    print('-' * 78)
    avail = info['availability']
    width = max(len(k) for k in avail) + 2
    line = ''
    for name, value in avail.items():
        mark = 'yes' if value else ('no ' if value is False else '?  ')
        line += f'{name:<{width}s}{mark}  '
        if len(line) > 60:
            print('  ' + line.rstrip())
            line = ''
    if line:
        print('  ' + line.rstrip())


def print_results(results: list[Result], dtype: str) -> int:
    print()
    print('=' * 78)
    print(f'results ({dtype})')
    print('=' * 78)
    header = (f'{"op":<15s} {"status":<6s} {"shape":<22s} '
              f'{"nnz":>10s} {"median ms":>10s} {"min ms":>9s} '
              f'{"GFLOP/s":>8s} {"rel.err":>10s}')
    print(header)
    print('-' * len(header))
    failures = 0
    for r in results:
        if r.status == 'FAIL':
            failures += 1
        nnz = '-' if r.nnz is None else f'{r.nnz}'
        med = '-' if r.median_ms is None else f'{r.median_ms:.3f}'
        low = '-' if r.min_ms is None else f'{r.min_ms:.3f}'
        gfs = '-' if r.gflops is None else f'{r.gflops:.2f}'
        err = '-' if r.rel_err is None else f'{r.rel_err:.2e}'
        print(f'{r.name:<15s} {r.status:<6s} {r.shape:<22s} '
              f'{nnz:>10s} {med:>10s} {low:>9s} {gfs:>8s} {err:>10s}')
        if r.reason:
            print(f'{"":<15s} -> {r.reason}')
    print('-' * len(header))
    total = len(results)
    print(f'{total - failures}/{total} cases ok')
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Validate and benchmark cupyx sparse on Ascend.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=int, default=0,
                        help='NPU device id')
    parser.add_argument('--n', type=int, default=20000,
                        help='matrix dimension for the sparse-heavy cases')
    parser.add_argument('--nnz-per-row', type=int, default=20)
    parser.add_argument('--dtype', default='float32',
                        choices=('float32', 'float64'))
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--only', default=None,
                        help='comma separated case names to run')
    parser.add_argument('--skip', default=None,
                        help='comma separated case names to skip')
    parser.add_argument('--no-bench', action='store_true',
                        help='correctness only (skip timing)')
    parser.add_argument('--no-check', action='store_true',
                        help='timing only (skip reference comparison)')
    parser.add_argument('--list', action='store_true',
                        help='print environment/capabilities and exit')
    parser.add_argument('--json', default=None,
                        help='write machine readable results to this path')
    args = parser.parse_args(argv)

    info = env_report(args.device)
    print_env(info)

    cases = list(CASES)
    if args.only:
        wanted = {s.strip() for s in args.only.split(',') if s.strip()}
        cases = [c for c in cases if c.name in wanted]
    if args.skip:
        dropped = {s.strip() for s in args.skip.split(',') if s.strip()}
        cases = [c for c in cases if c.name not in dropped]

    if args.list:
        print()
        print(f'{len(cases)} cases: '
              + ', '.join(c.name for c in cases))
        if args.json:
            _write_json(args.json, info, [])
        return 0

    if info['cusparse_module'] is None:
        print()
        print('cannot benchmark: cuSPARSE/ops-sparse is not available '
              '(see "cusparse error" above).')
        return 2

    ctx = Ctx(n=args.n, nnz_per_row=args.nnz_per_row, dtype=args.dtype,
              repeat=args.repeat, warmup=args.warmup,
              bench=not args.no_bench, check=not args.no_check)

    try:
        cupy.cuda.Device(args.device).use()
        info['device_name'] = _device_name(args.device)
        print(f'  {"device_name":24s}: {info["device_name"]}')
        cupy.zeros(1)          # probe: raises when there is no usable NPU
    except Exception as exc:
        print()
        print(f'cannot use device {args.device}: '
              f'{type(exc).__name__}: {exc}')
        return 2

    results = []
    for case in cases:
        sys.stdout.write(f'  running {case.name} ...')
        sys.stdout.flush()
        result = run_case(case, ctx)
        print(f' {result.status}')
        results.append(result)

    status = print_results(results, args.dtype)
    if args.json:
        _write_json(args.json, info, results)
        print(f'\nwrote {args.json}')
    return status


def _write_json(path: str, info: dict[str, Any],
                results: list[Result]) -> None:
    payload = {
        'env': {k: v for k, v in info.items() if k != 'availability'},
        'availability': info.get('availability', {}),
        'results': [asdict(r) for r in results],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == '__main__':
    sys.exit(main())
