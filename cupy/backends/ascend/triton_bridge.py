"""triton-ascend bridge: write kernels in Python, launch on Ascend NPU.

This is the M6 optional layer of docs/ascend/CustomKernel.md (§3A):

* ``@triton.jit`` Python kernels are compiled by triton-ascend and launched on
  the same aclrt runtime numpy-ascend uses -- **zero-copy**: cupy arrays are
  passed to the kernel as raw device pointers (``data_ptr``) plus
  shape/strides, never copied or made contiguous.
* The layer is optional: when triton-ascend is not installed (or the
  installed ``triton`` is a non-Ascend fork, e.g. triton-cpu) every entry
  point degrades gracefully -- either to a user-supplied ``fallback`` or to a
  clear ``RuntimeError``.

Public API
----------
- ``is_available()``        -- strict triton-ascend detection (cached)
- ``CuPyTensorAdapter``     -- zero-copy tensor view for triton kernels
- ``make_grid``             -- 1-D grid helper
- ``jit_ufunc``             -- decorator turning a ``@triton.jit`` kernel into
                               a cupy-friendly ufunc callable
"""

from __future__ import annotations

import functools
import os

__all__ = [
    'is_available',
    'CuPyTensorAdapter',
    'make_grid',
    'TritonUfunc',
    'jit_ufunc',
    'get_current_stream_ptr',
]


# ---------------------------------------------------------------------------
# availability probing
# ---------------------------------------------------------------------------
#: cached probe result; ``None`` = not probed yet. Tests may reset this.
_AVAILABILITY: bool | None = None


def _probe() -> bool:
    """Strictly detect triton-ascend.

    A plain ``triton`` import is NOT enough: the CPU fork (triton-cpu) and
    upstream NVIDIA triton both import as ``triton``. We accept only:
    1. the ``triton_ascend`` companion package, or
    2. a triton whose backend registry advertises an ascend/npu backend,
       *and* an ascend-capable driver.
    """
    try:
        import triton  # noqa: F401
    except Exception:
        return False

    try:
        import triton_ascend  # type: ignore # noqa: F401
        return True
    except Exception:
        pass

    try:
        from triton.backends import backends  # type: ignore
        if not any(
                'ascend' in name.lower() or 'npu' in name.lower()
                for name in list(backends) + [type(b).__name__ for b in backends.values()]):
            return False
    except Exception:
        return False

    # the fork must also expose a working driver, not just a stale registry
    try:
        from triton.runtime import driver  # type: ignore
        active = driver.active
        is_active = getattr(active, 'is_active', None)
        if callable(is_active):
            return bool(is_active())
        return True
    except Exception:
        return False


def is_available() -> bool:
    """Whether a real triton-ascend backend is importable and active."""
    global _AVAILABILITY
    if _AVAILABILITY is None:
        # CUPY_ASCEND_FORCE_TRITON=1 eases bring-up on partially installed
        # environments; the honest answer is still the probe result.
        if os.environ.get('CUPY_ASCEND_FORCE_TRITON') == '1':
            try:
                import triton  # noqa: F401
                _AVAILABILITY = True
            except Exception:
                _AVAILABILITY = False
        else:
            _AVAILABILITY = _probe()
    return _AVAILABILITY


def _reset_availability_cache() -> None:
    """Test hook: force re-probing on next :func:`is_available` call."""
    global _AVAILABILITY
    _AVAILABILITY = None


# ---------------------------------------------------------------------------
# zero-copy tensor adapter
# ---------------------------------------------------------------------------
class CuPyTensorAdapter:
    """Zero-copy view of a cupy array for triton kernel arguments.

    Exposes the minimal tensor interface triton launchers consume
    (``data_ptr()``, ``shape``, ``stride``, ``dtype``). No data is ever
    copied, padded or made contiguous -- kernels must consume the real
    strides (triton index math handles them natively).

    Accepts any duck-typed array with ``data.ptr`` (cupy ndarray or an
    equivalent device-memory holder) plus ``shape``/``dtype``/``itemsize``
    and byte-unit ``_strides`` (cupy) or element-unit ``strides`` (numpy-like).
    """

    __slots__ = ('_arr',)

    def __init__(self, arr):
        if not hasattr(arr, 'data') or not hasattr(arr.data, 'ptr'):
            raise TypeError(
                'CuPyTensorAdapter expects a cupy ndarray (device memory, '
                '.data.ptr); got %s. Host (numpy) arrays must be copied to '
                'the device explicitly.' % type(arr).__name__)
        self._arr = arr

    @property
    def array(self):
        """The wrapped array (never a copy)."""
        return self._arr

    def data_ptr(self) -> int:
        return int(self._arr.data.ptr)

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self) -> tuple:
        return tuple(self._arr.shape)

    @property
    def stride(self) -> tuple:
        """Element-unit strides, normalised from cupy's byte-unit strides."""
        itemsize = getattr(self._arr, 'itemsize', 1)
        byte_strides = getattr(self._arr, '_strides', None)
        if byte_strides is None:
            byte_strides = tuple(s * itemsize for s in self._arr.strides)
        return tuple(s // itemsize for s in byte_strides)

    def __repr__(self) -> str:
        return 'CuPyTensorAdapter(%r)' % (self._arr,)


def make_grid(n: int, block: int = 1024) -> tuple[int]:
    """1-D grid covering ``n`` elements in ``block``-sized programs."""
    if block <= 0:
        raise ValueError('block must be positive')
    return (max(1, (int(n) + block - 1) // block),)


def get_current_stream_ptr() -> int:
    """The active cupy/ascend stream pointer, for triton stream bridging."""
    from cupy.xpu import stream as stream_module
    return stream_module.get_current_stream_ptr()


# ---------------------------------------------------------------------------
# ufunc wrapper
# ---------------------------------------------------------------------------
class TritonUfunc:
    """A ``@triton.jit`` kernel wrapped as a cupy-friendly ufunc callable.

    Calling convention::

        my_add(x, y, out=z)                    # explicit out
        my_add(x, y, out=z, BLOCK=256)         # triton constexpr via kwargs

    Behaviour:
    * inputs are wrapped in :class:`CuPyTensorAdapter` (zero-copy);
    * the launch happens on the **current cupy stream**;
    * when triton-ascend is unavailable the user ``fallback`` runs instead
      (otherwise ``RuntimeError``).
    """

    def __init__(self, jit_fn, name: str | None = None, fallback=None):
        self._jit_fn = jit_fn
        self.name = name or getattr(jit_fn, '__name__', 'triton_ufunc')
        self._fallback = fallback

    @property
    def fallback(self):
        return self._fallback

    def _launch(self, grid, args, kwargs):
        """Actual triton launch; isolated for testability."""
        return self._jit_fn[grid](*args, **kwargs)

    def __call__(self, *args, out=None, grid=None, **kwargs):
        if not is_available():
            if self._fallback is not None:
                return self._fallback(*args, out=out, **kwargs)
            raise RuntimeError(
                'triton-ascend is not available; install it (or provide a '
                f'fallback for {self.name!r})')

        if out is None:
            raise TypeError(f'{self.name}() requires an explicit out= array')
        tensors = [CuPyTensorAdapter(a) for a in (*args, out)]

        n = int(out.size) if hasattr(out, 'size') else 1
        if grid is None:
            block = int(kwargs.get('BLOCK', 1024))
            grid = make_grid(n, block)

        # bridge the cupy stream into the launch (single-runtime guarantee)
        kwargs.setdefault('stream', get_current_stream_ptr())
        return self._launch(grid, tensors, kwargs)


def jit_ufunc(name: str | None = None, fallback=None):
    """Decorator: turn a ``@triton.jit`` kernel into a :class:`TritonUfunc`.

    Usage::

        @jit_ufunc(name='my_add', fallback=lambda x, y, out: out[...] ...)
        def my_add(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
            ...
    """
    def decorator(fn):
        return TritonUfunc(fn, name=name, fallback=fallback)
    return decorator
