from cupy import _util
from cupy._core import core
from cupy.cuda import compiler


_cuda_compute = False


cpdef _get_cuda_compute():
    global _cuda_compute

    if _cuda_compute is False:
        try:
            from cuda import compute
        except ImportError:
            _cuda_compute = None
        else:
            _cuda_compute = compute
    return _cuda_compute


cdef _compile_cpp_to_ltoir(str src):
    # cupy include paths: op sources may include cupy headers
    # (complex.cuh, float16.cuh)
    options = core.assemble_cupy_compiler_options(())
    return compiler._compile_module_with_cache(src, options, to_ltoir=True)


@_util.memoize(for_each_device=True)
def _make_raw_op(str src, str name):
    ltoir = _compile_cpp_to_ltoir(src)
    return _get_cuda_compute().op.RawOp(ltoir=ltoir, name=name)
