import threading


thread_local = threading.local()


cpdef inline bint is_old_fusing() except? -1:
    try:
        return thread_local.is_old_fusing
    except AttributeError:
        thread_local.is_old_fusing = False
    return False


cpdef inline bint is_new_fusing() except? -1:
    try:
        return thread_local.is_new_fusing
    except AttributeError:
        thread_local.is_new_fusing = False
    return False


cpdef inline bint is_fusing() except? -1:
    return is_old_fusing() or is_new_fusing()


def check_not_runtime():
    assert is_new_fusing()


def call_ufunc(fusion_op, *args, **kwargs):
    if is_new_fusing():
        return thread_local.history.call_ufunc(fusion_op, *args, **kwargs)
    import cupy
    return cupy._core.fusion._call_ufunc(fusion_op, *args, **kwargs)


def call_reduction(fusion_op, *args, **kwargs):
    if is_new_fusing():
        return thread_local.history.call_reduction(fusion_op, *args, **kwargs)
    import cupy
    return cupy._core.fusion._call_reduction(fusion_op, *args, **kwargs)


def call_indexing(fusion_op, *args, **kwargs):
    return thread_local.history.call_indexing(fusion_op, *args, **kwargs)
