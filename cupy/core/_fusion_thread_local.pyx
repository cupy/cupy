import threading


thread_local = threading.local()


cpdef inline bint is_fusing() except? -1:
    try:
        return thread_local.history is not None
    except AttributeError:
        thread_local.history = None
    return False


def check_not_runtime():
    assert thread_local.history is not None


def call_ufunc(fusion_op, *args, **kwargs):
    return thread_local.history.call_ufunc(fusion_op, *args, **kwargs)


def call_reduction(fusion_op, *args, **kwargs):
    return thread_local.history.call_reduction(fusion_op, *args, **kwargs)


def call_indexing(fusion_op, *args, **kwargs):
    return thread_local.history.call_indexing(fusion_op, *args, **kwargs)
