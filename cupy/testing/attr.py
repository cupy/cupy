try:
    import pytest
    _error = None
except ImportError as e:
    _error = e


def is_available():
    return _error is None


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


def get_error():
    return _error


if _error is None:
    gpu = pytest.mark.gpu
    cudnn = pytest.mark.cudnn
    slow = pytest.mark.slow

else:
    def _dummy_callable(*args, **kwargs):
        check_available()
        assert False  # Not reachable

    gpu = _dummy_callable
    cudnn = _dummy_callable
    slow = _dummy_callable


def multi_gpu(gpu_num):
    check_available()
    return pytest.mark.multi_gpu(gpu=gpu_num)
