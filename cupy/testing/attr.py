import pytest

gpu = pytest.mark.gpu
cudnn = pytest.mark.cudnn
slow = pytest.mark.slow


def multi_gpu(gpu_num):
    return pytest.mark.multi_gpu(gpu=gpu_num)
