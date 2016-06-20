from nose.plugins import attrib

gpu = attrib.attr('gpu')


def multi_gpu(gpu_num):
    return attrib.attr(gpu=gpu_num)
