from nose.plugins import attrib

gpu = attrib.attr('gpu')
cudnn = attrib.attr('gpu', 'cudnn')


def multi_gpu(gpu_num):
    return attrib.attr(gpu=gpu_num)
