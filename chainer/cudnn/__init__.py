try:
    from chainer.cudnn import cudnn

    available = cudnn.available
    enabled = cudnn.enabled

    Auto = cudnn.Auto
    get_ptr = cudnn.get_ptr
    get_default_handle = cudnn.get_default_handle
    shutdown = cudnn.shutdown
    get_tensor_desc = cudnn.get_tensor_desc
    get_conv_bias_desc = cudnn.get_conv_bias_desc
    get_filter4d_desc = cudnn.get_filter4d_desc
    get_conv2d_desc = cudnn.get_conv2d_desc
    get_pool2d_desc = cudnn.get_pool2d_desc

except Exception:
    available = False
    enabled = False
