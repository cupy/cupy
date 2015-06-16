def attr(*args, **kwargs):
    """Decorator that adds attributes.

    """
    def wrap(x):
        for k in args:
            setattr(x, k, True)
        for k, v in kwargs.items():
            setattr(x, k, v)
        return x
    return wrap

gpu   = attr('gpu')
cudnn = attr('gpu', 'cudnn')
