
def set_module(module):
    """Decorator for overriding __module__ on a function or class.

    .. seealso:: :func:`numpy.core.overrides.set_module`

    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
