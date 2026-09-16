# placeholder to reduce core code modification to upstream cupy

def _is_fusing():
    return False

def is_fusing():
    return False


def fuse(*args, **kwargs):
    """No-op stand-in for ``cupy._core.fusion.fuse`` (fusion is not ported).

    Upstream uses it as a decorator factory (``@fuse()``); with fusion disabled
    the upstream decorator is transparent too, so returning the wrapped function
    unchanged keeps the decorated code working -- e.g. ``cupy.unique``'s
    ``_unique_update_mask_equal_nan``, the only user in the tree.
    """
    if len(args) == 1 and not kwargs and callable(args[0]):
        return args[0]

    def _decorator(func):
        return func
    return _decorator
