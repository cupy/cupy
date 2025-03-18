import warnings


def __getattr__(name):
    import cupyx.scipy.sparse
    if hasattr(cupyx.scipy.sparse, name):
        msg = 'cupy.sparse is deprecated. Use cupyx.scipy.sparse instead.'
        warnings.warn(msg, DeprecationWarning)
        return getattr(cupyx.scipy.sparse, name)
    raise AttributeError(
        "module 'cupy.sparse' has no attribute {!r}".format(name))
