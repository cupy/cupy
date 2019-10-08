import contextlib
import threading


config = threading.local()
config.divide = None
config.over = None
config.under = None
config.invalid = None


@contextlib.contextmanager
def errstate(*, divide=None, over=None, under=None, invalid=None):
    """
    TODO(hvy): Write docs.
    """
    old_state = seterr(
        divide=divide, over=over, under=under, invalid=invalid)
    try:
        yield  # Return `None` similar to `numpy.errstate`.
    finally:
        seterr(**old_state)


def seterr(*, divide=None, over=None, under=None, invalid=None):
    """
    TODO(hvy): Write docs.
    """
    if divide is not None:
        raise NotImplementedError()
    if over is not None:
        raise NotImplementedError()
    if under is not None:
        raise NotImplementedError()
    if invalid is not None:
        raise NotImplementedError()

    old_state = geterr()

    config.divide = divide
    config.under = under
    config.over = over
    config.invalid = invalid

    return old_state


def geterr():
    """
    TODO(hvy): Write docs.
    """
    return dict(
        divide=config.divide,
        over=config.over,
        under=config.under,
        invalid=config.invalid,
    )
