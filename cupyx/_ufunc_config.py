import contextlib
import threading

config = threading.local()
config.divide = None
config.over = None
config.under = None
config.invalid = None
# In favor of performance, the `devInfo` input/output from cuSOLVER routine
# calls that is necessary to check the validity of the other outputs, are
# ignored, as D2H copy incurring device synchronizations would otherwise be
# required.
config.linalg = 'ignore'


@contextlib.contextmanager
def errstate(*, divide=None, over=None, under=None, invalid=None, linalg=None):
    """
    TODO(hvy): Write docs.
    """
    old_state = seterr(
        divide=divide, over=over, under=under, invalid=invalid, linalg=linalg)
    try:
        yield  # Return `None` similar to `numpy.errstate`.
    finally:
        seterr(**old_state)


def seterr(*, divide=None, over=None, under=None, invalid=None, linalg=None):
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
    if linalg is not None:
        if linalg not in ('ignore', 'raise'):
            raise NotImplementedError()

    old_state = geterr()

    config.divide = divide
    config.under = under
    config.over = over
    config.invalid = invalid
    config.linalg = linalg

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
        linalg=config.linalg,
    )
