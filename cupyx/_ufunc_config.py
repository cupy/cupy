import contextlib
import threading

_config = threading.local()


def get_config_divide():
    try:
        value = _config.divide
    except AttributeError:
        value = _config.divide = None
    return value


def get_config_over():
    try:
        value = _config.over
    except AttributeError:
        value = _config.over = None
    return value


def get_config_under():
    try:
        value = _config.under
    except AttributeError:
        value = _config.under = None
    return value


def get_config_invalid():
    try:
        value = _config.invalid
    except AttributeError:
        value = _config.invalid = None
    return value


def get_config_linalg():
    # In favor of performance, the `devInfo` input/output from cuSOLVER routine
    # calls that is necessary to check the validity of the other outputs, are
    # ignored, as D2H copy incurring device synchronizations would otherwise be
    # required.
    try:
        value = _config.linalg
    except AttributeError:
        value = _config.linalg = 'ignore'
    return value


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

    _config.divide = divide
    _config.under = under
    _config.over = over
    _config.invalid = invalid
    _config.linalg = linalg

    return old_state


def geterr():
    """
    TODO(hvy): Write docs.
    """
    return dict(
        divide=get_config_divide(),
        over=get_config_over(),
        under=get_config_under(),
        invalid=get_config_invalid(),
        linalg=get_config_linalg(),
    )
