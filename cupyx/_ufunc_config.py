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


def get_config_fallback_mode():
    try:
        value = _config.fallback_mode
    except AttributeError:
        value = _config.fallback_mode = 'ignore'
    return value


@contextlib.contextmanager
def errstate(*, divide=None, over=None, under=None,
             invalid=None, linalg=None, fallback_mode=None):
    """
    TODO(hvy): Write docs.
    """
    old_state = seterr(
        divide=divide, over=over, under=under,
        invalid=invalid, linalg=linalg, fallback_mode=fallback_mode)
    try:
        yield  # Return `None` similar to `numpy.errstate`.
    finally:
        seterr(**old_state)


def seterr(*, divide=None, over=None, under=None,
           invalid=None, linalg=None, fallback_mode=None):
    """
    TODO(hvy): Write docs.
    """
    old_state = geterr()

    if divide is not None:
        raise NotImplementedError()
    if over is not None:
        raise NotImplementedError()
    if under is not None:
        raise NotImplementedError()
    if invalid is not None:
        raise NotImplementedError()
    if linalg is not None:
        if linalg in ('ignore', 'raise'):
            _config.linalg = linalg
        else:
            raise NotImplementedError()
    if fallback_mode is not None:
        if fallback_mode in ['print', 'warn', 'ignore', 'raise']:
            _config.fallback_mode = fallback_mode
        elif fallback_mode in ['log', 'call']:
            raise NotImplementedError
        else:
            raise ValueError(
                '{} is not a valid dispatch type'.format(fallback_mode))

    _config.divide = divide
    _config.under = under
    _config.over = over
    _config.invalid = invalid

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
        fallback_mode=get_config_fallback_mode(),
    )
