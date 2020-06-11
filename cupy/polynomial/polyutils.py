import operator
import warnings


def _deprecate_as_int(x, desc):
    try:
        return operator.index(x)
    except TypeError as e:
        try:
            ix = int(x)
        except TypeError:
            pass
        else:
            if ix == x:
                warnings.warn(
                    'In future, this will raise TypeError, as {} will '
                    'need to be an integer not just an integral float.'
                    .format(desc),
                    DeprecationWarning,
                    stacklevel=3
                )
                return ix

        raise TypeError('{} must be an integer'.format(desc)) from e
