_ON = True
_OFF = False
_AUTO = None
_values = {
    'on': _ON,
    'ON': _ON,
    _ON: _ON,
    'off': _OFF,
    'OFF': _OFF,
    _OFF: _OFF,
    'auto': _AUTO,
    'AUTO': _AUTO,
    _AUTO: _AUTO,
}
_reprs = {
    _ON: 'ON',
    _OFF: 'OFF',
    _AUTO: 'AUTO',
}

_caches = {}


class Flag(object):

    """Ternary flag object for variables.

    It takes three values: ON, OFF, and AUTO.

    ON and OFF flag can be evaluated as a boolean value. These are converted
    to True and False, respectively. AUTO flag cannot be converted to boolean.
    In this case, ValueError is raised.

    Args:
        name (str, bool, or None): Name of the flag. Following values are
            allowed:

            - ``'on'``, ``'ON'``, or ``True`` for ON value
            - ``'off'``, ``'OFF'``, or ``False`` for OFF value
            - ``'auto'``, ``'AUTO'``, or ``None`` for AUTO value

    """
    def __new__(cls, name):
        if name in _flags:
            return _flags[name]
        flag = super(Flag, cls).__new__(cls)
        flag.value = _values[name]
        return flag

    def __bool__(self):
        value = self.value
        if value is _AUTO:
            raise ValueError('Flag AUTO cannot be converted to boolean')
        return value

    __nonzero__ = __bool__

    def __reduce__(self):
        return Flag, (self.value,)

    def __repr__(self):
        return _reprs[self.value]

    def __eq__(self, other):
        return self is Flag(other)

    def __ne__(self, other):
        return self is not Flag(other)

    def __lt__(self, other):
        raise RuntimeError('no order is defined between flags')

    __le__ = __gt__ = __ge__ = __lt__

    def __hash__(self):
        return hash(self.value)


_flags = {}

ON = Flag('ON')
"""Equivalent to Flag('on')."""

OFF = Flag('OFF')
"""Equivalent to Flag('off')."""

AUTO = Flag('AUTO')
"""Equivalent to Flag('auto')."""

_flags = {
    'on': ON,
    'ON': ON,
    _ON: ON,
    'off': OFF,
    'OFF': OFF,
    _OFF: OFF,
    'auto': AUTO,
    'AUTO': AUTO,
    _AUTO: AUTO,
}


def aggregate_flags(flags):
    """Returns an aggregated flag given a sequence of flags.

    If both ON and OFF are found, this function raises an error. Otherwise,
    either of ON and OFF that appeared is returned. If all flags are AUTO, then
    it returns AUTO.

    Args:
        flags (sequence of Flag): Input flags.

    Returns:
        Flag: The result of aggregation.

    """
    on = any([flag is ON for flag in flags])
    off = any([flag is OFF for flag in flags])
    if on:
        if off:
            raise ValueError('ON and OFF flags cannot be mixed.')
        else:
            return ON
    else:
        return OFF if off else AUTO
