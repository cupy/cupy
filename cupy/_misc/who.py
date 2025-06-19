import sys
import cupy


def who(vardict=None):
    """Print the CuPy arrays in the given dictionary.

    Prints out the name, shape, bytes and type of all of the ndarrays
    present in `vardict`.

    If there is no dictionary passed in or `vardict` is None then returns
    CuPy arrays in the globals() dictionary (all CuPy arrays in the
    namespace).

    Args:
        vardict : (None or dict)  A dictionary possibly containing ndarrays.
                  Default is globals() if `None` specified


    .. admonition:: Example

        >>> a = cupy.arange(10)
        >>> b = cupy.ones(20)
        >>> cupy.who()
        Name            Shape            Bytes            Type
        ===========================================================
        <BLANKLINE>
        a               10               80               int64
        b               20               160              float64
        <BLANKLINE>
        Upper bound on total bytes  =       240
        >>> d = {'x': cupy.arange(2.0),
        ... 'y': cupy.arange(3.0), 'txt': 'Some str',
        ... 'idx':5}
        >>> cupy.who(d)
        Name            Shape            Bytes            Type
        ===========================================================
        <BLANKLINE>
        x               2                16               float64
        y               3                24               float64
        <BLANKLINE>
        Upper bound on total bytes  =       40

    """

    # Implementation is largely copied from numpy.who()
    if vardict is None:
        frame = sys._getframe().f_back
        vardict = frame.f_globals
    sta = []
    cache = {}
    for name in sorted(vardict.keys()):
        if isinstance(vardict[name], cupy.ndarray):
            var = vardict[name]
            idv = id(var)
            if idv in cache.keys():
                namestr = '{} ({})'.format(name, cache[idv])
                original = 0
            else:
                cache[idv] = name
                namestr = name
                original = 1
            shapestr = ' x '.join(map(str, var.shape))
            bytestr = str(var.nbytes)
            sta.append(
                [namestr, shapestr, bytestr, var.dtype.name, original]
            )

    maxname = 0
    maxshape = 0
    maxbyte = 0
    totalbytes = 0
    for k in range(len(sta)):
        val = sta[k]
        if maxname < len(val[0]):
            maxname = len(val[0])
        if maxshape < len(val[1]):
            maxshape = len(val[1])
        if maxbyte < len(val[2]):
            maxbyte = len(val[2])
        if val[4]:
            totalbytes += int(val[2])

    if len(sta) > 0:
        sp1 = max(10, maxname)
        sp2 = max(10, maxshape)
        sp3 = max(10, maxbyte)
        prval = 'Name {} Shape {} Bytes {} Type'.format(
            sp1 * ' ', sp2 * ' ', sp3 * ' '
        )
        print("{}\n{}\n".format(prval, "=" * (len(prval) + 5)))

    for k in range(len(sta)):
        val = sta[k]
        print(
            '{} {} {} {} {} {} {}'.format(
                val[0],
                ' ' * (sp1 - len(val[0]) + 4),
                val[1],
                ' ' * (sp2 - len(val[1]) + 5),
                val[2],
                ' ' * (sp3 - len(val[2]) + 5),
                val[3],
            )
        )
    print('\nUpper bound on total bytes  =       {}'.format(totalbytes))
