import importlib


def _get_functions(obj, blacklist=None):
    return set([
        n for n in dir(obj)
        if ((blacklist is None or n not in blacklist)
            and callable(getattr(obj, n))  # callable
            and not isinstance(getattr(obj, n), type)  # not class
            and n[0].islower()  # starts with lower char
            and not n.startswith('__')  # not special methods
            )
    ])


def _import(mod, klass):
    obj = importlib.import_module(mod)
    if klass:
        obj = getattr(obj, klass)
        return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
    else:
        # ufunc is not a function
        return obj, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(
        base_mod, cupy_mod, base_type, klass, exclude_mod, blacklist):
    base_obj, base_fmt = _import(base_mod, klass)
    base_funcs = _get_functions(base_obj, blacklist)
    cp_obj, cp_fmt = _import(cupy_mod, klass)
    cp_funcs = _get_functions(cp_obj)

    if exclude_mod:
        exclude_obj, _ = _import(exclude_mod, klass)
        exclude_funcs = _get_functions(exclude_obj)
        base_funcs -= exclude_funcs
        cp_funcs -= exclude_funcs

    buf = []
    buf += [
        '.. csv-table::',
        '   :header: {}, CuPy'.format(base_type),
        '',
    ]
    for f in sorted(base_funcs):
        base_cell = base_fmt.format(f)
        cp_cell = r'\-'
        if f in cp_funcs:
            cp_cell = cp_fmt.format(f)
            if getattr(base_obj, f) is getattr(cp_obj, f):
                cp_cell = '{} (*alias of* {})'.format(cp_cell, base_cell)
        line = '   {}, {}'.format(base_cell, cp_cell)
        buf.append(line)

    buf += [
        '',
        '.. Summary:',
        '   Number of NumPy functions: {}'.format(len(base_funcs)),
        '   Number of functions covered by CuPy: {}'.format(
            len(cp_funcs & base_funcs)),
        '   CuPy specific functions:',
    ] + [
        '   - {}'.format(f) for f in sorted(cp_funcs - base_funcs)
    ]
    return buf


def _section(
        header, base_mod, cupy_mod,
        base_type='NumPy', klass=None, exclude_mod=None, blacklist=None):
    return [
        header,
        '~' * len(header),
        '',
    ] + _generate_comparison_rst(
        base_mod, cupy_mod, base_type, klass, exclude_mod, blacklist
    ) + [
        '',
    ]


def generate():
    buf = []

    buf += [
        'NumPy / CuPy APIs',
        '-----------------',
        '',
    ]
    buf += _section(
        'Module-Level',
        'numpy', 'cupy', blacklist=[
            'add_docstring',
            'add_newdoc',
            'add_newdoc_ufunc',
            'fastCopyAndTranspose',
            'test',
        ])
    buf += _section(
        'Multi-Dimensional Array',
        'numpy', 'cupy', klass='ndarray')
    buf += _section(
        'Linear Algebra',
        'numpy.linalg', 'cupy.linalg', blacklist=['test'])
    buf += _section(
        'Discrete Fourier Transform',
        'numpy.fft', 'cupy.fft', blacklist=['test'])
    buf += _section(
        'Random Sampling',
        'numpy.random', 'cupy.random', blacklist=['test'])

    buf += [
        'SciPy / CuPy APIs',
        '-----------------',
        '',
    ]
    buf += _section(
        'Discrete Fourier Transform',
        'scipy.fft', 'cupyx.scipy.fft', 'SciPy', blacklist=['test'])
    buf += _section(
        'Legacy Discrete Fourier Transform',
        'scipy.fftpack', 'cupyx.scipy.fftpack', 'SciPy', blacklist=['test'])
    buf += _section(
        'Advanced Linear Algebra',
        'scipy.linalg', 'cupyx.scipy.linalg', 'SciPy',
        exclude_mod='numpy.linalg', blacklist=['test'])
    buf += _section(
        'Multidimensional Image Processing',
        'scipy.ndimage', 'cupyx.scipy.ndimage', 'SciPy', blacklist=['test'])
    buf += _section(
        'Signal processing',
        'scipy.signal', 'cupyx.scipy.signal', 'SciPy', blacklist=['test'])
    buf += _section(
        'Sparse Matrices',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy', blacklist=['test'])
    buf += _section(
        'Sparse Linear Algebra',
        'scipy.sparse.linalg', 'cupyx.scipy.sparse.linalg', 'SciPy',
        blacklist=['test'])
    buf += _section(
        'Compressed sparse graph routines',
        'scipy.sparse.csgraph', 'cupyx.scipy.sparse.csgraph', 'SciPy',
        blacklist=['test'])
    buf += _section(
        'Special Functions',
        'scipy.special', 'cupyx.scipy.special', 'SciPy',
        blacklist=['test'])
    buf += _section(
        'Statistical Functions',
        'scipy.stats', 'cupyx.scipy.stats', 'SciPy',
        blacklist=['test'])


    # numpy.array_api is not ready yet...
    #    buf += [
    #        'NumPy / CuPy Array APIs',
    #        '-----------------------',
    #        '',
    #    ]
    #    buf += _section(
    #        'Python array API compliance',
    #        'numpy.array_api', 'cupy.array_api', 'NumPy')

    return '\n'.join(buf)
