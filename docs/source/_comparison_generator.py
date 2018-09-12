import importlib


def _get_functions(obj):
    return set([
        n for n in dir(obj)
        if (n not in ['test']  # not in blacklist
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
        return obj, ':func:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(
        base_mod, cupy_mod, base_type, klass, exclude_mod):
    base_obj, base_fmt = _import(base_mod, klass)
    base_funcs = _get_functions(base_obj)
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
        cp_cell = cp_fmt.format(f) if f in cp_funcs else r'\-'
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
        '   - {}'.format(f) for f in (cp_funcs - base_funcs)
    ]
    return buf


def _section(
        header, base_mod, cupy_mod,
        base_type='NumPy', klass=None, exclude=None):
    return [
        header,
        '~' * len(header),
        '',
    ] + _generate_comparison_rst(
        base_mod, cupy_mod, base_type, klass, exclude
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
        'numpy', 'cupy')
    buf += _section(
        'Multi-Dimensional Array',
        'numpy', 'cupy', klass='ndarray')
    buf += _section(
        'Linear Algebra',
        'numpy.linalg', 'cupy.linalg')
    buf += _section(
        'Discrete Fourier Transform',
        'numpy.fft', 'cupy.fft')
    buf += _section(
        'Random Sampling',
        'numpy.random', 'cupy.random')

    buf += [
        'SciPy / CuPy APIs',
        '-----------------',
        '',
    ]
    buf += _section(
        'Sparse Matrices',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy')
    buf += _section(
        'Sparse Linear Algebra',
        'scipy.sparse.linalg', 'cupyx.scipy.sparse.linalg', 'SciPy')
    buf += _section(
        'Advanced Linear Algebra',
        'scipy.linalg', 'cupyx.scipy.linalg', 'SciPy', exclude='numpy.linalg')
    buf += _section(
        'Multidimensional Image Processing',
        'scipy.ndimage', 'cupyx.scipy.ndimage', 'SciPy')
    buf += _section(
        'Special Functions',
        'scipy.special', 'cupyx.scipy.special', 'SciPy')

    return '\n'.join(buf)
