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


def _generate_comparison_rst(base_obj, cupy_obj, base_type):
    base_funcs = _get_functions(importlib.import_module(base_obj))
    cp_funcs = _get_functions(importlib.import_module(cupy_obj))

    buf = []
    buf += [
        '.. csv-table::',
        '   :header: {}, CuPy'.format(base_type),
        '',
    ]
    for f in sorted(base_funcs):
        if f in cp_funcs:
            line = '   :obj:`{0}.{1}`, :obj:`{2}.{1}`'.format(
                base_obj, f, cupy_obj)
        else:
            line = '   :obj:`{0}.{1}`, \-'.format(base_obj, f)
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


def _section(header, base_obj, cupy_obj, base_type='NumPy'):
    return [
        header,
        '~' * len(header),
        '',
    ] + _generate_comparison_rst(base_obj, cupy_obj, base_type) + [
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
        'numpy.ndarray', 'cupy.ndarray')
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
        'scipy.linalg', 'cupyx.scipy.linalg', 'SciPy')
    buf += _section(
        'Multidimensional Image Processing',
        'scipy.ndimage', 'cupyx.scipy.ndimage', 'SciPy')
    buf += _section(
        'Special Functions',
        'scipy.special', 'cupyx.scipy.special', 'SciPy')

    return '\n'.join(buf)
