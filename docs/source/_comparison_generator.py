import importlib
import inspect

import numpy


_footnotes = {}


def _get_functions(obj, exclude=None):
    return set([
        n for n, target in [(n, getattr(obj, n)) for n in dir(obj)]
        if (
            # not in exclude list
            (exclude is None or n not in exclude)
            # not module:
            and not inspect.ismodule(target)
            # not constant:
            and not isinstance(target, (int, float, bool, str, numpy.bool_))
            # not exceptions or warning classes:
            and (not inspect.isclass(target) or
                 not issubclass(target, (BaseException,)))
            # not private/special method:
            and not n.startswith('_')
            )
    ])


def _import(mod, klass):
    obj = importlib.import_module(mod)
    if klass:
        obj = getattr(obj, klass)
        return obj, ':obj:`{}.{}.{{}}`'.format(mod, klass)
    else:
        # ufunc is not a function
        return obj, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(
        base_mod, cupy_mod, base_type, klass, exclude_mod, exclude,
        footnotes=None):
    base_obj, base_fmt = _import(base_mod, klass)
    base_funcs = _get_functions(base_obj, exclude)
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
        footnote_id = None
        if footnotes is not None and f in footnotes:
            footnote_id = _footnotes.setdefault(
                footnotes[f], f'f{len(_footnotes)}')

        base_cell = base_fmt.format(f)
        cp_cell = r'\-'
        if f in cp_funcs:
            cp_cell = cp_fmt.format(f)
            if getattr(base_obj, f) is getattr(cp_obj, f):
                cp_cell = '{} (*alias of* {})'.format(cp_cell, base_cell)
        if footnote_id is not None:
            cp_cell += f' [#{footnote_id}]_'
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
        base_type='NumPy', klass=None, exclude_mod=None, exclude=None,
        footnotes=None, header_char='~'):
    return [
        header,
        header_char * len(header),
        '',
    ] + _generate_comparison_rst(
        base_mod, cupy_mod, base_type, klass, exclude_mod, exclude, footnotes
    ) + [
        '',
    ]


_deprecated = 'Not supported as it has been deprecated in NumPy.'
_np_matrix = (
    'Use of :class:`numpy.matrix` is discouraged in NumPy and thus'
    ' we have no plan to add it to CuPy.')
_np_poly1d = (
    'Use of :class:`numpy.poly1d` is discouraged in NumPy and thus'
    ' we have stopped adding functions with the interface.')
_dtype_na = (
    '`object` and string dtypes are not supported in GPU and thus'
    ' left unimplemented in CuPy.')
_dtype_time = (
    '`datetime64` and `timedelta64` dtypes are currently unsupported.')
_dtype_rec = (
    'Structured arrays and record arrays are currently unsupported.')
_float_error_state = (
    'Floating point error handling depends on CPU-specific features'
    ' which is not available in GPU.')
_byte_order = 'Not supported as GPUs only support little-endian byte-encoding.'


def generate():
    buf = []

    buf += [
        'NumPy / CuPy APIs',
        '-----------------',
        '',
    ]
    buf += _section(
        'Module-Level',
        'numpy', 'cupy', exclude=[
            'add_docstring',
            'add_newdoc',
            'add_newdoc_ufunc',
            '_add_newdoc_ufunc',
            'fastCopyAndTranspose',
            'kernel_version',
            'test',
            'Tester',
        ], footnotes={
            'Datetime64': _deprecated,  # NumPy 1.20
            'Uint64': _deprecated,  # NumPy 1.20
            'mafromtxt': _deprecated,  # NumPy 1.17
            'alen': _deprecated,  # NumPy 1.18
            'asscalar': _deprecated,  # NumPy 1.16
            'loads': _deprecated,  # NumPy 1.15
            'ndfromtxt': _deprecated,  # NumPy 1.17
            'set_numeric_ops': _deprecated,  # NumPy 1.16

            'asmatrix': _np_matrix,
            'bmat': _np_matrix,
            'mat': _np_matrix,
            'matrix': _np_matrix,

            'poly': _np_poly1d,
            'polyder': _np_poly1d,
            'polydiv': _np_poly1d,
            'polyint': _np_poly1d,

            'Bytes0': _dtype_na,  # also deprecated in NumPy 1.20
            'bytes0': _dtype_na,
            'bytes_': _dtype_na,
            'character': _dtype_na,
            'chararray': _dtype_na,
            'compare_chararrays': _dtype_na,
            'flexible': _dtype_na,
            'object0': _dtype_na,
            'object_': _dtype_na,
            'Str0': _dtype_na,  # also deprecated in NumPy 1.20
            'str0': _dtype_na,
            'str_': _dtype_na,
            'string_': _dtype_na,
            'unicode_': _dtype_na,
            'void': _dtype_na,
            'void0': _dtype_na,

            'busday_count': _dtype_time,
            'busday_offset': _dtype_time,
            'busdaycalendar': _dtype_time,
            'is_busday': _dtype_time,
            'isnat': _dtype_time,
            'datetime64': _dtype_time,
            'datetime_as_string': _dtype_time,
            'datetime_data': _dtype_time,
            'timedelta64': _dtype_time,

            'fromregex': _dtype_rec,
            'recarray': _dtype_rec,
            'recfromcsv': _dtype_rec,
            'recfromtxt': _dtype_rec,
            'record': _dtype_rec,

            'errstate': _float_error_state,
            'geterr': _float_error_state,
            'geterrcall': _float_error_state,
            'geterrobj': _float_error_state,
            'seterr': _float_error_state,
            'seterrcall': _float_error_state,
            'seterrobj': _float_error_state,
        })
    buf += _section(
        'Multi-Dimensional Array',
        'numpy', 'cupy', klass='ndarray', footnotes={
            'tostring': _deprecated,

            'byteswap': _byte_order,
            'newbyteorder': _byte_order,
        })
    buf += _section(
        'Linear Algebra',
        'numpy.linalg', 'cupy.linalg', exclude=['test'])
    buf += _section(
        'Discrete Fourier Transform',
        'numpy.fft', 'cupy.fft', exclude=['test'])
    buf += _section(
        'Random Sampling',
        'numpy.random', 'cupy.random', exclude=['test'])
    buf += _section(
        'Polynomials',
        'numpy.polynomial', 'cupy.polynomial', exclude=['test'])
    buf += _section(
        'Power Series',
        'numpy.polynomial.polynomial', 'cupy.polynomial.polynomial',
        header_char='"')
    buf += _section(
        'Polyutils',
        'numpy.polynomial.polyutils', 'cupy.polynomial.polyutils',
        header_char='"')

    buf += [
        'SciPy / CuPy APIs',
        '-----------------',
        '',
    ]
    buf += _section(
        'Discrete Fourier Transform',
        'scipy.fft', 'cupyx.scipy.fft', 'SciPy', exclude=['test'])
    buf += _section(
        'Legacy Discrete Fourier Transform',
        'scipy.fftpack', 'cupyx.scipy.fftpack', 'SciPy', exclude=['test'])
    buf += _section(
        'Interpolation',
        'scipy.interpolate', 'cupyx.scipy.interpolate', 'SciPy',
        exclude=['test'])
    buf += _section(
        'Advanced Linear Algebra',
        'scipy.linalg', 'cupyx.scipy.linalg', 'SciPy',
        exclude_mod='numpy.linalg', exclude=['test'])
    buf += _section(
        'Multidimensional Image Processing',
        'scipy.ndimage', 'cupyx.scipy.ndimage', 'SciPy', exclude=['test'])
    buf += _section(
        'Signal processing',
        'scipy.signal', 'cupyx.scipy.signal', 'SciPy', exclude=['test'])
    buf += _section(
        'Sparse Matrices',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy', exclude=['test'])
    buf += _section(
        'Sparse Matrices: COOrdinate Format',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy', klass='coo_matrix', 
        exclude=['test'])
    buf += _section(
        'Sparse Matrices: Compressed Sparse Column',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy', klass='csc_matrix', 
        exclude=['test'])
    buf += _section(
        'Sparse Matrices: Compressed Sparse Row',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy', klass='csr_matrix', 
        exclude=['test'])
    buf += _section(
        'Sparse Matrices: DIAgonal Storage',
        'scipy.sparse', 'cupyx.scipy.sparse', 'SciPy', klass='dia_matrix', 
        exclude=['test'])
    buf += _section(
        'Sparse Linear Algebra',
        'scipy.sparse.linalg', 'cupyx.scipy.sparse.linalg', 'SciPy',
        exclude=['test'])
    buf += _section(
        'Compressed sparse graph routines',
        'scipy.sparse.csgraph', 'cupyx.scipy.sparse.csgraph', 'SciPy',
        exclude=['test'])
    buf += _section(
        'Special Functions',
        'scipy.special', 'cupyx.scipy.special', 'SciPy',
        exclude=['test'])
    buf += _section(
        'Statistical Functions',
        'scipy.stats', 'cupyx.scipy.stats', 'SciPy',
        exclude=['test'])

    # numpy.array_api is not ready yet...
    #    buf += [
    #        'NumPy / CuPy Array APIs',
    #        '-----------------------',
    #        '',
    #    ]
    #    buf += _section(
    #        'Python array API compliance',
    #        'numpy.array_api', 'cupy.array_api', 'NumPy')

    buf += [
        '',
        '.. rubric:: Footnotes',
        '',
    ] + [
        f'.. [#{footnote_id}] {footnote}'
        for footnote, footnote_id in _footnotes.items()
    ]

    return '\n'.join(buf)
