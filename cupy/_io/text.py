# flake8: NOQA
# "flake8: NOQA" to suppress warning "H104  File contains nothing but comments"
import numpy
import cupy


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """Save an array to a text file.
    Args:
        fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
        X : 1D or 2D array_like
            Data to be saved to a text file.
        fmt : str or sequence of strs, optional
            A single format (%10.5f), a sequence of formats, or a
            multi-format string, e.g. 'Iteration %d -- %10.5f', in which
            case `delimiter` is ignored. For complex `X`, the legal options
            for `fmt` are:
            * a single specifier, `fmt='%.4e'`, resulting in numbers formatted
            like `' (%s+%sj)' % (fmt, fmt)`
            * a full string specifying every real and imaginary part, e.g.
            `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns
            * a list of specifiers, one per column - in this case, the real
            and imaginary part must have separate specifiers,
            e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns
        delimiter : str, optional
            String or character separating columns.
        newline : str, optional
            String or character separating lines.
            .. versionadded:: 1.5.0
        header : str, optional
            String that will be written at the beginning of the file.
            .. versionadded:: 1.7.0
        footer : str, optional
            String that will be written at the end of the file.
            .. versionadded:: 1.7.0
        comments : str, optional
            String that will be prepended to the ``header`` and ``footer``
            strings,
            to mark them as comments. Default: '# ',  as expected by e.g.
            ``numpy.loadtxt``.
            .. versionadded:: 1.7.0
        encoding : {None, str}, optional
            Encoding used to encode the outputfile. Does not apply to output
            streams. If the encoding is something other than 'bytes' or
            'latin1'
            you will not be able to load the file in NumPy versions < 1.14.
            Default
            is 'latin1'.
            .. versionadded:: 1.14.0
    """
    numpy.savetxt(fname, cupy.asnumpy(X), fmt, delimiter, newline, header,
                  footer, comments, encoding)


# TODO(okuta): Implement genfromtxt
