import numpy

import cupy


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """Returns the string representation of an array.

    Args:
        arr (array_like): Input array. It should be able to feed to
            :func:`cupy.asnumpy`.
        max_line_width (int): The maximum number of line lengths.
        precision (int): Floating point precision. It uses the current printing
            precision of NumPy.
        suppress_small (bool): If ``True``, very small numbers are printed as
            zeros

    Returns:
        str: The string representation of ``arr``.

    .. seealso:: :func:`numpy.array_repr`

    """
    return numpy.array_repr(cupy.asnumpy(arr), max_line_width, precision,
                            suppress_small)


def array_str(arr, max_line_width=None, precision=None, suppress_small=None):
    """Returns the string representation of the content of an array.

    Args:
        arr (array_like): Input array. It should be able to feed to
            :func:`cupy.asnumpy`.
        max_line_width (int): The maximum number of line lengths.
        precision (int): Floating point precision. It uses the current printing
            precision of NumPy.
        suppress_small (bool): If ``True``, very small number are printed as
            zeros.

    .. seealso:: :func:`numpy.array_str`

    """
    return numpy.array_str(cupy.asnumpy(arr), max_line_width, precision,
                           suppress_small)


def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=numpy._NoValue, formatter=None, threshold=None,
                 edgeitems=None, sign=None, floatmode=None, suffix="",
                 legacy=None):
    """Return a string representation of an array.

    Args:
        a (array_like): Input array. It should be able to feed to
            :func:`cupy.asnumpy`.
        max_line_width : int, optional
            Inserts newlines if text is longer than `max_line_width`.
            Defaults to ``numpy.get_printoptions()['linewidth']``.
        precision : int or None, optional
            Floating point precision.
            Defaults to ``numpy.get_printoptions()['precision']``.
        suppress_small : bool, optional
            Represent numbers "very close" to zero as zero; default is False.
            Very close is defined by precision: if the precision is 8, e.g.,
            numbers smaller (in absolute value) than 5e-9 are represented as
            zero.
            Defaults to ``numpy.get_printoptions()['suppress']``.
        separator : str, optional
            Inserted between elements.
        prefix : str, optional
        suffix : str, optional
            The length of the prefix and suffix strings are used to respectively
            align and wrap the output. An array is typically printed as::
            prefix + array2string(a) + suffix
            The output is left-padded by the length of the prefix string, and
            wrapping is forced at the column ``max_line_width - len(suffix)``.
            It should be noted that the content of prefix and suffix strings are
            not included in the output.
        style : _NoValue, optional
            Has no effect, do not use.
            .. deprecated:: 1.14.0
        formatter : dict of callables, optional
            If not None, the keys should indicate the type(s) that the respective
            formatting function applies to.  Callables should return a string.
            Types that are not specified (by their corresponding keys) are handled
            by the default formatters.  Individual types for which a formatter
            can be set are:
            - 'bool'
            - 'int'
            - 'timedelta' : a `numpy.timedelta64`
            - 'datetime' : a `numpy.datetime64`
            - 'float'
            - 'longfloat' : 128-bit floats
            - 'complexfloat'
            - 'longcomplexfloat' : composed of two 128-bit floats
            - 'void' : type `numpy.void`
            - 'numpystr' : types `numpy.string_` and `numpy.unicode_`
            Other keys that can be used to set a group of types at once are:
            - 'all' : sets all types
            - 'int_kind' : sets 'int'
            - 'float_kind' : sets 'float' and 'longfloat'
            - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
            - 'str_kind' : sets 'numpystr'
        threshold : int, optional
            Total number of array elements which trigger summarization
            rather than full repr.
            Defaults to ``numpy.get_printoptions()['threshold']``.
        edgeitems : int, optional
            Number of array items in summary at beginning and end of
            each dimension.
            Defaults to ``numpy.get_printoptions()['edgeitems']``.
        sign : string, either '-', '+', or ' ', optional
            Controls printing of the sign of floating-point types. If '+', always
            print the sign of positive values. If ' ', always prints a space
            (whitespace character) in the sign position of positive values.  If
            '-', omit the sign character of positive values.
            Defaults to ``numpy.get_printoptions()['sign']``.
        floatmode : str, optional
            Controls the interpretation of the `precision` option for
            floating-point types.
            Defaults to ``numpy.get_printoptions()['floatmode']``.
            Can take the following values:
            - 'fixed': Always print exactly `precision` fractional digits,
            even if this would print more or fewer digits than
            necessary to specify the value uniquely.
            - 'unique': Print the minimum number of fractional digits necessary
            to represent each value uniquely. Different elements may
            have a different number of digits.  The value of the
            `precision` option is ignored.
            - 'maxprec': Print at most `precision` fractional digits, but if
            an element can be uniquely represented with fewer digits
            only print it with that many.
            - 'maxprec_equal': Print at most `precision` fractional digits,
            but if every element in the array can be uniquely
            represented with an equal number of fewer digits, use that
            many digits for all elements.
        legacy : string or `False`, optional
            If set to the string `'1.13'` enables 1.13 legacy printing mode. This
            approximates numpy 1.13 print output by including a space in the sign
            position of floats and different behavior for 0d arrays. If set to
            `False`, disables legacy mode. Unrecognized strings will be ignored
            with a warning for forward compatibility.
            .. versionadded:: 1.14.0

        .. seealso:: :func:`numpy.array2string`
    """
    return numpy.array2string(cupy.asnumpy(a), max_line_width=max_line_width,
                              precision=precision,
                              suppress_small=suppress_small,
                              separator=separator,
                              prefix=prefix,
                              formatter=formatter,
                              threshold=threshold,
                              edgeitems=edgeitems,
                              sign=sign,
                              floatmode=floatmode,
                              suffix=suffix,
                              legacy=legacy)
