import numpy


_dtype_to_ctype = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}


class _CodeBlock:
    """Code fragment for the readable format.
    """

    def __init__(self, head, codes):
        self._head = '' if head == '' else head + ' '
        self._codes = codes

    def _to_str_list(self, indent_width=0):
        codes = []
        codes.append(' ' * indent_width + self._head + '{')
        for code in self._codes:
            next_indent_width = indent_width + 2
            if isinstance(code, str):
                codes.append(' ' * next_indent_width + code)
            elif isinstance(code, _CodeBlock):
                codes += code._to_str_list(indent_width=next_indent_width)
            else:
                assert False
        codes.append(' ' * indent_width + '}')
        return codes

    def __str__(self):
        """Emit CUDA program like the following format.

        <<head>> {
          <<begin codes>>
          ...;
          <<end codes>>
        }
        """

        return '\n'.join(self._to_str_list())
