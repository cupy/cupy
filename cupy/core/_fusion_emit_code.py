import string

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


class _CodeBlock(object):
    """Code fragment for the readable format.

    RFC(asi1024): I implemented this class to emit human readble CUDA code,
    but I feel the current behavior is not intuitive...
    """

    def __init__(self, *args, **kwargs):
        codes = []
        for item in args:
            if isinstance(item, str):
                code = string.Template(item).substitute(**kwargs)
            elif isinstance(item, list):
                code = _CodeBlock(*item, **kwargs)
            elif isinstance(item, _CodeBlock):
                code = item
            else:
                assert False, 'Not supported: {}'.format(type(item))
            codes.append(code)
        self.codes = codes

    def _to_str_list(self, indent_width=0):
        codes = []
        for code in self.codes:
            if isinstance(code, str):
                codes.append(' ' * indent_width + code)
            elif isinstance(code, _CodeBlock):
                new_indent_width = indent_width + 4
                codes += code._to_str_list(indent_width=new_indent_width)
        return codes

    def __str__(self):
        return '\n'.join(self._to_str_list())
