import contextlib
import string

import six

from cupy.core import compile


class Templater(object):

    def __init__(self, **kwargs):
        self.repl = kwargs

    def __call__(self, s):
        return string.Template(s).substitute(**self.repl)


class _IndentedConstructContext(object):

    def __init__(self, emitter, indent):
        self.emitter = emitter
        self.indent = indent

    def __enter__(self):
        self.emitter.incr_indent(self.indent)
        return self

    def __exit__(self, type, value, traceback):
        self.emitter.incr_indent(-self.indent)

    def emit_construct(self, code):
        self.emitter.incr_indent(-self.indent)
        for line in code.split('\n'):
            self.emitter.emit_line(line)
        self.emitter.incr_indent(self.indent)


class KernelCodeEmitter(object):

    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        self._s = six.StringIO()
        self._indent = 0
        self._indent_str = ''

    def incr_indent(self, indent):
        self._indent += indent
        self._indent_str = ' ' * self._indent

    def indent_lines(self, code, indent):
        code = code.split('\n')
        ind = ' ' * (self._indent + indent)
        lines = []
        for line in code:
            lines.append(ind + line)
        return '\n'.join(lines)

    def write(self, code):
        self._s.write(code)

    def emit_line(self, line):
        s = self._s
        s.write(self._indent_str)
        s.write(line)
        s.write('\n')

    def emit_lines(self, lines):
        s = self._s
        ind = self._indent_str
        for line in lines.split('\n'):
            s.write(ind)
            s.write(line)
            s.write('\n')

    def indented_construct(self, indent=2):
        return _IndentedConstructContext(self, indent)

    def get_function(self, options):
        code = self._s.getvalue()
        module = compile.compile_with_cache(code, options)
        return module.get_function(self.kernel_name)

    @contextlib.contextmanager
    def construct_kernel_entry_function(
            self, param_list):

        kernel_params_decl = param_list.get_kernel_params_decl()
        temp = Templater(
            kernel_name=self.kernel_name,
            kernel_params_decl=kernel_params_decl)

        self.emit_line('// Kernel function')
        with self.indented_construct() as c:
            c.emit_construct(temp(
                '''extern "C" __global__ void ${kernel_name}(${kernel_params_decl}) {'''))  # NOQA
            yield
            c.emit_construct('''}''')
