class CodeBlock:
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
            elif isinstance(code, CodeBlock):
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
