import collections

import cupy


class SingleViewCalculator(object):
    def __init__(self, ioperand, subscript):
        self.subscript = subscript
        self.ioperand = ioperand
        self.labels = set(self.subscript)
        self.label_to_axis = collections.defaultdict(list)
        for i, label in enumerate(subscript):
            self.label_to_axis[label].append(i)

    def __call__(self):
        self.result = self.ioperand
        count_dict = collections.Counter(self.subscript)
        for label in set(self.subscript):
            if count_dict[label] == 1:
                continue
            axes_to_diag = []
            for i, char in enumerate(self.subscript):
                if char == label:
                    axes_to_diag.append(i)
            for axis in reversed(axes_to_diag[1:]):
                self.result = self.result.diagonal(0, axis, axes_to_diag[0])
                self.result = cupy.rollaxis(self.result, -1, axes_to_diag[0])
                self.subscript = self.subscript[:axis] + \
                                 self.subscript[axis+1:]


class SummedViewCalculator(object):
    def __init__(self, ioperand, input_subscript, output_subscript):
        self.ioperand = ioperand
        self.subscript = input_subscript
        self.label_to_summed = set(input_subscript) - set(output_subscript)
        self.axes_to_summed = []
        for i, label in enumerate(input_subscript):
            if label in self.label_to_summed:
                self.axes_to_summed.append(i)

    def __call__(self):
        if self.axes_to_summed:
            self.result = cupy.sum(self.ioperand,
                                   axis=tuple(self.axes_to_summed))
        else:
            self.result = self.ioperand
        for label in self.label_to_summed:
            self.subscript = self.subscript.replace(label, '')


class TransposedViewCalculator(object):
    def __init__(self, ioperand, input_subscript, output_subscript):
        assert len(input_subscript) == len(output_subscript)
        assert set(input_subscript) == set(output_subscript)
        self.ioperand = ioperand
        self.input_subscript = input_subscript
        self.output_subscript = output_subscript

    def __call__(self):
        transpose_orders = []
        for label in self.output_subscript:
            transpose_orders.append(self.input_subscript.find(label))
        if transpose_orders == sorted(transpose_orders):
            self.result = self.ioperand
        else:
            self.result = self.ioperand.transpose(transpose_orders)


class CombinedViewCalculator(object):
    def __init__(self, subscripts, ioperands):
        self.subscripts = subscripts
        self.ioperands = ioperands

    def __call__(self):
        self.result = self.ioperands[0]
        for ioperand in self.ioperands[1:]:
            # TODO(fukatani): add up at here if enable.
            self.result = cupy.tensordot(self.result, ioperand, axes=0)
        self.subscript = ''.join(self.subscripts)


def get_dummy_labels(label_list):
    dummy_label_set = set([])
    count_dict = collections.Counter(label_list)
    for label, count in count_dict.items():
        if count >= 2:
            dummy_label_set.add(label)
    return dummy_label_set


def einsum(subscripts, *inputs):
    # TODO(fukatani): raise Exception.
    # TODO(fukatani): Support '...'
    # TODO(fukatani): Support optimization.

    subscripts = subscripts.replace(' ', '')
    inputs = [a if isinstance(a, cupy.ndarray) else cupy.asarray(a)
              for a in inputs]

    arrow_pos = subscripts.find('->')
    if arrow_pos == -1:
        input_subscripts = subscripts
        label_list = list(input_subscripts.replace(',', ''))
        out_label_set = set(label_list) - get_dummy_labels(label_list)
        output_subscript = ''.join(sorted(list(out_label_set)))
    else:
        input_subscripts = subscripts[:arrow_pos]
        output_subscript = subscripts[arrow_pos+2:]

    input_subscripts_list = input_subscripts.split(',')
    i_parsers = []
    for subscript, ioperand in zip(input_subscripts_list, inputs):
        calc = SingleViewCalculator(ioperand, subscript)
        calc()
        i_parsers.append(calc)

    if len(inputs) >= 2:
        i_subscripts = [i_parser.subscript for i_parser in i_parsers]
        i_results = [i_parser.result for i_parser in i_parsers]
        calc = CombinedViewCalculator(i_subscripts, i_results)
        calc()
        calc = SingleViewCalculator(calc.result, calc.subscript)
        calc()
    else:
        calc = i_parsers[0]

    calc = SummedViewCalculator(calc.result, calc.subscript,
                                output_subscript)
    calc()
    calc = TransposedViewCalculator(calc.result, calc.subscript,
                                    output_subscript)
    calc()
    return calc.result
