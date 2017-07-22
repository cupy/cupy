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


# TODO(fukatani): Delete here.
# if __name__ == '__main__':
#     A = numpy.arange(9).reshape(3, 3)
#     assert (my_einsum('ii', A) == numpy.einsum('ii', A)).all()
#
#     A = numpy.arange(8).reshape(2, 2, 2)
#     assert (my_einsum('ijk', A) == numpy.einsum('ijk', A)).all()
#     assert (my_einsum('iii', A) == numpy.einsum('iii', A)).all()
#     assert (my_einsum('iij', A) == numpy.einsum('iij', A)).all()
#     assert (my_einsum('iji', A) == numpy.einsum('iji', A)).all()
#     assert (my_einsum('iii->i', A) == numpy.einsum('iii->i', A)).all()
#     assert (my_einsum('ijj->ij', A) == numpy.einsum('ijj->ij', A)).all()
#     assert (my_einsum('iij->ij', A) == numpy.einsum('iij->ij', A)).all()
#     assert (my_einsum('iji->ij', A) == numpy.einsum('iji->ij', A)).all()
#
#     assert (my_einsum('ijk->ikj', A) == numpy.einsum('ijk->ikj', A)).all()
#     assert (my_einsum('ijk->jik', A) == numpy.einsum('ijk->jik', A)).all()
#     assert (my_einsum('kji->ikj', A) == numpy.einsum('kji->ikj', A)).all()
#
#     A = numpy.arange(16).reshape(2, 2, 2, 2)
#     assert (my_einsum('iijk->ijk', A) == numpy.einsum('iijk->ijk', A)).all()
#     assert (my_einsum('ijkj->ijk', A) == numpy.einsum('ijkj->ijk', A)).all()
#     assert (my_einsum('ijkj->kij', A) == numpy.einsum('ijkj->kij', A)).all()
#
#     assert (my_einsum('iiij->ij', A) == numpy.einsum('iiij->ij', A)).all()
#     assert (my_einsum('iiji->ij', A) == numpy.einsum('iiji->ij', A)).all()
#     assert (my_einsum('iijj->ij', A) == numpy.einsum('iijj->ij', A)).all()
#     assert (my_einsum('ijij->ij', A) == numpy.einsum('ijij->ij', A)).all()
#     assert (my_einsum('jiji->ji', A) == numpy.einsum('jiji->ji', A)).all()
#
#     assert (my_einsum('iiij->j', A) == numpy.einsum('iiij->j', A)).all()
#     assert (my_einsum('iiij->i', A) == numpy.einsum('iiij->i', A)).all()
#     assert (my_einsum('ijii->j', A) == numpy.einsum('ijii->j', A)).all()
#     assert (my_einsum('ijii->i', A) == numpy.einsum('ijii->i', A)).all()
#     assert (my_einsum('ijij', A) == numpy.einsum('ijij', A)).all()
#
#     A = numpy.arange(3)
#     B = numpy.arange(4)
#     C = numpy.arange(2)
#     assert (my_einsum('i,j', A, B) == numpy.einsum('i,j', A, B)).all()
#     assert (my_einsum('i,j,k', A, B, C) == numpy.einsum('i,j,k', A, B, C)).all()
#
#     A = numpy.arange(4).reshape(2, 2)
#     B = numpy.arange(4).reshape(2, 2)
#     C = numpy.arange(2)
#     assert (my_einsum('ij,kl->ijkl', A, B) == numpy.einsum('ij,kl->ijkl', A, B)).all()
#     assert (my_einsum('ij,kl,m->ijklm', A, B, C) == numpy.einsum('ij,kl,m->ijklm', A, B, C)).all()
#
#     assert (my_einsum('ij,ij->ij', A, B) == numpy.einsum('ij,ij->ij', A, B)).all()
#     assert (my_einsum('ij,ji->ij', A, B) == numpy.einsum('ij,ji->ij', A, B)).all()

