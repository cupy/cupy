import unittest

from chainer import testing
import numpy

import einsum


def _from_str_subscript(subscript):
    # subscript should be lower case (a-z)
    return [
        (Ellipsis if char == '@' else ord(char) - ord('a'))
        for char in subscript.replace('...', '@')
    ]


@testing.parameterize(*testing.product_dict(
    [
        {'subscripts': 'ij,jk->ik', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': ',ij->i', 'shapes': ((), (3, 4),)},
        {'subscripts': 'kj,ji->ik', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': 'ij,jk,kl->il', 'shapes': ((5, 2), (2, 3), (3, 4))},
        {'subscripts': 'ij,ij->i', 'shapes': ((2, 3), (2, 3))},
        {'subscripts': 'ij,jk', 'shapes': ((2, 3), (3, 4))},
        {'subscripts': 'i->', 'shapes': ((3,),)},
        {'subscripts': 'ii', 'shapes': ((2, 2),)},
        {'subscripts': 'ii->i', 'shapes': ((2, 2),)},
        {'subscripts': 'j,j', 'shapes': ((3,), (3))},
        {'subscripts': 'j,ij', 'shapes': ((3,), (2, 3))},
        {'subscripts': 'j,iij', 'shapes': ((3,), (2, 2, 3))},
        {'subscripts': 'iij,kkj', 'shapes': ((2, 2, 3), (4, 4, 3))},
        {'subscripts': '...ij,...jk->...ik',
         'shapes': ((2, 1, 2, 3), (2, 1, 3, 4))},
        {'subscripts': 'i...j,jk...->k...i', 'shapes': ((4, 2, 3), (3, 5, 2))},
        {'subscripts': 'ii...,...jj', 'shapes': ((2, 2, 4), (4, 3, 3))},
        {'subscripts': '...i,i', 'shapes': ((2, 2, 3), (3,))},
        {'subscripts': 'i...,i->...i', 'shapes': ((3, 2, 2), (3,))},
    ],
    testing.product({
        'dtype': [numpy.float32, numpy.float64],
        'subscript_type': ['str', 'int'],
    }),
))
class TestEinSum(unittest.TestCase):

    def setUp(self):
        self.inputs = tuple([
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.shapes
        ])
        self.forward_answer = numpy.einsum(*self._get_args(self.inputs))
        self.g = self._setup_tensor(
            -1, 1, self.forward_answer.shape, self.dtype)
        self.gg_inputs = tuple([
            self._setup_tensor(-1, 1, shape, self.dtype)
            for shape in self.shapes
        ])
        self.op = lambda *xs: einsum.einsum(*self._get_args(xs))

    def _get_args(self, xs):
        if self.subscript_type == 'str':
            return (self.subscripts,) + xs
        else:
            args = []
            subscripts = self.subscripts.split('->')
            for in_subscript, x in zip(subscripts[0].split(','), xs):
                args.extend([x, _from_str_subscript(in_subscript)])
            if len(subscripts) == 2:
                args.append(_from_str_subscript(subscripts[1]))
            return tuple(args)

    def _setup_tensor(self, _min, _max, shape, dtype):
        return numpy.random.uniform(_min, _max, shape).astype(dtype)

    def check_forward(self, inputs_data, atol=1e-4, rtol=1e-5):
        out = self.op(*inputs_data)
        testing.assert_allclose(out, self.forward_answer, atol, rtol)

    def test_forward_cpu(self):
        self.check_forward(self.inputs)


testing.run_module(__name__, __file__)
