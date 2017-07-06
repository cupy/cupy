import unittest

import numpy

import cupy
from cupy import cuda
from cupy import testing

# Setup for optimize einsum
chars = 'abcdefghij'
sizes = numpy.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3])
global_size_dict = {}
for size, char in zip(sizes, chars):
    global_size_dict[char] = size


class TestEinSum(unittest.TestCase):
    def test_einsum_errors(self):
        # Need enough arguments
        with self.assertRaises(ValueError):
            numpy.einsum()
        with self.assertRaises(ValueError):
            numpy.einsum("")

        # subscripts must be a string
        with self.assertRaises(TypeError):
            numpy.einsum(0, 0)

        # other keyword arguments are rejected
        with self.assertRaises(TypeError):
            numpy.einsum("", 0, bad_arg=0)

        # number of operands must match count in subscripts string
        with self.assertRaises(ValueError):
            numpy.einsum("", 0, 0)
        with self.assertRaises(ValueError):
            numpy.einsum(",", 0, [0], [0])

        with self.assertRaises(ValueError):
            numpy.einsum(",", [0])

        # can't have more subscripts than dimensions in the operand
        with self.assertRaises(ValueError):
            numpy.einsum("i", 0)

        with self.assertRaises(ValueError):
            numpy.einsum("ij", [0, 0])

        # invalid subscript character
        with self.assertRaises(ValueError):
            numpy.einsum("i%", [0, 0])
        with self.assertRaises(ValueError):
            numpy.einsum("j$", [0, 0])
        with self.assertRaises(ValueError):
            numpy.einsum("i->&", [0, 0])

        # output subscripts must appear in inumpy.t
        with self.assertRaises(ValueError):
            numpy.einsum("i->ij", [0, 0])

        # output subscripts may only be specified once
        with self.assertRaises(ValueError):
            numpy.einsum("ij->jij", [[0, 0], [0, 0]])

        # dimensions much match when being collapsed
        with self.assertRaises(ValueError):
            numpy.einsum("ii", numpy.arange(6).reshape(2, 3))
        with self.assertRaises(ValueError):
            numpy.einsum("ii->", numpy.arange(6).reshape(2, 3))

        # broadcasting to new dimensions must be enabled explicitly
        with self.assertRaises(ValueError):
            numpy.einsum("i->i", numpy.arange(6).reshape(2, 3))

    def test_einsum_views(self):
        a = numpy.arange(6)
        a.shape = (2, 3)

        b = numpy.einsum("ij", a)
        self.assertTrue(b.base is a)
        testing.assert_array_equal(b, a)

        # transpose
        a = numpy.arange(6)
        a.shape = (2, 3)

        b = numpy.einsum("ji", a)
        self.assertTrue(b.base is a)
        testing.assert_array_equal(b, a.T)

        # diagonal
        a = numpy.arange(9)
        a.shape = (3, 3)

        b = numpy.einsum("ii->i", a)
        self.assertTrue(b.base is a)
        testing.assert_array_equal(b, [a[i, i] for i in range(3)])

        # diagonal with various ways of broadcasting an additional dimension
        a = numpy.arange(27)
        a.shape = (3, 3, 3)

        b = numpy.einsum("jii->ij", a)
        self.assertTrue(b.base is a)
        testing.assert_array_equal(b, [a[:, i, i] for i in range(3)])

        # triple diagonal
        a = numpy.arange(27)
        a.shape = (3, 3, 3)

        b = numpy.einsum("iii->i", a)
        self.assertTrue(b.base is a)
        testing.assert_array_equal(b, [a[i, i, i] for i in range(3)])

        # swap axes
        a = numpy.arange(24)
        a.shape = (2, 3, 4)

        b = numpy.einsum("ijk->jik", a)
        self.assertTrue(b.base is a)
        testing.assert_array_equal(b, a.swapaxes(0, 1))

    def check_einsum_sums(self, dtype):
        # Check various sums.  Does many sizes to exercise unrolled loops.

        # sum(a, axis=-1)
        for n in range(1, 17):
            a = numpy.arange(n, dtype=dtype)
            testing.assert_array_equal(numpy.einsum("i->", a),
                         numpy.sum(a, axis=-1).astype(dtype))

        # trace(a)
        for n in range(1, 17):
            a = numpy.arange(n*n, dtype=dtype).reshape(n, n)
            testing.assert_array_equal(numpy.einsum("ii", a),
                         numpy.trace(a).astype(dtype))

        # outer(a,b)
        for n in range(1, 17):
            a = numpy.arange(3, dtype=dtype) + 1
            b = numpy.arange(n, dtype=dtype) + 1
            testing.assert_array_equal(numpy.einsum("i,j", a, b),
                         numpy.outer(a, b))

        # matvec(a,b) / a.dot(b) where a is matrix, b is vector
        for n in range(1, 17):
            a = numpy.arange(4*n, dtype=dtype).reshape(4, n)
            b = numpy.arange(n, dtype=dtype)
            testing.assert_array_equal(numpy.einsum("ij, j", a, b),
                         numpy.dot(a, b))

            c = numpy.einsum("ij,j", a, b)
            testing.assert_array_equal(c, numpy.dot(a, b))

        for n in range(1, 17):
            a = numpy.arange(4*n, dtype=dtype).reshape(4, n)
            b = numpy.arange(n, dtype=dtype)
            testing.assert_array_equal(numpy.einsum("ji,j", a.T, b.T),
                         numpy.dot(b.T, a.T))

            c = numpy.einsum("ji,j", a.T, b.T)
            testing.assert_array_equal(c, numpy.dot(b.T, a.T))

        # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
        for n in range(1, 17):
            if n < 8 or dtype != 'f2':
                a = numpy.arange(4*n, dtype=dtype).reshape(4, n)
                b = numpy.arange(n*6, dtype=dtype).reshape(n, 6)
                testing.assert_array_equal(numpy.einsum("ij,jk", a, b),
                             numpy.dot(a, b))

        for n in range(1, 17):
            a = numpy.arange(4*n, dtype=dtype).reshape(4, n)
            b = numpy.arange(n*6, dtype=dtype).reshape(n, 6)
            c = numpy.einsum("ij,jk", a, b)
            testing.assert_array_equal(c, numpy.dot(a, b).astype(dtype))

        # matrix triple product (note this is not currently an efficient
        # way to multiply 3 matrices)
        a = numpy.arange(12, dtype=dtype).reshape(3, 4)
        b = numpy.arange(20, dtype=dtype).reshape(4, 5)
        c = numpy.arange(30, dtype=dtype).reshape(5, 6)
        if dtype != 'f2':
            testing.assert_array_equal(numpy.einsum("ij,jk,kl", a, b, c),
                         a.dot(b).dot(c))

        d = numpy.einsum("ij,jk,kl", a, b, c)
        tgt = a.dot(b)
        tgt = tgt.dot(c).astype(dtype)
        testing.assert_array_equal(d, tgt)

        # tensordot(a, b)
        if numpy.dtype(dtype) != numpy.dtype('f2'):
            a = numpy.arange(60, dtype=dtype).reshape(3, 4, 5)
            b = numpy.arange(24, dtype=dtype).reshape(4, 3, 2)
            testing.assert_array_equal(numpy.einsum("ijk, jil -> kl", a, b),
                         numpy.tensordot(a, b, axes=([1, 0], [0, 1])))

            c = numpy.einsum("ijk,jil->kl", a, b)
            testing.assert_array_equal(c, numpy.tensordot(a, b,
                         axes=([1, 0], [0, 1])))

        # logical_and(logical_and(a!=0, b!=0), c!=0)
        a = numpy.array([1,   3,   -2,   0,   12,  13,   0,   1], dtype=dtype)
        b = numpy.array([0,   3.5, 0.,   -2,  0,   1,    3,   12], dtype=dtype)
        c = numpy.array([True, True, False, True, True, False, True, True])
        testing.assert_array_equal(numpy.einsum("i,i,i->i", a, b, c,
            numpy.logical_and(numpy.logical_and(a != 0, b != 0), c != 0)))

        a = numpy.arange(9, dtype=dtype)
        testing.assert_array_equal(numpy.einsum(",i->", 3, a), 3*numpy.sum(a))
        testing.assert_array_equal(numpy.einsum("i,->", a, 3), 3*numpy.sum(a))

        # Various stride0, contiguous, and SSE aligned variants
        for n in range(1, 25):
            a = numpy.arange(n, dtype=dtype)
            if numpy.dtype(dtype).itemsize > 1:
                testing.assert_array_equal(numpy.einsum("i,i", a, a), numpy.dot(a, a))
                testing.assert_array_equal(numpy.einsum("i,->i", a, 2), 2*a)
                testing.assert_array_equal(numpy.einsum(",i->i", 2, a), 2*a)
                testing.assert_array_equal(numpy.einsum("i,->", a, 2), 2*numpy.sum(a))
                testing.assert_array_equal(numpy.einsum(",i->", 2, a), 2*numpy.sum(a))

                testing.assert_array_equal(numpy.einsum("i,i", a[1:], a[:-1]),
                             numpy.dot(a[1:], a[:-1]))
                testing.assert_array_equal(numpy.einsum("i,->i", a[1:], 2), 2*a[1:])
                testing.assert_array_equal(numpy.einsum(",i->i", 2, a[1:]), 2*a[1:])
                testing.assert_array_equal(numpy.einsum("i,->", a[1:], 2),
                             2*numpy.sum(a[1:]))
                testing.assert_array_equal(numpy.einsum(",i->", 2, a[1:]),
                             2*numpy.sum(a[1:]))

        # An object array, summed as the data type
        a = numpy.arange(9, dtype=object)

        b = numpy.einsum("i->", a, dtype=dtype, casting='unsafe')
        testing.assert_array_equal(b, numpy.sum(a))
        testing.assert_array_equal(b.dtype, numpy.dtype(dtype))

    def test_einsum_sums_int8(self):
        self.check_einsum_sums('i1')

    def test_einsum_sums_uint8(self):
        self.check_einsum_sums('u1')

    def test_einsum_sums_int16(self):
        self.check_einsum_sums('i2')

    def test_einsum_sums_uint16(self):
        self.check_einsum_sums('u2')

    def test_einsum_sums_int32(self):
        self.check_einsum_sums('i4')

    def test_einsum_sums_uint32(self):
        self.check_einsum_sums('u4')

    def test_einsum_sums_int64(self):
        self.check_einsum_sums('i8')

    def test_einsum_sums_uint64(self):
        self.check_einsum_sums('u8')

    def test_einsum_sums_float16(self):
        self.check_einsum_sums('f2')

    def test_einsum_sums_float32(self):
        self.check_einsum_sums('f4')

    def test_einsum_sums_float64(self):
        self.check_einsum_sums('f8')

    def test_einsum_sums_longdouble(self):
        self.check_einsum_sums(numpy.longdouble)

    def test_einsum_broadcast(self):
        dims = [2, 3, 4, 5]
        a = numpy.arange(numpy.prod(dims)).reshape(dims)
        v = numpy.arange(dims[2])
        ref = numpy.einsum('ijkl,k->ijl', a, v)
        testing.assert_array_equal(numpy.einsum('ijkl,k', a, v), ref)
