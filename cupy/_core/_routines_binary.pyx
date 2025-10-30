from cupy._core._kernel import create_ufunc


cdef _create_bit_op(name, op, no_bool, doc='', scatter_op=None):
    types = () if no_bool else ('??->?',)
    return create_ufunc(
        'cupy_' + name,
        types + ('bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
                 'LL->L', 'qq->q', 'QQ->Q'),
        'out0 = in0 %s in1' % op,
        doc=doc, scatter_op=scatter_op)


cdef _bitwise_and = _create_bit_op(
    'bitwise_and', '&', False,
    '''Computes the bitwise AND of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_and`

    ''',
    scatter_op='and')


cdef _bitwise_or = _create_bit_op(
    'bitwise_or', '|', False,
    '''Computes the bitwise OR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_or`

    ''',
    scatter_op='or')


cdef _bitwise_xor = _create_bit_op(
    'bitwise_xor', '^', False,
    '''Computes the bitwise XOR of two arrays elementwise.

    Only integer and boolean arrays are handled.

    .. seealso:: :data:`numpy.bitwise_xor`

    ''',
    scatter_op='xor')


cdef _invert = create_ufunc(
    'cupy_invert',
    (('?->?', 'out0 = !in0'), 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I',
     'l->l', 'L->L', 'q->q', 'Q->Q'),
    'out0 = ~in0',
    doc='''Computes the bitwise NOT of an array elementwise.

    Only integer and boolean arrays are handled.

    .. note::
        :func:`cupy.bitwise_not` is an alias for :func:`cupy.invert`.

    .. seealso:: :data:`numpy.invert`

    ''')


cdef _bitwise_count = create_ufunc(
    'cupy_bitwise_count',
    (
        'b->B', 'B->B', 'h->B', 'H->B', 'i->B', 'I->B',
        'l->B', 'L->B', 'q->B', 'Q->B',
    ),
    'out0 = _cupy_bitcount(in0);',
    doc='''Computes the number of 1-bits in each element.

    Only integer arrays are handled.

    .. seealso:: :data:`numpy.bitwise_count`

    ''',
    preamble='''
#include <cupy/cuda_workaround.h>

template <typename T>
__device__ inline int _cupy_bitcount(T x) {
    if constexpr (sizeof(T) <= 4) {
        unsigned int ux = static_cast<unsigned int>(x);
        if constexpr (std::is_signed<T>::value) {
            if(x<0){
                ux = ~ux + 1u;
            }
        }
        return static_cast<int>(__popc(ux));
    } else {
        unsigned long long ux = static_cast<unsigned long long>(x);
        if constexpr (std::is_signed<T>::value) {
            if(x<0){
                ux = ~ux + 1ull;
            }
        }
        return static_cast<int>(__popcll(ux));
    }
}
''')


cdef _left_shift = _create_bit_op(
    'left_shift', '<<', True,
    '''Shifts the bits of each integer element to the left.

    Only integer arrays are handled.

    .. seealso:: :data:`numpy.left_shift`

    ''')


cdef _right_shift = _create_bit_op(
    'right_shift', '>>', True,
    '''Shifts the bits of each integer element to the right.

    Only integer arrays are handled

    .. seealso:: :data:`numpy.right_shift`

    ''')


# Variables to expose to Python
# (cythonized data cannot be exposed to Python, even with cpdef.)
bitwise_and = _bitwise_and
bitwise_or = _bitwise_or
bitwise_xor = _bitwise_xor
bitwise_count = _bitwise_count
invert = _invert
left_shift = _left_shift
right_shift = _right_shift
