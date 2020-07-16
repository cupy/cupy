from cupy.core._kernel import create_ufunc


_complex_cast_copy = '''
template<typename T, typename U>
__device__ void cast_copy(const U& x, T& y) {y = T(x);}
template<typename T>
__device__ void cast_copy(const float16& x, complex<T>& y) {
    // explicitly use user-defined type conversion functions instead of
    // implicit conversions to avoid any potential issue of multiple
    // candidates available
    y = x.operator complex<T>();
}
template<typename T, typename U>
__device__ void cast_copy(const complex<U>& x, complex<T>& y) {
    y = complex<T>(x);
}
template<typename T, typename U>
__device__ void cast_copy(const complex<U>& x, T& y) {y = T(x.real());}
'''


elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'cast_copy(in0, out0)',
    preamble=_complex_cast_copy, default_casting='unsafe')


elementwise_copy_where = create_ufunc(
    'cupy_copy_where',
    ('??->?', 'b?->b', 'B?->B', 'h?->h', 'H?->H', 'i?->i', 'I?->I', 'l?->l',
     'L?->L', 'q?->q', 'Q?->Q', 'e?->e', 'f?->f', 'd?->d', 'F?->F', 'D?->D'),
    'if (in1) cast_copy(in0, out0)',
    preamble=_complex_cast_copy, default_casting='unsafe')
# complex numbers requires out0 = complex<T>(in0)
