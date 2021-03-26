from cupy import _core


def _negative_gcd_error():
    raise TypeError("gcd cannot be computed with boolean arrays")


def _negative_lcm_error():
    raise TypeError("lcm cannot be computed with boolean arrays")


_gcd_preamble = '''
template <typename T> inline __device__ T gcd(T in0, T in1) {
  T r;
  while (in1 != 0) {
    r = in0 % in1;
    in0 = in1;
    in1 = r;
  }
  if (in0 < 0)
    return -in0;
  return in0;
}
'''

gcd = _core.create_ufunc(
    'cupy_gcd',
    (('??->?', _negative_gcd_error),
     'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q'),
    'out0 = gcd(in0, in1)',
    preamble=_gcd_preamble,
    doc='''Computes gcd of ``x1`` and ``x2`` elementwise.

    .. seealso:: :data:`numpy.gcd`

    ''')

_lcm_preamble = _gcd_preamble + '''
template <typename T> inline __device__ T lcm(T in0, T in1) {
  T r = gcd(in0, in1);
  if (r == 0)
    return 0;
  r = in0 / r * in1;
  if (r < 0)
    return -r;
  return r;
}
'''

lcm = _core.create_ufunc(
    'cupy_lcm',
    (('??->?', _negative_lcm_error),
     'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
     'LL->L', 'qq->q', 'QQ->Q'),
    'out0 = lcm(in0, in1)',
    preamble=_lcm_preamble,
    doc='''Computes lcm of ``x1`` and ``x2`` elementwise.

    .. seealso:: :data:`numpy.lcm`

    ''')
