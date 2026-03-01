"""Determination of spline kernel weights (adapted from SciPy)

See more verbose comments for each case there:
https://github.com/scipy/scipy/blob/eba29d69846ab1299976ff4af71c106188397ccc/scipy/ndimage/src/ni_splines.c#L7

``spline_weights_inline`` is a dict where the key is the spline order and the
value is the spline weight initialization code.
"""
from __future__ import annotations


spline_weights_inline = {}

# Note: This order = 1 case is currently unused (order = 1 has a different code
# path in _interp_kernels.py). I think that existing code is a bit more
# efficient.
spline_weights_inline[1] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + (W)0.5);
weights_{j}[0] = (W)1.0 - wx;
weights_{j}[1] = wx;
'''

spline_weights_inline[2] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + (W)0.5);
weights_{j}[1] = (W)0.75 - wx * wx;
wy = (W)0.5 - wx;
weights_{j}[0] = (W)0.5 * wy * wy;
weights_{j}[2] = (W)1.0 - weights_{j}[0] - weights_{j}[1];
'''

spline_weights_inline[3] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + (W)0.5);
wy = (W)1.0 - wx;
weights_{j}[1] = (wx * wx * (wx - (W)2.0) * (W)3.0 + (W)4.0) / (W)6.0;
weights_{j}[2] = (wy * wy * (wy - (W)2.0) * (W)3.0 + (W)4.0) / (W)6.0;
weights_{j}[0] = wy * wy * wy / (W)6.0;
weights_{j}[3] = (W)1.0 - weights_{j}[0] - weights_{j}[1] - weights_{j}[2];
'''

spline_weights_inline[4] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + 0.5);
wy = wx * wx;
weights_{j}[2] = wy * (wy * (W)0.25 - (W)0.625) + (W)115.0 / (W)192.0;
wy = (W)1.0 + wx;
weights_{j}[1] =
    wy * (wy * (wy * ((W)5.0 - wy) / (W)6.0 - (W)1.25) + (W)5.0 / (W)24.0) +
    (W)55.0 / (W)96.0;
wy = (W)1.0 - wx;
weights_{j}[3] =
    wy * (wy * (wy * ((W)5.0 - wy) / (W)6.0 - (W)1.25) + (W)5.0 / (W)24.0) +
    (W)55.0 / (W)96.0;
wy = (W)0.5 - wx;
wy = wy * wy;
weights_{j}[0] = wy * wy / (W)24.0;
weights_{j}[4] = (W)1.0 - weights_{j}[0] - weights_{j}[1]
                        - weights_{j}[2] - weights_{j}[3];
'''

spline_weights_inline[5] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + (W)0.5);
wy = wx * wx;
weights_{j}[2] = wy * (wy * ((W)0.25 - wx / (W)12.0) - (W)0.5) + (W)0.55;
wy = (W)1.0 - wx;
wy = wy * wy;
weights_{j}[3] =
    wy * (wy * ((W)0.25 - ((W)1.0 - wx) / (W)12.0) - (W)0.5) + (W)0.55;
wy = wx + (W)1.0;
weights_{j}[1] =
    wy * (wy * (wy * (wy * (wy / (W)24.0 - (W)0.375) + (W)1.25) - (W)1.75)
   + (W)0.625) + (W)0.425;
wy = (W)2.0 - wx;
weights_{j}[4] =
    wy * (wy * (wy * (wy * (wy / (W)24.0 - (W)0.375) + (W)1.25) - (W)1.75)
    + (W)0.625) + (W)0.425;
wy = (W)1.0 - wx;
wy = wy * wy;
weights_{j}[0] = ((W)1.0 - wx) * wy * wy / 120.0;
weights_{j}[5] = (W)1.0 - weights_{j}[0] - weights_{j}[1] - weights_{j}[2]
                        - weights_{j}[3] - weights_{j}[4];
'''
