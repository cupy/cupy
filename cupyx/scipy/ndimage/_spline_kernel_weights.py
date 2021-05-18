"""Determination of spline kernel weights (adapted from SciPy)

See more verbose comments for each case there:
https://github.com/scipy/scipy/blob/eba29d69846ab1299976ff4af71c106188397ccc/scipy/ndimage/src/ni_splines.c#L7   # NOQA

``spline_weights_inline`` is a dict where the key is the spline order and the
value is the spline weight initialization code.
"""

spline_weights_inline = {}

# Note: This order = 1 case is currently unused (order = 1 has a different code
# path in _interp_kernels.py). I think that existing code is a bit more
# efficient.
spline_weights_inline[1] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + 0.5);
weights_{j}[0] = 1.0 - wx;
weights_{j}[1] = wx;
'''

spline_weights_inline[2] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + 0.5);
weights_{j}[1] = 0.75 - wx * wx;
wy = 0.5 - wx;
weights_{j}[0] = 0.5 * wy * wy;
weights_{j}[2] = 1.0 - weights_{j}[0] - weights_{j}[1];
'''

spline_weights_inline[3] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + 0.5);
wy = 1.0 - wx;
weights_{j}[1] = (wx * wx * (wx - 2.0) * 3.0 + 4.0) / 6.0;
weights_{j}[2] = (wy * wy * (wy - 2.0) * 3.0 + 4.0) / 6.0;
weights_{j}[0] = wy * wy * wy / 6.0;
weights_{j}[3] = 1.0 - weights_{j}[0] - weights_{j}[1] - weights_{j}[2];
'''

spline_weights_inline[4] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + 0.5);
wy = wx * wx;
weights_{j}[2] = wy * (wy * 0.25 - 0.625) + 115.0 / 192.0;
wy = 1.0 + wx;
weights_{j}[1] = wy * (wy * (wy * (5.0 - wy) / 6.0 - 1.25) + 5.0 / 24.0) +
             55.0 / 96.0;
wy = 1.0 - wx;
weights_{j}[3] = wy * (wy * (wy * (5.0 - wy) / 6.0 - 1.25) + 5.0 / 24.0) +
             55.0 / 96.0;
wy = 0.5 - wx;
wy = wy * wy;
weights_{j}[0] = wy * wy / 24.0;
weights_{j}[4] = 1.0 - weights_{j}[0] - weights_{j}[1]
                     - weights_{j}[2] - weights_{j}[3];
'''

spline_weights_inline[5] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + 0.5);
wy = wx * wx;
weights_{j}[2] = wy * (wy * (0.25 - wx / 12.0) - 0.5) + 0.55;
wy = 1.0 - wx;
wy = wy * wy;
weights_{j}[3] = wy * (wy * (0.25 - (1.0 - wx) / 12.0) - 0.5) + 0.55;
wy = wx + 1.0;
weights_{j}[1] = wy * (wy * (wy * (wy * (wy / 24.0 - 0.375) + 1.25) - 1.75)
                   + 0.625) + 0.425;
wy = 2.0 - wx;
weights_{j}[4] = wy * (wy * (wy * (wy * (wy / 24.0 - 0.375) + 1.25) - 1.75)
                  + 0.625) + 0.425;
wy = 1.0 - wx;
wy = wy * wy;
weights_{j}[0] = (1.0 - wx) * wy * wy / 120.0;
weights_{j}[5] = 1.0 - weights_{j}[0] - weights_{j}[1] - weights_{j}[2]
                     - weights_{j}[3] - weights_{j}[4];
'''
