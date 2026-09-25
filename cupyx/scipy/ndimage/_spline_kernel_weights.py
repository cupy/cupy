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
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + ({F})0.5);
weights_{j}[0] = ({F})1.0 - wx;
weights_{j}[1] = wx;
'''

spline_weights_inline[2] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + ({F})0.5);
weights_{j}[1] = ({F})0.75 - wx * wx;
wy = ({F})0.5 - wx;
weights_{j}[0] = ({F})0.5 * wy * wy;
weights_{j}[2] = ({F})1.0 - weights_{j}[0] - weights_{j}[1];
'''

spline_weights_inline[3] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + ({F})0.5);
wy = ({F})1.0 - wx;
weights_{j}[1] = (wx * wx * (wx - ({F})2.0) * ({F})3.0 + ({F})4.0) / ({F})6.0;
weights_{j}[2] = (wy * wy * (wy - ({F})2.0) * ({F})3.0 + ({F})4.0) / ({F})6.0;
weights_{j}[0] = wy * wy * wy / ({F})6.0;
weights_{j}[3] = ({F})1.0 - weights_{j}[0] - weights_{j}[1] - weights_{j}[2];
'''

spline_weights_inline[4] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + ({F})0.5);
wy = wx * wx;
weights_{j}[2] = wy * (wy * ({F})0.25 - ({F})0.625) + ({F})115.0 / ({F})192.0;
wy = ({F})1.0 + wx;
weights_{j}[1] =
    wy * (wy * (wy * (({F})5.0 - wy) / ({F})6.0 - ({F})1.25)
    + ({F})5.0 / ({F})24.0) + ({F})55.0 / ({F})96.0;
wy = ({F})1.0 - wx;
weights_{j}[3] =
    wy * (wy * (wy * (({F})5.0 - wy) / ({F})6.0 - ({F})1.25)
    + ({F})5.0 / ({F})24.0) + ({F})55.0 / ({F})96.0;
wy = ({F})0.5 - wx;
wy = wy * wy;
weights_{j}[0] = wy * wy / ({F})24.0;
weights_{j}[4] = ({F})1.0 - weights_{j}[0] - weights_{j}[1]
                        - weights_{j}[2] - weights_{j}[3];
'''

spline_weights_inline[5] = '''
wx = c_{j} - floor({order} & 1 ? c_{j} : c_{j} + ({F})0.5);
wy = wx * wx;
weights_{j}[2] =
    wy * (wy * (({F})0.25 - wx / ({F})12.0) - ({F})0.5) + ({F})0.55;
wy = ({F})1.0 - wx;
wy = wy * wy;
weights_{j}[3] =
    wy * (wy * (({F})0.25 - (({F})1.0 - wx) / ({F})12.0) - ({F})0.5)
    + ({F})0.55;
wy = wx + ({F})1.0;
weights_{j}[1] =
    wy * (wy * (wy * (wy * (wy / ({F})24.0 - ({F})0.375) + ({F})1.25)
    - ({F})1.75) + ({F})0.625) + ({F})0.425;
wy = ({F})2.0 - wx;
weights_{j}[4] =
    wy * (wy * (wy * (wy * (wy / ({F})24.0 - ({F})0.375) + ({F})1.25)
    - ({F})1.75) + ({F})0.625) + ({F})0.425;
wy = ({F})1.0 - wx;
wy = wy * wy;
weights_{j}[0] = (({F})1.0 - wx) * wy * wy / ({F})120.0;
weights_{j}[5] = ({F})1.0 - weights_{j}[0] - weights_{j}[1] - weights_{j}[2]
                        - weights_{j}[3] - weights_{j}[4];
'''
