"""
This file holds the pre-computed coefficients for determining if convolve
should use the 'direct' or 'fft' method based on the mode, data type, and size
of inputs.

This file is also runnable to compute the coefficients for a specific machine.
Those values can then be used as a replacement for these values.
"""

import argparse
import pprint
import warnings
import sys

import numpy

import cupy
from cupy import random
from cupyx import time
from cupyx.scipy import signal

# The precomputed coefficients
# This constant is the only thing used outside of this file
# If there are custom constants for a particular machine then this can be
# replaced with those values.
COEFFICIENTS = {
    'full': {'bool': [(-0.005118985557622385, 27885.21567403135, 394.25),
                      (-0.004452489945339739, 9605.700566858779, 72.0),
                      (-0.003898356856675272, 4657.869840730239, 32.0)],
             'complex128': [(-0.003602809500909493, 6187.85305144547, 318.25),
                            (-0.003209823674246226, 6973.818404462925, 180.0),
                            (-0.0030106957937321766,
                             5795.845043649038,
                             114.33333333333333)],
             'complex64': [(-0.0037338665587297493, 8833.48949058245, 266.95),
                           (-0.0037043258926359097, 9799.516454979803, 144.5),
                           (-0.0032905066460946473, 7811.0928813428445, 96.0)],
             'float16': [(-0.004314390177520294,
                          16339.894048394941,
                          297.34999999999997),
                         (-0.004243150989361764, 7714.536979921651, 60.0),
                         (-0.0038036263701403186,
                          4030.0223577160505,
                          26.666666666666668)],
             'float32': [(-0.003799250541753231, 14687.776217055893, 301.15),
                         (-0.004192740672292496, 8011.580563897867, 66.0),
                         (-0.00397393617749687,
                          4553.345698848279,
                          26.666666666666668)],
             'float64': [(-0.0035008434522388575, 11466.715181891112, 318.25),
                         (-0.004041007694510236, 7187.028408097934, 72.0),
                         (-0.003699600857239047, 3894.8918320109656, 32.0)],
             'int16': [(5.784231210957058e-05, 1093.53741365357, 8879.65),
                       (0, 0, 5928),
                       (-8.497678593610897e-05, 2616.551851457351, 1200.0)],
             'int32': [(2.9908786537706238e-05, 1194.2948835582754, 8746.65),
                       (-0.00010582559235666762, 3383.6839047022354, 2964.0),
                       (0, 0, 4096)],
             'int64': [(0.00014101128862835992, 538.9324381850201, 7015.75),
                       (0, 0, 5700),
                       (0, 0, 3600)],
             'int8': [(-0.00014901548152621371, 1037.1698762487006, 8982.25),
                      (0, 0, 5776),
                      (0, 0, 3600)],
             'uint16': [(-0.0003083396646168873, 1091.9947936569852, 9381.25),
                        (-5.679393783846121e-05, 3151.1522854024042, 2964.0),
                        (0, 0, 3600)],
             'uint32': [(8.915081092888521e-05, 898.0307597479647, 8879.65),
                        (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                        (0, 0, 4096)],
             'uint64': [(8.485312683575943e-05, 577.5368821825002, 7015.75),
                        (3.530345661632985e-05, 2800.293336955581, 2812.5),
                        (0, 0, 3600)],
             'uint8': [(-0.00021679223720785969, 824.3612286238374, 9774.55),
                       (-2.847070631360968e-05, 3056.3601984687307, 2964.0),
                       (0, 0, 3600)]},
    'same': {'bool': [(-0.0011570899247636062, 7376.651097949346, 12430.75),
                      (0.0003559223265081784, 3810.325048656898, 3612.5),
                      (0.0003596936679259876, 3963.911275395881, 1734.0)],
             'complex128': [(0.002556127733277625, 162.16416260697537, 3883.6),
                            (0.0009434970081089001, 1792.932411526184, 2112.0),
                            (0.001091373788822928,
                             2423.384257559164,
                             1637.6666666666667)],
             'complex64': [(0.0013722426552799439,
                            368.03026291874875,
                            5431.15),
                           (0.0005991144669265609, 2775.4912712600835, 2850.0),
                           (0.0007184327632755326, 3629.206822127669, 1944.0)],
             'float16': [(0.0007066899032049229,
                          519.6506868104207,
                          8465.449999999999),
                         (0.0001843369513800664, 3054.412626313679, 2964.0),
                         (9.666587710865876e-05, 2976.9369477429295, 1536.0)],
             'float32': [(0.00042064034214836, 678.1305755076914, 8338.15),
                         (0.00029615817359235217, 2949.0213027315, 2700.0),
                         (0, 0, 4608)],
             'float64': [(0.0011346663369928285, 741.9507892295803, 6324.15),
                         (0.00046845973790037737, 2667.4696399284617, 2145.0),
                         (0.00036495482316214046,
                          2525.193341485625,
                          1365.3333333333333)],
             'int16': [(-0.0001697668145829692, 1212.5142070179993, 8879.65),
                       (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                       (0, 0, 4608)],
             'int32': [(4.610913044911275e-05, 931.9155241415308, 8879.65),
                       (-8.789808287278889e-05, 3301.173044130459, 2964.0),
                       (0, 0, 4608)],
             'int64': [(0.00018257046912303229, 535.880924771625, 7036.65),
                       (0, 0, 5625),
                       (0, 0, 4096)],
             'int8': [(-6.19382100833658e-05, 1016.4964580405341, 8879.65),
                      (-6.713335411064331e-05, 3215.075626484873, 2964.0),
                      (0, 0, 4608)],
             'uint16': [(-0.00011897071056083923, 1154.0357434775453, 8879.65),
                        (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                        (0, 0, 4608)],
             'uint32': [(4.610913044911275e-05, 931.9155241415308, 8879.65),
                        (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                        (0, 0, 4608)],
             'uint64': [(0.000134645293377509, 572.6296620854185, 7036.65),
                        (0, 0, 5625),
                        (0, 0, 4096)],
             'uint8': [(-3.1356842649455525e-05, 796.2907906811608, 9659.6),
                       (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                       (0, 0, 4608)]},
    'valid': {'bool': [(-0.005502753469179086,
                        40434.47343401687,
                        378.09999999999997),
                       (-0.003960125590929909, 13582.065764902447, 72.0),
                       (-0.003293934131645603,
                        7855.550474991797,
                        41.666666666666664)],
              'complex128': [(-0.003146467408357361,
                              5504.248179412682,
                              301.15),
                             (-0.0024339899432850483,
                              5876.209968033281,
                              162.0),
                             (-0.0021623634981552814,
                              5848.236419961045,
                              130.66666666666666)],
              'complex64': [(-0.003167741951223018, 7800.274091942971, 238.45),
                            (-0.0027504177695792486, 8925.383285809932, 144.5),
                            (-0.002210452092454292,
                             8299.335941468986,
                             114.33333333333333)],
              'float16': [(-0.003729125140130681,
                           13838.404237838691,
                           297.34999999999997),
                          (-0.0037171551392420274, 9776.632434829238, 60.0),
                          (-0.0031963970339022353, 6906.704731808321, 32.0)],
              'float32': [(-0.00341510147973399, 13314.933919265311, 318.25),
                          (-0.0036076604178616276, 9527.946239209863, 66.0),
                          (-0.0029145625168262622,
                           6422.563736642928,
                           37.333333333333336)],
              'float64': [(-0.0030489945369032607, 10681.174085831486, 318.25),
                          (-0.0031997200240619834, 7707.611435559373, 72.0),
                          (-0.0030017238384078358,
                           5486.760979885637,
                           37.333333333333336)],
              'int16': [(1.650392939403755e-06, 640.9470256866944, 8640.25),
                        (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                        (0, 0, 4608)],
              'int32': [(3.978240775679074e-05, 683.8367066595004, 7671.25),
                        (-2.237912616393704e-05, 3255.678294026991, 2555.0),
                        (0, 0, 4608)],
              'int64': [(0.0003127423930601318, 392.35202561743654, 5596.45),
                        (1.898198917451951e-05, 2464.754356442952, 2450.0),
                        (0, 0, 4050)],
              'int8': [(-0.00040432501162083404, 1002.6889963085307, 8640.25),
                       (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                       (0, 0, 4608)],
              'uint16': [(0.00012809002975891135, 558.4640366416234, 8640.25),
                         (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                         (0, 0, 4608)],
              'uint32': [(0.00016935736770611317, 774.215928756722, 7453.7),
                         (4.6748057310157264e-05, 3098.1306857800146, 2555.0),
                         (0, 0, 4608)],
              'uint64': [(0.00022883879157818533, 451.3749745008608, 5596.45),
                         (0, 0, 4900),
                         (0, 0, 3600)],
              'uint8': [(1.0041551310416064e-05, 951.2977545842091, 8879.65),
                        (-0.00010048346534433912, 3347.9910881364385, 2964.0),
                        (0, 0, 4608)]}}


# These are the sizes that are sampled when computing the coefficients
# The values are:
#   * size to test against (or None for 'symmetric')
#   * min expected size
#   * max expected size
SIZES = (
    # 1d
    ((None, 2000, 30000),
     (50000, 1500, 20000),
     (100000, 1000, 15000),
     (200000, 500, 15000),
     (500000, 250, 15000),
     (1000000, 100, 15000),
     (2000000, 50, 15000)),

    # 2d
    ((None, 25, 150),
     (150, 20, 150),
     (250, 15, 125),
     (375, 10, 125),
     (500, 8, 100),
     (1000, 4, 100),
     (1500, 2, 100)),

    # 3d
    ((None, 7, 30),
     (30, 6, 28),
     (50, 5, 26),
     (75, 3, 25),
     (100, 2, 25),
     (125, 1, 25)),

    # Higher dimensions look like 3d and take much more time
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute the coefficients to determine when to use 'fft' "
                    "or 'direct' method in convolve")
    parser.add_argument('--progress', '-p', action='store_true',
                        help="display computation progress and information")
    parser.add_argument('--raw', '-r', action='store_true',
                        help="also output the raw crossovers found; "
                             "this is useful to average many results")

    args = parser.parse_args()

    if args.raw:
        coeffs, raw = get_coefficients(args.progress, True)
    else:
        coeffs = get_coefficients(args.progress)

    if args.progress:
        print()
    pprint.pprint(coeffs)

    if args.raw:
        print()
        pprint.pprint(raw)


def get_coefficients(display=False, raw=False):
    # Does not take into account:
    #   * contiguous-ness of data
    #   * distribution of size among multidimensional shape
    #   * fft prime divisors

    # This works by using binary searches until we find the crossover
    # between 'direct' and 'fft' methods.

    # Disable the cupyx.time.repeat warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    results = {}
    if raw:
        raw_data = {}

    for mode in ('full', 'same', 'valid'):
        results[mode] = {}
        if raw:
            raw_data[mode] = {}
        for dtype in (cupy.int8, cupy.int16, cupy.int32, cupy.int64,
                      cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64,
                      cupy.float16, cupy.float32, cupy.float64,
                      cupy.complex64, cupy.complex128, cupy.bool_):
            dtype = cupy.dtype(dtype)
            results[mode][dtype.name] = []
            if raw:
                raw_data[mode][dtype.name] = []
            args = (dtype, mode)

            for ndim, sizes in enumerate(SIZES, 1):
                X, Y = [], []
                for x, min, max in sizes:
                    fixed = None if x is None else (x,)*ndim
                    y = cupy.core.internal.prod(
                        _get_crossover(ndim, min, max, fixed, *args))
                    X.append(y if x is None else x**ndim)
                    Y.append(y)

                try:
                    coeffs = _compute_coeffs(ndim, X, Y)
                except ValueError as ex:
                    print("Failed to compute the coefficients for {} {} {}".
                          format(ndim, X, Y), file=sys.stderr)
                    print(ex, file=sys.stderr)
                    coeffs = (0, 0, 0)

                results[mode][dtype.name].append(coeffs)
                if raw:
                    raw_data[mode][dtype.name].append((X, Y))

                if display:
                    print(mode, dtype.name, '{}d'.format(ndim),
                          "  y < {1:.0f} * exp({0:.3e}*sqrt(x)) + {2:.1f}".
                          format(*coeffs))
                    if raw:
                        print(X)
                        print(Y)

    return (results, raw_data) if raw else results


def _compute_coeffs(ndim, X, Y):
    X, Y = numpy.array(X), numpy.array(Y)
    if Y[0] == Y[-1]:
        return 0, 0, Y[0]
    c = Y.min()/ndim if ndim > 1 else 0.95*Y.min()
    a, b = numpy.polyfit(numpy.sqrt(X), numpy.log(Y-c), 1, w=numpy.sqrt(Y-c))
    return float(a), float(numpy.exp(b)), float(c)


def _get_crossover(ndim, min, max, fixed=None, *args):
    # Samples shapes between min and max (duplicated for each dimension) and
    # the other data is fixed to the shape given in fixed (or uses the same
    # shape if it isn't provided/None). For multidimensional runs, this then
    # uses a similar procedure but just for the final dimension.

    stopping_space = 4 if ndim == 1 else 0
    while min < max - stopping_space:
        middle = (min + max + 1) // 2
        shape = (middle,)*ndim
        if _is_direct_faster(shape, fixed or shape, *args):
            min = middle
        else:
            max = middle - 1
    if ndim == 1:
        return ((min+max+1)//2,)

    # Refine last dimension
    y, max = min, min + ndim + 1  # really high dimensions need more than +1
    shape_but_last = (y,)*(ndim-1)
    while min < max:
        middle = (min + max + 1) // 2
        shape = shape_but_last + (middle,)
        if _is_direct_faster(shape, fixed or shape, *args):
            min = middle
        else:
            max = middle - 1
    return shape_but_last + (min,)


def _is_direct_faster(shape1, shape2, dtype=float, mode='full'):

    # Allocate data
    dtype = cupy.dtype(dtype)
    if dtype.kind == 'b':
        in1 = random.random(shape1, cupy.float32) > 0.5
        in2 = random.random(shape2, cupy.float32) > 0.5
    elif dtype.kind == 'i':
        iinfo = cupy.iinfo(dtype)
        in1 = random.randint(iinfo.min, iinfo.max, shape1, dtype)
        in2 = random.randint(iinfo.min, iinfo.max, shape1, dtype)
    elif dtype.kind == 'u':
        iinfo = cupy.iinfo(dtype)
        in1 = random.randint(0, iinfo.max, shape1, dtype)
        in2 = random.randint(0, iinfo.max, shape1, dtype)
    elif dtype.kind == 'c':
        subdt = cupy.dtype('f{}'.format(dtype.itemsize//2))
        in1 = random.random(shape1, subdt) + random.random(shape1, subdt)*1j
        in2 = random.random(shape2, subdt) + random.random(shape2, subdt)*1j
    elif dtype.kind == 'f' and dtype.itemsize < 4:
        in1 = random.random(shape1, cupy.float32).astype(dtype)
        in2 = random.random(shape2, cupy.float32).astype(dtype)
    else:  # dtype.kind == 'f'
        in1 = random.random(shape1, dtype)
        in2 = random.random(shape2, dtype)

    # Get timing data
    dt = time.repeat(signal.convolve, (in1, in2, mode, 'direct'),
                     n_repeat=50, n_warmup=2)
    ft = time.repeat(signal.convolve, (in1, in2, mode, 'fft'),
                     n_repeat=50, n_warmup=2)
    dir_times = dt.gpu_times.mean(1)[0]  # TODO: or min(1)?
    fft_times = ft.gpu_times.mean(1)[0]

    # Return results
    return bool(dir_times < fft_times)


if __name__ == "__main__":
    main()
