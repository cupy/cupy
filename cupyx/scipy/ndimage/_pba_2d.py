import math
import numbers
import os

import cupy

from ._util import _get_inttype

if hasattr(math, 'lcm'):
    lcm = math.lcm
else:
    """Fallback implementation of least common multiple (lcm)"""
    def _lcm(a, b):
        return abs(b * (a // math.gcd(a, b)))

    def lcm(*integers):
        nargs = len(integers)
        if not all(isinstance(a, numbers.Integral) for a in integers):
            raise TypeError("all arguments must be integers")
        if nargs == 0:
            return 1
        res = int(integers[0])
        if nargs == 1:
            return abs(res)
        for i in range(1, nargs):
            x = int(integers[i])
            res = _lcm(res, x)
        return res


pba2d_defines_template = """

// MARKER is used to mark blank pixels in the texture.
// Any uncolored pixels will have x = MARKER.
// Input texture should have x = MARKER for all pixels other than sites
#define MARKER      {marker}
#define BLOCKSIZE   {block_size_2d}
#define pixel_int2_t {pixel_int2_t}                // typically short2 (int2 for images with > 32k pixels per side)
#define make_pixel(x, y)  {make_pixel_func}(x, y)  // typically make_short2 (make_int2 images with > 32k pixels per side

"""  # noqa


def _init_marker(int_dtype):
    """use a minimum value that is appropriate to the integer dtype"""
    if int_dtype == cupy.int16:
        # marker = cupy.iinfo(int_dtype).min
        marker = -32768
    elif int_dtype == cupy.int32:
        # divide by two so we don't have to promote other intermediate int
        # variables to 64-bit int
        marker = -2147483648 // 2
    else:
        raise ValueError(
            "expected int_dtype to be either cupy.int16 or cupy.int32"
        )
    return marker


@cupy.memoize(True)
def get_pba2d_src(block_size_2d=64, marker=-32768, pixel_int2_t="short2"):
    make_pixel_func = "make_" + pixel_int2_t

    pba2d_code = pba2d_defines_template.format(
        block_size_2d=block_size_2d,
        marker=marker,
        pixel_int2_t=pixel_int2_t,
        make_pixel_func=make_pixel_func
    )
    kernel_directory = os.path.join(os.path.dirname(__file__), "cuda")
    with open(os.path.join(kernel_directory, "pba_kernels_2d.h"), "rt") as f:
        pba2d_kernels = "\n".join(f.readlines())

    pba2d_code += pba2d_kernels
    return pba2d_code


def _get_block_size(check_warp_size=False):
    if check_warp_size:
        dev = cupy.cuda.runtime.getDevice()
        device_properties = cupy.cuda.runtime.getDeviceProperties(dev)
        return int(device_properties["warpSize"])
    else:
        return 32


@cupy.memoize(for_each_device=True)
def _get_pack_kernel(int_type, marker=-32768):
    """Pack coordinates into array of type short2 (or int2).

    This kernel works with 2D input data, `arr` (typically boolean).

    The output array, `out` will be 3D with a signed integer dtype.
    It will have size 2 on the last axis so that it can be viewed as a CUDA
    vector type such as `int2` or `float2`.
    """
    code = f"""
    if (arr[i]) {{
        out[2*i] = {marker};
        out[2*i + 1] = {marker};
    }} else {{
        int shape_1 = arr.shape()[1];
        int _i = i;
        int ind_1 = _i % shape_1;
        _i /= shape_1;
        out[2*i] = ind_1;   // out.x
        out[2*i + 1] = _i;  // out.y
    }}
    """
    return cupy.ElementwiseKernel(
        in_params="raw B arr",
        out_params="raw I out",
        operation=code,
        options=("--std=c++11",),
    )


def _pack_int2(arr, marker=-32768, int_dtype=cupy.int16):
    if arr.ndim != 2:
        raise ValueError("only 2d arr suppported")
    int2_dtype = cupy.dtype({"names": ["x", "y"], "formats": [int_dtype] * 2})
    out = cupy.zeros(arr.shape + (2,), dtype=int_dtype)
    assert out.size == 2 * arr.size
    pack_kernel = _get_pack_kernel(
        int_type="short" if int_dtype == cupy.int16 else "int",
        marker=marker
    )
    pack_kernel(arr, out, size=arr.size)
    out = cupy.squeeze(out.view(int2_dtype))
    return out


def _unpack_int2(img, make_copy=False, int_dtype=cupy.int16):
    temp = img.view(int_dtype).reshape(img.shape + (2,))
    if make_copy:
        temp = temp.copy()
    return temp


def _determine_padding(shape, padded_size, block_size):
    # all kernels assume equal size along both axes, so pad up to equal size if
    # shape is not isotropic
    orig_sy, orig_sx = shape
    if orig_sx != padded_size or orig_sy != padded_size:
        padding_width = (
            (0, padded_size - orig_sy), (0, padded_size - orig_sx)
        )
    else:
        padding_width = None
    return padding_width


def _generate_shape(ndim, int_type, var_name="out", raw_var=True):
    code = ""
    if not raw_var:
        var_name = "_raw_" + var_name
    for i in range(ndim):
        code += f"{int_type} shape_{i} = {var_name}.shape()[{i}];\n"
    return code


def _generate_indices_ops(ndim, int_type):
    code = f"{int_type} _i = i;\n"
    for j in range(ndim - 1, 0, -1):
        code += f"{int_type} ind_{j} = _i % shape_{j};\n_i /= shape_{j};\n"
    code += f"{int_type} ind_0 = _i;"
    return code


def _get_distance_kernel_code(int_type, dist_int_type, raw_out_var=True):
    code = _generate_shape(
        ndim=2, int_type=int_type, var_name="dist", raw_var=raw_out_var
    )
    code += _generate_indices_ops(ndim=2, int_type=int_type)
    code += f"""
    {int_type} tmp;
    {dist_int_type} sq_dist;
    tmp = y[i] - ind_0;
    sq_dist = tmp * tmp;
    tmp = x[i] - ind_1;
    sq_dist += tmp * tmp;
    dist[i] = sqrt(static_cast<F>(sq_dist));
    """
    return code


@cupy.memoize(for_each_device=True)
def _get_distance_kernel(int_type, dist_int_type):
    """Returns kernel computing the Euclidean distance from coordinates."""
    operation = _get_distance_kernel_code(
        int_type, dist_int_type, raw_out_var=True
    )
    return cupy.ElementwiseKernel(
        in_params="raw I y, raw I x",
        out_params="raw F dist",
        operation=operation,
        options=("--std=c++11",),
    )


def _get_aniso_distance_kernel_code(int_type, raw_out_var=True):
    code = _generate_shape(
        ndim=2, int_type=int_type, var_name="dist", raw_var=raw_out_var
    )
    code += _generate_indices_ops(ndim=2, int_type=int_type)
    code += """
    F tmp;
    F sq_dist;
    tmp = static_cast<F>(y[i] - ind_0) * sampling[0];
    sq_dist = tmp * tmp;
    tmp = static_cast<F>(x[i] - ind_1) * sampling[1];
    sq_dist += tmp * tmp;
    dist[i] = sqrt(sq_dist);
    """
    return code


@cupy.memoize(for_each_device=True)
def _get_aniso_distance_kernel(int_type):
    """Returns kernel computing the Euclidean distance from coordinates."""
    operation = _get_aniso_distance_kernel_code(int_type, raw_out_var=True)
    return cupy.ElementwiseKernel(
        in_params="raw I y, raw I x, raw F sampling",
        out_params="raw F dist",
        operation=operation,
        options=("--std=c++11",),
    )


def _distance_tranform_arg_check(distances_out, indices_out,
                                 return_distances, return_indices):
    """Raise a RuntimeError if the arguments are invalid"""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            "at least one of return_distances/return_indices must be True")
    if distances_out and not return_distances:
        error_msgs.append(
            "return_distances must be True if distances is supplied"
        )
    if indices_out and not return_indices:
        error_msgs.append("return_indices must be True if indices is supplied")
    if error_msgs:
        raise RuntimeError(", ".join(error_msgs))


def _check_distances(distances, shape, dtype):
    if distances.shape != shape:
        raise RuntimeError("distances array has wrong shape")
    if distances.dtype != dtype:
        raise RuntimeError(
            f"distances array must have dtype: {dtype}")


def _check_indices(indices, shape, itemsize):
    if indices.shape != shape:
        raise RuntimeError("indices array has wrong shape")
    if indices.dtype.kind not in 'iu':
        raise RuntimeError(
            "indices array must have an integer dtype"
        )
    elif indices.dtype.itemsize < itemsize:
        raise RuntimeError(
            f"indices dtype must have itemsize > {itemsize}"
        )


def _pba_2d(arr, sampling=None, return_distances=True, return_indices=False,
            block_params=None, check_warp_size=False, *,
            float64_distances=False, distances=None, indices=None):

    indices_inplace = isinstance(indices, cupy.ndarray)
    dt_inplace = isinstance(distances, cupy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, indices_inplace, return_distances, return_indices
    )

    # input_arr: a 2D image
    #    For each site at (x, y), the pixel at coordinate (x, y) should contain
    #    the pair (x, y). Pixels that are not sites should contain the pair
    #    (MARKER, MARKER)

    # Note: could query warp size here, but for now just assume 32 to avoid
    #       overhead of querying properties
    block_size = _get_block_size(check_warp_size)

    if block_params is None:
        padded_size = math.ceil(max(arr.shape) / block_size) * block_size

        # should be <= size / block_size. sy must be a multiple of m1
        m1 = padded_size // block_size
        # size must be a multiple of m2
        m2 = max(1, min(padded_size // block_size, block_size))
        # m2 must also be a power of two
        m2 = 2**math.floor(math.log2(m2))
        if padded_size % m2 != 0:
            raise RuntimeError("error in setting default m2")
        m3 = min(min(m1, m2), 2)
    else:
        if any(p < 1 for p in block_params):
            raise ValueError("(m1, m2, m3) in blockparams must be >= 1")
        m1, m2, m3 = block_params
        if math.log2(m2) % 1 > 1e-5:
            raise ValueError("m2 must be a power of 2")
        multiple = lcm(block_size, m1, m2, m3)
        padded_size = math.ceil(max(arr.shape) / multiple) * multiple

    if m1 > padded_size // block_size:
        raise ValueError(
            f"m1 too large. must be <= padded arr.shape[0] // {block_size}"
        )
    if m2 > padded_size // block_size:
        raise ValueError(
            f"m2 too large. must be <= padded arr.shape[1] // {block_size}"
        )
    if m3 > padded_size // block_size:
        raise ValueError(
            f"m3 too large. must be <= padded arr.shape[1] // {block_size}"
        )
    for m in (m1, m2, m3):
        if padded_size % m != 0:
            raise ValueError(
                f"Largest dimension of image ({padded_size}) must be evenly "
                f"disivible by each element of block_params: {(m1, m2, m3)}."
            )

    shape_max = max(arr.shape)
    if shape_max <= 32768:
        int_dtype = cupy.int16
        pixel_int2_type = "short2"
    else:
        if shape_max > (1 << 24):
            # limit to coordinate range to 2**24 due to use of __mul24 in
            # coordinate TOID macro
            raise ValueError(
                f"maximum axis size of {1 << 24} exceeded, for image with "
                f"shape {arr.shape}"
            )
        int_dtype = cupy.int32
        pixel_int2_type = "int2"

    marker = _init_marker(int_dtype)

    orig_sy, orig_sx = arr.shape
    padding_width = _determine_padding(arr.shape, padded_size, block_size)
    if padding_width is not None:
        arr = cupy.pad(arr, padding_width, mode="constant", constant_values=1)
    size = arr.shape[0]

    input_arr = _pack_int2(arr, marker=marker, int_dtype=int_dtype)
    output = cupy.zeros_like(input_arr)

    int2_dtype = cupy.dtype({"names": ["x", "y"], "formats": [int_dtype] * 2})
    margin = cupy.empty((2 * m1 * size,), dtype=int2_dtype)

    # phase 1 of PBA. m1 must divide texture size and be <= 64
    pba2d = cupy.RawModule(
        code=get_pba2d_src(
            block_size_2d=block_size,
            marker=marker,
            pixel_int2_t=pixel_int2_type,
        )
    )
    kernelFloodDown = pba2d.get_function("kernelFloodDown")
    kernelFloodUp = pba2d.get_function("kernelFloodUp")
    kernelPropagateInterband = pba2d.get_function("kernelPropagateInterband")
    kernelUpdateVertical = pba2d.get_function("kernelUpdateVertical")
    kernelCreateForwardPointers = pba2d.get_function(
        "kernelCreateForwardPointers"
    )
    kernelDoubleToSingleList = pba2d.get_function("kernelDoubleToSingleList")

    if sampling is None:
        kernelProximatePoints = pba2d.get_function("kernelProximatePoints")
        kernelMergeBands = pba2d.get_function("kernelMergeBands")
        kernelColor = pba2d.get_function("kernelColor")
    else:
        kernelProximatePoints = pba2d.get_function(
            "kernelProximatePointsWithSpacing"
        )
        kernelMergeBands = pba2d.get_function("kernelMergeBandsWithSpacing")
        kernelColor = pba2d.get_function("kernelColorWithSpacing")

    block = (block_size, 1, 1)
    grid = (math.ceil(size / block[0]), m1, 1)
    bandSize1 = size // m1
    # kernelFloodDown modifies input_arr in-place
    kernelFloodDown(
        grid,
        block,
        (input_arr, input_arr, size, bandSize1),
    )
    # kernelFloodUp modifies input_arr in-place
    kernelFloodUp(
        grid,
        block,
        (input_arr, input_arr, size, bandSize1),
    )
    # kernelFloodUp fills values into margin
    kernelPropagateInterband(
        grid,
        block,
        (input_arr, margin, size, bandSize1),
    )
    # kernelUpdateVertical stores output into an intermediate array of
    # transposed shape
    kernelUpdateVertical(
        grid,
        block,
        (input_arr, margin, output, size, bandSize1),
    )

    # phase 2
    block = (block_size, 1, 1)
    grid = (math.ceil(size / block[0]), m2, 1)
    bandSize2 = size // m2
    if sampling is None:
        sampling_args = ()
    else:
        # Originally the shape is (y, x) and sampling[1] corresponds to y.
        # However, kernelUpdateVertical transposed the image, so
        # we are now working with (x, y) instead. Need sampling ordered
        # accordingly.
        sampling = tuple(map(float, sampling))
        sampling_args = (sampling[0], sampling[1])
    kernelProximatePoints(
        grid,
        block,
        (output, input_arr, size, bandSize2) + sampling_args,
    )
    kernelCreateForwardPointers(
        grid,
        block,
        (input_arr, input_arr, size, bandSize2),
    )
    # Repeatly merging two bands into one
    noBand = m2
    while noBand > 1:
        grid = (math.ceil(size / block[0]), noBand // 2)
        kernelMergeBands(
            grid,
            block,
            (output, input_arr, input_arr, size, size // noBand) + sampling_args,  # noqa
        )
        noBand //= 2
    # Replace the forward link with the X coordinate of the seed to remove
    # the need of looking at the other texture. We need it for coloring.
    grid = (math.ceil(size / block[0]), size)
    kernelDoubleToSingleList(
        grid,
        block,
        (output, input_arr, input_arr, size),
    )

    # Phase 3 of PBA
    block = (block_size, m3, 1)
    grid = (math.ceil(size / block[0]), 1, 1)
    kernelColor(
        grid,
        block,
        (input_arr, output, size) + sampling_args,
    )

    output = _unpack_int2(output, make_copy=False, int_dtype=int_dtype)
    # make sure to crop any padding that was added here!
    x = output[:orig_sy, :orig_sx, 0]
    y = output[:orig_sy, :orig_sx, 1]

    vals = ()
    if return_distances:
        dtype_out = cupy.float64 if float64_distances else cupy.float32
        if dt_inplace:
            _check_distances(distances, y.shape, dtype_out)
        else:
            distances = cupy.zeros(y.shape, dtype=dtype_out)

        # make sure maximum possible distance doesn"t overflow
        max_possible_dist = sum((s - 1)**2 for s in y.shape)
        dist_int_type = "int" if max_possible_dist < 2**31 else "ptrdiff_t"

        if sampling is None:
            distance_kernel = _get_distance_kernel(
                int_type=_get_inttype(distances),
                dist_int_type=dist_int_type,
            )
            distance_kernel(y, x, distances, size=distances.size)
        else:
            distance_kernel = _get_aniso_distance_kernel(
                int_type=_get_inttype(distances),
            )
            sampling = cupy.asarray(sampling, dtype=dtype_out)
            distance_kernel(y, x, sampling, distances, size=distances.size)

        vals = vals + (distances,)
    if return_indices:
        if indices_inplace:
            _check_indices(indices, (arr.ndim,) + arr.shape, x.dtype.itemsize)
            indices[0, ...] = y
            indices[1, ...] = x
        else:
            indices = cupy.stack((y, x), axis=0)
        vals = vals + (indices,)
    return vals
