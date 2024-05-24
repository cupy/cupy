from typing import NamedTuple

import numpy

import cupy
import math
from cupy import _core


def delete(arr, indices, axis=None):
    """
    Delete values from an array along the specified axis.

    Args:
        arr (cupy.ndarray):
            Values are deleted from a copy of this array.
        indices (slice, int or array of ints):
            These indices correspond to values that will be deleted from the
            copy of `arr`.
            Boolean indices are treated as a mask of elements to remove.
        axis (int or None):
            The axis along which `indices` correspond to values that will be
            deleted. If `axis` is not given, `arr` will be flattened.

    Returns:
        cupy.ndarray:
            A copy of `arr` with values specified by `indices` deleted along
            `axis`.

    .. warning:: This function may synchronize the device.

    .. seealso:: :func:`numpy.delete`.
    """

    if axis is None:

        arr = arr.ravel()

        if isinstance(indices, cupy.ndarray) and indices.dtype == cupy.bool_:
            return arr[~indices]

        mask = cupy.ones(arr.size, dtype=bool)
        mask[indices] = False
        return arr[mask]

    else:

        if isinstance(indices, cupy.ndarray) and indices.dtype == cupy.bool_:
            return cupy.compress(~indices, arr, axis=axis)

        mask = cupy.ones(arr.shape[axis], dtype=bool)
        mask[indices] = False
        return cupy.compress(mask, arr, axis=axis)


# TODO(okuta): Implement insert


def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Args:
        arr (array_like):
            Values are appended to a copy of this array.
        values (array_like):
            These values are appended to a copy of ``arr``.  It must be of the
            correct shape (the same shape as ``arr``, excluding ``axis``).  If
            ``axis`` is not specified, ``values`` can be any shape and will be
            flattened before use.
        axis (int or None):
            The axis along which ``values`` are appended.  If ``axis`` is not
            given, both ``arr`` and ``values`` are flattened before use.

    Returns:
        cupy.ndarray:
            A copy of ``arr`` with ``values`` appended to ``axis``.  Note that
            ``append`` does not occur in-place: a new array is allocated and
            filled.  If ``axis`` is None, ``out`` is a flattened array.

    .. seealso:: :func:`numpy.append`
    """
    # TODO(asi1024): Implement fast path for scalar inputs.
    arr = cupy.asarray(arr)
    values = cupy.asarray(values)
    if axis is None:
        return _core.concatenate_method(
            (arr.ravel(), values.ravel()), 0).ravel()
    return _core.concatenate_method((arr, values), axis)


_resize_kernel = _core.ElementwiseKernel(
    'raw T x, int64 size', 'T y',
    'y = x[i % size]',
    'cupy_resize',
)


def resize(a, new_shape):
    """Return a new array with the specified shape.

    If the new array is larger than the original array, then the new
    array is filled with repeated copies of ``a``.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of ``a``.

    Args:
        a (array_like): Array to be resized.
        new_shape (int or tuple of int): Shape of resized array.

    Returns:
        cupy.ndarray:
            The new array is formed from the data in the old array, repeated
            if necessary to fill out the required number of elements.  The
            data are repeated in the order that they are stored in memory.

    .. seealso:: :func:`numpy.resize`
    """
    if numpy.isscalar(a):
        return cupy.full(new_shape, a)
    a = cupy.asarray(a)
    if a.size == 0:
        return cupy.zeros(new_shape, dtype=a.dtype)
    out = cupy.empty(new_shape, a.dtype)
    _resize_kernel(a, a.size, out)
    return out


_first_nonzero_krnl = _core.ReductionKernel(
    'T data, int64 len',
    'int64 y',
    'data == T(0) ? len : _j',
    'min(a, b)',
    'y = a',
    'len',
    'first_nonzero'
)


def trim_zeros(filt, trim='fb'):
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Returns the trimmed array

    Args:
        filt(cupy.ndarray): Input array
        trim(str, optional):
            'fb' default option trims the array from both sides.
            'f' option trim zeros from front.
            'b' option trim zeros from back.

    Returns:
        cupy.ndarray: trimmed input

    .. seealso:: :func:`numpy.trim_zeros`

    """
    if filt.ndim == 0:
        return filt
    if filt.ndim > 1:
        raise NotImplementedError('Multi-dimensional trim is not supported')
    start = 0
    end = filt.size
    trim = trim.upper()
    if 'F' in trim:
        start = _first_nonzero_krnl(filt, filt.size).item()
    if 'B' in trim:
        end = filt.size - _first_nonzero_krnl(filt[::-1], filt.size).item()
    return filt[start:end]


@_core.fusion.fuse()
def _unique_update_mask_equal_nan(mask, x0):
    mask1 = cupy.logical_not(cupy.isnan(x0))
    mask[:] = cupy.logical_and(mask, mask1)


_unique_nd_module = _core.RawModule(code='''
static const __device__ unsigned long long crc64_tab[256] = {
    0x0000000000000000, 0x7ad870c830358979,
    0xf5b0e190606b12f2, 0x8f689158505e9b8b,
    0xc038e5739841b68f, 0xbae095bba8743ff6,
    0x358804e3f82aa47d, 0x4f50742bc81f2d04,
    0xab28ecb46814fe75, 0xd1f09c7c5821770c,
    0x5e980d24087fec87, 0x24407dec384a65fe,
    0x6b1009c7f05548fa, 0x11c8790fc060c183,
    0x9ea0e857903e5a08, 0xe478989fa00bd371,
    0x7d08ff3b88be6f81, 0x07d08ff3b88be6f8,
    0x88b81eabe8d57d73, 0xf2606e63d8e0f40a,
    0xbd301a4810ffd90e, 0xc7e86a8020ca5077,
    0x4880fbd87094cbfc, 0x32588b1040a14285,
    0xd620138fe0aa91f4, 0xacf86347d09f188d,
    0x2390f21f80c18306, 0x594882d7b0f40a7f,
    0x1618f6fc78eb277b, 0x6cc0863448deae02,
    0xe3a8176c18803589, 0x997067a428b5bcf0,
    0xfa11fe77117cdf02, 0x80c98ebf2149567b,
    0x0fa11fe77117cdf0, 0x75796f2f41224489,
    0x3a291b04893d698d, 0x40f16bccb908e0f4,
    0xcf99fa94e9567b7f, 0xb5418a5cd963f206,
    0x513912c379682177, 0x2be1620b495da80e,
    0xa489f35319033385, 0xde51839b2936bafc,
    0x9101f7b0e12997f8, 0xebd98778d11c1e81,
    0x64b116208142850a, 0x1e6966e8b1770c73,
    0x8719014c99c2b083, 0xfdc17184a9f739fa,
    0x72a9e0dcf9a9a271, 0x08719014c99c2b08,
    0x4721e43f0183060c, 0x3df994f731b68f75,
    0xb29105af61e814fe, 0xc849756751dd9d87,
    0x2c31edf8f1d64ef6, 0x56e99d30c1e3c78f,
    0xd9810c6891bd5c04, 0xa3597ca0a188d57d,
    0xec09088b6997f879, 0x96d1784359a27100,
    0x19b9e91b09fcea8b, 0x636199d339c963f2,
    0xdf7adabd7a6e2d6f, 0xa5a2aa754a5ba416,
    0x2aca3b2d1a053f9d, 0x50124be52a30b6e4,
    0x1f423fcee22f9be0, 0x659a4f06d21a1299,
    0xeaf2de5e82448912, 0x902aae96b271006b,
    0x74523609127ad31a, 0x0e8a46c1224f5a63,
    0x81e2d7997211c1e8, 0xfb3aa75142244891,
    0xb46ad37a8a3b6595, 0xceb2a3b2ba0eecec,
    0x41da32eaea507767, 0x3b024222da65fe1e,
    0xa2722586f2d042ee, 0xd8aa554ec2e5cb97,
    0x57c2c41692bb501c, 0x2d1ab4dea28ed965,
    0x624ac0f56a91f461, 0x1892b03d5aa47d18,
    0x97fa21650afae693, 0xed2251ad3acf6fea,
    0x095ac9329ac4bc9b, 0x7382b9faaaf135e2,
    0xfcea28a2faafae69, 0x8632586aca9a2710,
    0xc9622c4102850a14, 0xb3ba5c8932b0836d,
    0x3cd2cdd162ee18e6, 0x460abd1952db919f,
    0x256b24ca6b12f26d, 0x5fb354025b277b14,
    0xd0dbc55a0b79e09f, 0xaa03b5923b4c69e6,
    0xe553c1b9f35344e2, 0x9f8bb171c366cd9b,
    0x10e3202993385610, 0x6a3b50e1a30ddf69,
    0x8e43c87e03060c18, 0xf49bb8b633338561,
    0x7bf329ee636d1eea, 0x012b592653589793,
    0x4e7b2d0d9b47ba97, 0x34a35dc5ab7233ee,
    0xbbcbcc9dfb2ca865, 0xc113bc55cb19211c,
    0x5863dbf1e3ac9dec, 0x22bbab39d3991495,
    0xadd33a6183c78f1e, 0xd70b4aa9b3f20667,
    0x985b3e827bed2b63, 0xe2834e4a4bd8a21a,
    0x6debdf121b863991, 0x1733afda2bb3b0e8,
    0xf34b37458bb86399, 0x8993478dbb8deae0,
    0x06fbd6d5ebd3716b, 0x7c23a61ddbe6f812,
    0x3373d23613f9d516, 0x49aba2fe23cc5c6f,
    0xc6c333a67392c7e4, 0xbc1b436e43a74e9d,
    0x95ac9329ac4bc9b5, 0xef74e3e19c7e40cc,
    0x601c72b9cc20db47, 0x1ac40271fc15523e,
    0x5594765a340a7f3a, 0x2f4c0692043ff643,
    0xa02497ca54616dc8, 0xdafce7026454e4b1,
    0x3e847f9dc45f37c0, 0x445c0f55f46abeb9,
    0xcb349e0da4342532, 0xb1eceec59401ac4b,
    0xfebc9aee5c1e814f, 0x8464ea266c2b0836,
    0x0b0c7b7e3c7593bd, 0x71d40bb60c401ac4,
    0xe8a46c1224f5a634, 0x927c1cda14c02f4d,
    0x1d148d82449eb4c6, 0x67ccfd4a74ab3dbf,
    0x289c8961bcb410bb, 0x5244f9a98c8199c2,
    0xdd2c68f1dcdf0249, 0xa7f41839ecea8b30,
    0x438c80a64ce15841, 0x3954f06e7cd4d138,
    0xb63c61362c8a4ab3, 0xcce411fe1cbfc3ca,
    0x83b465d5d4a0eece, 0xf96c151de49567b7,
    0x76048445b4cbfc3c, 0x0cdcf48d84fe7545,
    0x6fbd6d5ebd3716b7, 0x15651d968d029fce,
    0x9a0d8ccedd5c0445, 0xe0d5fc06ed698d3c,
    0xaf85882d2576a038, 0xd55df8e515432941,
    0x5a3569bd451db2ca, 0x20ed197575283bb3,
    0xc49581ead523e8c2, 0xbe4df122e51661bb,
    0x3125607ab548fa30, 0x4bfd10b2857d7349,
    0x04ad64994d625e4d, 0x7e7514517d57d734,
    0xf11d85092d094cbf, 0x8bc5f5c11d3cc5c6,
    0x12b5926535897936, 0x686de2ad05bcf04f,
    0xe70573f555e26bc4, 0x9ddd033d65d7e2bd,
    0xd28d7716adc8cfb9, 0xa85507de9dfd46c0,
    0x273d9686cda3dd4b, 0x5de5e64efd965432,
    0xb99d7ed15d9d8743, 0xc3450e196da80e3a,
    0x4c2d9f413df695b1, 0x36f5ef890dc31cc8,
    0x79a59ba2c5dc31cc, 0x037deb6af5e9b8b5,
    0x8c157a32a5b7233e, 0xf6cd0afa9582aa47,
    0x4ad64994d625e4da, 0x300e395ce6106da3,
    0xbf66a804b64ef628, 0xc5bed8cc867b7f51,
    0x8aeeace74e645255, 0xf036dc2f7e51db2c,
    0x7f5e4d772e0f40a7, 0x05863dbf1e3ac9de,
    0xe1fea520be311aaf, 0x9b26d5e88e0493d6,
    0x144e44b0de5a085d, 0x6e963478ee6f8124,
    0x21c640532670ac20, 0x5b1e309b16452559,
    0xd476a1c3461bbed2, 0xaeaed10b762e37ab,
    0x37deb6af5e9b8b5b, 0x4d06c6676eae0222,
    0xc26e573f3ef099a9, 0xb8b627f70ec510d0,
    0xf7e653dcc6da3dd4, 0x8d3e2314f6efb4ad,
    0x0256b24ca6b12f26, 0x788ec2849684a65f,
    0x9cf65a1b368f752e, 0xe62e2ad306bafc57,
    0x6946bb8b56e467dc, 0x139ecb4366d1eea5,
    0x5ccebf68aecec3a1, 0x2616cfa09efb4ad8,
    0xa97e5ef8cea5d153, 0xd3a62e30fe90582a,
    0xb0c7b7e3c7593bd8, 0xca1fc72bf76cb2a1,
    0x45775673a732292a, 0x3faf26bb9707a053,
    0x70ff52905f188d57, 0x0a2722586f2d042e,
    0x854fb3003f739fa5, 0xff97c3c80f4616dc,
    0x1bef5b57af4dc5ad, 0x61372b9f9f784cd4,
    0xee5fbac7cf26d75f, 0x9487ca0fff135e26,
    0xdbd7be24370c7322, 0xa10fceec0739fa5b,
    0x2e675fb4576761d0, 0x54bf2f7c6752e8a9,
    0xcdcf48d84fe75459, 0xb71738107fd2dd20,
    0x387fa9482f8c46ab, 0x42a7d9801fb9cfd2,
    0x0df7adabd7a6e2d6, 0x772fdd63e7936baf,
    0xf8474c3bb7cdf024, 0x829f3cf387f8795d,
    0x66e7a46c27f3aa2c, 0x1c3fd4a417c62355,
    0x935745fc4798b8de, 0xe98f353477ad31a7,
    0xa6df411fbfb21ca3, 0xdc0731d78f8795da,
    0x536fa08fdfd90e51, 0x29b7d047efec8728,
};

__device__ unsigned long long crc64(
    const unsigned long long s,
    unsigned long long crc
) {
    unsigned long long j;
    unsigned long long mask = 0xff;

    for (j = 0; j < 8; j++) {
        unsigned char byte = (s & mask) >> (8 * j);
        crc = crc64_tab[(unsigned char)crc ^ byte] ^ (crc >> 8);
        mask <<= 8;
    }
    return crc;
}

__global__ void map_crc64(
    double* x,
    unsigned long long* crc,
    int n
) {
    __shared__ unsigned long long block_crc[512];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n_threads = gridDim.x * blockDim.x;
    for(int row = idx / blockDim.x; row < n_rows; row += n_threads) {
        for(int pos = idx; pos < row_sz; pos += blockDim.x) {
            unsigned long long* x_u = reinterpret_cast<unsigned long long*>(x);
            unsigned long long x_crc = crc64(x_u[row_sz * row + pos], 0);
            block_crc[blockIdx.x] = x_crc;

            __syncthreads();

            for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                if (threadIdx.x < stride) {
                    unsigned long long this_crc = block_crc[threadIdx.x];
                    unsigned long long other_crc = block_crc[threadIdx.x + stride];
                    // int other_len = ((64 - __clzll(other_crc)) + 8 - 1) / 8;
                    block_crc[threadIdx.x] = crc64(other_crc, this_crc);
                }
                __syncthreads();
            }

            if(threadIdx.x == 0) {
                int cur_row_block = pos / blockDim.x;
                crc[blocks_per_row * row + cur_row_block] = block_crc[threadIdx.x];
            }
        }

    }
}

''', options=('-std=c++11',), name_expressions=['map_crc64'])  # NOQA


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None, *, equal_nan=True):
    """Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Args:
        ar(array_like): Input array. This will be flattened if it is not
            already 1-D.
        return_index(bool, optional): If True, also return the indices of `ar`
            (along the specified axis, if provided, or in the flattened array)
            that result in the unique array.
        return_inverse(bool, optional): If True, also return the indices of the
            unique array (for the specified axis, if provided) that can be used
            to reconstruct `ar`.
        return_counts(bool, optional): If True, also return the number of times
            each unique item appears in `ar`.
        axis(int or None, optional): The axis to operate on. If None, ar will
            be flattened. If an integer, the subarrays indexed by the given
            axis will be flattened and treated as the elements of a 1-D array
            with the dimension of the given axis, see the notes for more
            details. The default is None.
        equal_nan(bool, optional): If True, collapse multiple NaN values in the
            return array into one.

    Returns:
        cupy.ndarray or tuple:
            If there are no optional outputs, it returns the
            :class:`cupy.ndarray` of the sorted unique values. Otherwise, it
            returns the tuple which contains the sorted unique values and
            following.

            * The indices of the first occurrences of the unique values in the
              original array. Only provided if `return_index` is True.
            * The indices to reconstruct the original array from the
              unique array. Only provided if `return_inverse` is True.
            * The number of times each of the unique values comes up in the
              original array. Only provided if `return_counts` is True.

    Notes:
       When an axis is specified the subarrays indexed by the axis are sorted.
       This is done by making the specified axis the first dimension of the
       array (move the axis to the first dimension to keep the order of the
       other axes) and then flattening the subarrays in C order.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.unique`
    """
    if axis is None:
        ret = _unique_1d(ar, return_index=return_index,
                         return_inverse=return_inverse,
                         return_counts=return_counts,
                         equal_nan=equal_nan, inverse_shape=ar.shape)
        return ret

    ar = cupy.moveaxis(ar, axis, 0)

    # The array is reshaped into a contiguous 2D array
    orig_shape = ar.shape
    idx = cupy.arange(0, orig_shape[0], dtype=cupy.uint64)
    ar = ar.reshape(orig_shape[0], math.prod(orig_shape[1:]))
    ar = cupy.ascontiguousarray(ar)
    is_unsigned = cupy.issubdtype(ar.dtype, cupy.unsignedinteger)
    is_complex = cupy.iscomplexobj(ar)

    block_sz = 512
    n = ar.size
    n_blocks = (n + block_sz - 1) // block_sz

    breakpoint()
    crc = cupy.empty_like(ar, dtype=cupy.uint64)
    crc_comp = _unique_nd_module.get_function('map_crc64')
    crc_comp((n_blocks,), (block_sz,), (ar, crc, n))

    ar_cmp = ar
    if is_unsigned:
        ar_cmp = ar.astype(cupy.intp)

    def compare_axis_elems(idx1, idx2):
        left, right = ar_cmp[idx1], ar_cmp[idx2]
        comp = cupy.trim_zeros(left - right, 'f')
        if comp.shape[0] > 0:
            diff = comp[0]
            if is_complex and cupy.isnan(diff):
                return True
            return diff < 0
        return False

    # The array is sorted lexicographically using the first item of each
    # element on the axis
    sorted_indices = cupy.empty(orig_shape[0], dtype=cupy.intp)
    queue = [(idx.tolist(), 0)]
    while queue != []:
        current, off = queue.pop(0)
        if current == []:
            continue

        mid_elem = current[0]
        left = []
        right = []
        for i in range(1, len(current)):
            if compare_axis_elems(current[i], mid_elem):
                left.append(current[i])
            else:
                right.append(current[i])

        elem_pos = off + len(left)
        queue.append((left, off))
        queue.append((right, elem_pos + 1))

        sorted_indices[elem_pos] = mid_elem

    ar = ar[sorted_indices]

    if ar.size > 0:
        mask = cupy.empty(ar.shape, dtype=cupy.bool_)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]

        mask = cupy.any(mask, axis=1)
    else:
        # If empty, then the mask should grab the first empty array as the
        # unique one
        mask = cupy.ones((ar.shape[0]), dtype=cupy.bool_)
        mask[1:] = False

    # Index the input array with the unique elements and reshape it into the
    # original size and dimension order
    ar = ar[mask]
    ar = ar.reshape(mask.sum().item(), *orig_shape[1:])
    ar = cupy.moveaxis(ar, 0, axis)

    ret = ar,
    if return_index:
        ret += sorted_indices[mask],
    if return_inverse:
        imask = cupy.cumsum(mask) - 1
        inv_idx = cupy.empty(mask.shape, dtype=cupy.intp)
        inv_idx[sorted_indices] = imask
        ret += inv_idx,
    if return_counts:
        nonzero = cupy.nonzero(mask)[0]  # may synchronize
        idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],

    if len(ret) == 1:
        ret = ret[0]
    return ret


def _unique_1d(ar, return_index=False, return_inverse=False,
               return_counts=False, equal_nan=True, inverse_shape=None):
    ar = cupy.asarray(ar).flatten()

    if return_index or return_inverse:
        perm = ar.argsort()
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = cupy.empty(aux.shape, dtype=cupy.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    if equal_nan:
        _unique_update_mask_equal_nan(mask[1:], aux[:-1])

    ret = aux[mask]
    if not return_index and not return_inverse and not return_counts:
        return ret

    ret = ret,
    if return_index:
        ret += perm[mask],
    if return_inverse:
        imask = cupy.cumsum(mask) - 1
        inv_idx = cupy.empty(mask.shape, dtype=cupy.intp)
        inv_idx[perm] = imask
        ret += inv_idx.reshape(inverse_shape),
    if return_counts:
        nonzero = cupy.nonzero(mask)[0]  # may synchronize
        idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
        idx[:-1] = nonzero
        idx[-1] = mask.size
        ret += idx[1:] - idx[:-1],
    return ret


# Array API compatible unique_XXX wrappers

class UniqueAllResult(NamedTuple):
    values: cupy.ndarray
    indices: cupy.ndarray
    inverse_indices: cupy.ndarray
    counts: cupy.ndarray


class UniqueCountsResult(NamedTuple):
    values: cupy.ndarray
    counts: cupy.ndarray


class UniqueInverseResult(NamedTuple):
    values: cupy.ndarray
    inverse_indices: cupy.ndarray


def unique_all(x):
    """
    Find the unique elements of an array, and counts, inverse and indices.

    This function is an Array API compatible alternative to:

    >>> x = cupy.array([1, 1, 2])
    >>> np.unique(x, return_index=True, return_inverse=True,
    ...           return_counts=True, equal_nan=False)
    (array([1, 2]), array([0, 2]), array([0, 0, 1]), array([2, 1]))

    Parameters
    ----------
    x : ndarray
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * indices - The first occurring indices for each unique element.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.
    numpy.unique_all

    """
    result = unique(
        x,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        equal_nan=False
    )
    return UniqueAllResult(*result)


def unique_counts(x):
    """
    Find the unique elements and counts of an input array `x`.

    This function is an Array API compatible alternative to:

    >>> x = cupy.array([1, 1, 2])
    >>> cupy.unique(x, return_counts=True, equal_nan=False)
    (array([1, 2]), array([2, 1]))

    Parameters
    ----------
    x : ndarray
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.
    np.unique_counts

    """
    result = unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        equal_nan=False
    )
    return UniqueCountsResult(*result)


def unique_inverse(x):
    """
    Find the unique elements of `x` and indices to reconstruct `x`.

    This function is Array API compatible alternative to:

    >>> x = cupy.array([1, 1, 2])
    >>> cupy.unique(x, return_inverse=True, equal_nan=False)
    (array([1, 2]), array([0, 0, 1]))

    Parameters
    ----------
    x : ndarray
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.

    See Also
    --------
    unique : Find the unique elements of an array.
    numpy.unique_inverse

    """
    result = unique(
        x,
        return_index=False,
        return_inverse=True,
        return_counts=False,
        equal_nan=False
    )
    return UniqueInverseResult(*result)


def unique_values(x):
    """
    Returns the unique elements of an input array `x`.

    This function is Array API compatible alternative to:

    >>> x = cupy.array([1, 1, 2])
    >>> cupy.unique(x, equal_nan=False)
    array([1, 2])

    Parameters
    ----------
    x : ndarray
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : ndarray
        The unique elements of an input array.

    See Also
    --------
    unique : Find the unique elements of an array.
    numpy.unique_values

    """
    return unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        equal_nan=False
    )
