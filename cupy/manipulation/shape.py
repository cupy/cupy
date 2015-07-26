import collections

import numpy

from cupy import internal


def reshape(a, newshape):
    # TODO(beam2d): Support ordering option
    newshape = internal.infer_unknown_dimension(newshape, a.size)
    if len(newshape) == 1 and \
       isinstance(newshape[0], collections.Iterable):
        newshape = tuple(newshape[0])

    size = a.size
    if numpy.prod(newshape, dtype=int) != size:
        raise RuntimeError('Total size mismatch on reshape')

    newstrides = internal.get_strides_for_nocopy_reshape(a, newshape)
    if newstrides is not None:
        newarray = a.view()
    else:
        newarray = a.copy()
        newstrides = internal.get_strides_for_nocopy_reshape(
            newarray, newshape)
    newarray._shape = newshape
    newarray._strides = newstrides
    newarray._update_f_contiguity()
    return newarray


def ravel(a):
    # TODO(beam2d): Support ordering option
    return reshape(a, (a.size,))
