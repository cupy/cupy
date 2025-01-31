# distutils: language = c++


cpdef create_bbox_slice_tuple(coord_index_type[:, ::1] bbox,
                              coord_index_type max_int):
    """Support function for cupyx.scipy.ndimage.find_objects

    Creates a list of slice tuples from an array of bounding box coordinates.
    """
    cdef list slices = []
    cdef list slices_inner = []
    cdef coord_index_type n = bbox.shape[0]
    cdef coord_index_type i = 0
    cdef int ndim = bbox.shape[1] // 2

    for i in range(n):
        if bbox[i, 0] == max_int:
            # A value of max_int for the starting coordinate means label `i`
            # was not present in the image. We store None in this case.
            slices.append(None)
        else:
            # Append a slice tuple corresponding to the bounding box
            slices_inner = []
            for d in range(ndim):
                slices_inner.append(slice(bbox[i, 2*d], bbox[i, 2*d + 1]))
            slices.append(tuple(slices_inner))

    return slices
