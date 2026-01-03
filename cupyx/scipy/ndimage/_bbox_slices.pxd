from cython cimport fused_type


ctypedef fused coord_index_type:
    unsigned int
    unsigned long long

cpdef create_bbox_slice_tuple(coord_index_type[:, ::1] bbox,
                              coord_index_type max_int)
