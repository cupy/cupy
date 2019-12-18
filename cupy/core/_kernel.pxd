cpdef create_ufunc(name, ops, routine=*, preamble=*, doc=*,
<<<<<<< HEAD
                   default_casting=*, loop_prep=*)
cpdef create_reduction_func(name, ops, routine=*, identity=*, preamble=*)
=======
                   default_casting=*, loop_prep=*, out_ops=*)

cpdef tuple _get_args_info(list args)

cpdef str _get_kernel_params(tuple params, tuple args_info)

cdef tuple _broadcast(list args, tuple params, bint use_size)

cdef list _get_out_args(list out_args, tuple out_types, tuple out_shape,
                        casting)

cdef list _get_out_args_with_params(
    list out_args, tuple out_types, tuple out_shape, tuple out_params,
    bint is_size_specified)

cdef tuple _guess_routine_from_dtype(list ops, object dtype)

cdef tuple _guess_routine(
    str name, dict cache, list ops, list in_args, dtype, list out_ops)

cdef _check_array_device_id(ndarray arr, int device_id)

cdef list _preprocess_args(int dev_id, args, bint use_c_scalar)

cdef tuple _reduce_dims(list args, tuple params, tuple shape)
>>>>>>> b544536a3... Merge pull request #2076 from okuta/fix-true-divide
