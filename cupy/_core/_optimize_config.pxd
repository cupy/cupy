cdef object _thread_local
cdef dict _contexts


cdef class _OptimizationConfig:

    cdef readonly object optimize_impl
    cdef readonly int max_trials
    cdef readonly float timeout
    cdef readonly float expected_total_time_per_trial
    cdef readonly float max_total_time_per_trial


cdef class _OptimizationContext:

    cdef readonly str key
    cdef readonly _OptimizationConfig config
    cdef readonly dict _params_map
    cdef readonly bint _dirty


cpdef _OptimizationContext get_current_context()
