# distutils: language = c++

"""Wrapper of CUB functions for CuPy API."""

from cpython cimport sequence
from libcpp.map cimport map as cpp_map 
from libcpp.string cimport string as cpp_string

from cupy_backends.cuda.api.driver cimport Stream as Stream_t
from cupy_backends.cuda.api cimport runtime
from cupy.core.core cimport _internal_ascontiguousarray
from cupy.core.core cimport _internal_asfortranarray
from cupy.core.internal cimport _contig_axes
from cupy.cuda cimport common
from cupy.cuda cimport device
from cupy.cuda cimport memory
from cupy.cuda cimport stream


###############################################################################
# Extern
###############################################################################

cdef extern from '../core/include/cupy/jitify/jitify.hpp' namespace "jitify::detail" nogil:
    cpp_map[cpp_string, cpp_string]& get_jitsafe_headers_map()
