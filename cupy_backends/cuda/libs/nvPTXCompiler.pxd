from libc.stdint cimport intptr_t


# Currently CUDA Python does not provide wrapper for nvPTXCompiler.

###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef void* Handle 'nvPTXCompilerHandle'
    ctypedef int Result 'nvPTXCompileResult'


###############################################################################
# APIs
###############################################################################

cpdef tuple getVersion()
cpdef intptr_t create(str src) except? 0
cpdef destroy(intptr_t compiler)
cpdef compile(intptr_t compiler, options)
cpdef bytes getCompiledProgram(intptr_t compiler)
cpdef str getErrorLog(intptr_t compiler)
cpdef str getInfoLog(intptr_t compiler)
