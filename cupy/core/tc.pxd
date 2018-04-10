from libcpp.string cimport string
from libcpp.vector cimport vector

from cupy.core.dlpack cimport DLTensor


cdef extern from 'tc/core/execution_engine.h' namespace 'tc':

    cdef cppclass ExecutionEngine:

        ExecutionEngine() except +
        void define(const string& language)
        size_t compile(
            const string& name,
            const vector[const DLTensor*]& inputs,
            const MappingOptions& options)


cdef extern from 'tc/core/mapping_options.h' namespace 'tc':

    cdef cppclass MappingOptions:

        @staticmethod
        MappingOptions makeNaiveMappingOptions()
