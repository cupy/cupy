from libcpp cimport bool
from libcpp.functional cimport function
from libcpp.string cimport string
from libcpp.vector cimport vector

from cupy.core.dlpack cimport DLTensor


cdef extern from "<chrono>" namespace "std::chrono::high_resolution_clock":

    cppclass duration:
        ctypedef rep '_Rep'
        rep count()
        pass


cdef extern from "tc/core/execution_engine.h" namespace "tc":

    cdef cppclass ExecutionEngine:

        ExecutionEngine() except +
        void define(const string& language)
        size_t compile(
            const string& name,
            const vector[const DLTensor*]& inputs,
            const MappingOptions& options)
        vector[const DLTensor*] inferOutputTensorInfo(
            const string& name,
            const vector[const DLTensor*]& inTensorPtrs)
        duration run(
            size_t handle,
            const vector[const DLTensor*]& inputs,
            const vector[DLTensor*]& outputs,
            bool profile)


cdef extern from "tc/core/mapping_options.h" namespace "tc":

    cdef cppclass MappingOptions:

        @staticmethod
        MappingOptions makeNaiveMappingOptions()
