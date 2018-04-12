from libcpp cimport bool
from libcpp.functional cimport function
from libcpp.string cimport string
from libcpp.vector cimport vector

from cupy.core.dlpack cimport DLTensor


cdef extern from "<chrono>" namespace "std::chrono::high_resolution_clock":

    cppclass duration:
        pass


cdef extern from "tc/core/execution_engine.h" namespace "tc::ExecutionEngine":

    cppclass ExecutorInfo:
        pass

ctypedef bool f(const ExecutorInfo*)
ctypedef function[f] prunning_ftype

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
            bool profile,
            prunning_ftype prunningFunction)


cdef extern from "tc/core/mapping_options.h" namespace "tc":

    cdef cppclass MappingOptions:

        MappingOptions(const string& str)

        @staticmethod
        MappingOptions makeNaiveMappingOptions()
        
        @staticmethod
        MappingOptions makeSingleThreadMappingOptions()
        
        @staticmethod
        MappingOptions makePointwiseMappingOptions()
        
        @staticmethod
        MappingOptions makeMlpMappingOptions()
        
        @staticmethod
        MappingOptions makeConvolutionMappingOptions()

        @staticmethod
        MappingOptions makeGroupConvolutionMappingOptions()
