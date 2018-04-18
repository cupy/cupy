from libcpp cimport bool
from libcpp.functional cimport function
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t

from cupy.core.dlpack cimport DLTensor
from cupy.core.dlpack cimport DLManagedTensor


cdef extern from "<chrono>" namespace "std::chrono::high_resolution_clock":

    cdef cppclass duration:
        pass


cdef extern from "tc/core/execution_engine.h" namespace "tc::ExecutionEngine":

    cdef cppclass ExecutorInfo:
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


cdef extern from "mapping_options.pb.h" namespace "tc":

    cdef cppclass MappingOptionsProto:

        MappingOptionsProto() except +
        size_t ByteSizeLong()
        string SerializeAsString()
        bool IsInitialized()


cdef extern from "tc/core/mapping_options.h" namespace "tc":

    cdef cppclass MappingOptions:

        MappingOptions(const string& str)
        string toProtobufSerializedString()

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

        MappingOptionsProto proto


cdef extern from "llvm/ADT/Optional.h" namespace "llvm":

    cdef cppclass Optional[T]:
        
        T* getPointer()
        T& getValue()


cdef extern from "tc/autotuner/parameters.h" namespace "tc::autotune":

    cdef cppclass TuningParameterFixer:
        pass


ctypedef vector[const DLTensor*] ConstDLTensorVec
ctypedef vector[DLTensor*] DLTensorVec


cdef extern from "tc/autotuner/genetic_autotuner.h" namespace "tc::autotune::detail":

    cdef cppclass GeneticAutotuner:

        GeneticAutotuner(const string& tc) except +
        void storeCaches(const string& filename)

        vector[MappingOptions] load(
            const string& cacheFileName,
            const string& tcName,
            const vector[const DLTensor*]& inputs,
            const size_t numCandidates)

        Optional[MappingOptions] tune(
            const string& cacheFileName,
            const string& tcName,
            const unordered_map[size_t, ConstDLTensorVec]& inputs,
            unordered_map[size_t, DLTensorVec]& outputs,
            MappingOptions baseMapping,
            vector[MappingOptions] startingPoints,
            const TuningParameterFixer& fixedParams)


cdef extern from "tc/core/flags.h" namespace "tc":

    cdef uint32_t FLAGS_tuner_gen_pop_size
    cdef uint32_t FLAGS_tuner_gen_crossover_rate
    cdef uint32_t FLAGS_tuner_gen_mutation_rate
    cdef uint32_t FLAGS_tuner_gen_generations
    cdef uint32_t FLAGS_tuner_gen_number_elites
    cdef uint32_t FLAGS_tuner_threads
    cdef string FLAGS_tuner_gpus
    cdef bool FLAGS_tuner_print_best
    cdef string FLAGS_tuner_proto
    cdef string FLAGS_tuner_rng_restore
    cdef bool FLAGS_tuner_gen_restore_from_proto
    cdef uint32_t FLAGS_tuner_gen_restore_number
    cdef bool FLAGS_tuner_gen_log_generations
    cdef uint64_t FLAGS_tuner_min_launch_total_threads


cdef extern from "tc/autotuner/genetic_tuning_harness.h" namespace "tc::autotune::detail":

    vector[size_t] parseGpus()


cdef extern from "tc/autotuner/parameters.h" namespace "tc::autotune":

    cdef cppclass TuningParameterFixer:

        void fromMappingOptions(const MappingOptions& options)
