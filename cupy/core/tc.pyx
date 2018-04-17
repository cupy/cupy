from cpython cimport pycapsule
from cython.operator cimport dereference
from libc.stdint cimport int64_t
from libc.stdlib cimport free
from libcpp.pair cimport pair

import cupy

from cupy.core.core cimport ndarray
from cupy.core.dlpack cimport DLManagedTensor
from cupy.core.dlpack cimport DLTensor
from cupy.core.tc cimport ConstDLTensorVec
from cupy.core.tc cimport DLTensorVec
from cupy.core.tc cimport duration
from cupy.core.tc cimport ExecutionEngine
from cupy.core.tc cimport ExecutorInfo
from cupy.core.tc cimport GeneticAutotuner
from cupy.core.tc cimport MappingOptions
from cupy.core.tc cimport Optional
from cupy.core.tc cimport parseGpus
from cupy.core.tc cimport prunning_ftype
from cupy.core.tc cimport TuningParameterFixer

from cupy.core.tc cimport FLAGS_tuner_gen_pop_size
from cupy.core.tc cimport FLAGS_tuner_gen_crossover_rate
from cupy.core.tc cimport FLAGS_tuner_gen_mutation_rate
from cupy.core.tc cimport FLAGS_tuner_gen_generations
from cupy.core.tc cimport FLAGS_tuner_gen_number_elites
from cupy.core.tc cimport FLAGS_tuner_threads
from cupy.core.tc cimport FLAGS_tuner_gpus
from cupy.core.tc cimport FLAGS_tuner_proto
from cupy.core.tc cimport FLAGS_tuner_gen_restore_from_proto
from cupy.core.tc cimport FLAGS_tuner_gen_restore_number
from cupy.core.tc cimport FLAGS_tuner_gen_log_generations
from cupy.core.tc cimport FLAGS_tuner_min_launch_total_threads


try:
    import tensor_comprehensions as tc
except ImportError:
    pass


FLAGS_tuner_gen_pop_size = 10
FLAGS_tuner_gen_crossover_rate = 80
FLAGS_tuner_gen_mutation_rate = 7
FLAGS_tuner_gen_generations = 2
FLAGS_tuner_gen_number_elites = 1
FLAGS_tuner_threads = 8
FLAGS_tuner_gpus = "0"
FLAGS_tuner_proto = "/tmp/tuner.txt"
FLAGS_tuner_gen_restore_from_proto = False
FLAGS_tuner_gen_restore_number = 10
FLAGS_tuner_gen_log_generations = False
FLAGS_tuner_min_launch_total_threads = 64


cdef class TCKernel:

    cdef ExecutionEngine* engine_ptr
    cdef size_t handle
    cdef string language
    cdef string name

    def __init__(self, language, name, inputs, options='naive'):
        self.engine_ptr = new ExecutionEngine()
        self.compile(language, name, inputs, options=options)

    def __dealloc__(self):
        free(self.engine_ptr)

    cdef string create_options(self, options):
        # TODO(mitmul): This function can return a dereferenced
        # MappingOptions when TC is updated (int the latest code,
        # MappingOptions class has a nullary constructor)
        if options == 'pointwise':
            return MappingOptions.makePointwiseMappingOptions().toProtobufSerializedString()
        elif options == 'mlp':
            return MappingOptions.makeMlpMappingOptions().toProtobufSerializedString()
        elif options == 'conv':
            return MappingOptions.makeConvolutionMappingOptions().toProtobufSerializedString()
        elif options == 'group_conv':
            return MappingOptions.makeGroupConvolutionMappingOptions().toProtobufSerializedString()
        elif options == 'naive':
            return MappingOptions.makeNaiveMappingOptions().toProtobufSerializedString()
        elif isinstance(options, tc.Options):
            return MappingOptions(options.serializeToProtobuf()).toProtobufSerializedString()
        else:
            raise ValueError('Given options argument is invalid.')

    def compile(self, language, name, inputs, options):
        self.language = language.encode('utf-8')
        self.engine_ptr.define(self.language)
        cdef vector[const DLTensor*] input_tensors
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            input_tensors.push_back(&dlm_tensor.dl_tensor)

        cdef string name_str = name.encode('utf-8')
        # TODO(mitmul): "create_options" should return MappingOptions object directly
        cdef size_t handle = self.engine_ptr.compile(
            name_str, input_tensors, MappingOptions(self.create_options(options)))
        self.handle = handle
        self.name = name_str


    def prepareOutputs(self, name, inputs):
        cdef vector[const DLTensor*] input_tensors
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            input_tensors.push_back(&dlm_tensor.dl_tensor)

        cdef vector[const DLTensor*] output_tensors = \
            self.engine_ptr.inferOutputTensorInfo(name, input_tensors)

        ndarray_outputs = []
        for output in output_tensors:
            if output.dtype.code == 0:  # integer type
                if output.dtype.bits == 8:
                    dtype = cupy.int8
                elif output.dtype.bits == 16:
                    dtype = cupy.int16
                elif output.dtype.bits == 32:
                    dtype = cupy.int32
                elif output.dtype.bits == 64:
                    dtype = cupy.int64
            elif output.dtype.code == 1:  # unsigned integer type
                if output.dtype.bits == 8:
                    dtype = cupy.uint8
                elif output.dtype.bits == 16:
                    dtype = cupy.uint16
                elif output.dtype.bits == 32:
                    dtype = cupy.uint32
                elif output.dtype.bits == 64:
                    dtype = cupy.uint64
            elif output.dtype.code == 2:  # float type
                if output.dtype.bits == 16:
                    dtype = cupy.float16
                elif output.dtype.bits == 32:
                    dtype = cupy.float32
                elif output.dtype.bits == 64:
                    dtype = cupy.float64
            else:
                raise TypeError('Unknown dtype')
            shape = [output.shape[i] for i in range(output.ndim)]
            ndarray_outputs.append(ndarray(shape, dtype))

        return ndarray_outputs
        
    cdef vector[const DLTensor*] toConstDlpackTensors(self, arrays):
        cdef vector[const DLTensor*] tensors
        for array in arrays:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            tensors.push_back(<const DLTensor*>(&dlm_tensor.dl_tensor))
        return tensors

    cdef vector[DLTensor*] toDlpackTensors(self, arrays):
        cdef vector[DLTensor*] tensors
        for array in arrays:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            tensors.push_back(&dlm_tensor.dl_tensor)
        return tensors

    def autotune(self, inputs, cache_name, base_mapping, starting_points=None, pop_size=10,
                 crossover_rate=80, mutation_rate=7, generations=2,
                 number_elites=1, threads=8, gpus=[0], proto="/tmp/tuner.txt",
                 restore_from_proto=False, restore_number=10, log_generations=False,
                 tuner_min_launch_total_threads=64):

        global FLAGS_tuner_gen_pop_size
        global FLAGS_tuner_gen_crossover_rate
        global FLAGS_tuner_gen_mutation_rate
        global FLAGS_tuner_gen_generations
        global FLAGS_tuner_gen_number_elites
        global FLAGS_tuner_threads
        global FLAGS_tuner_gpus
        global FLAGS_tuner_proto
        global FLAGS_tuner_gen_restore_from_proto
        global FLAGS_tuner_gen_restore_number
        global FLAGS_tuner_gen_log_generations
        global FLAGS_tuner_min_launch_total_threads
        FLAGS_tuner_gen_pop_size = pop_size
        FLAGS_tuner_gen_crossover_rate = crossover_rate
        FLAGS_tuner_gen_mutation_rate = mutation_rate
        FLAGS_tuner_gen_generations = generations
        FLAGS_tuner_gen_number_elites = number_elites
        FLAGS_tuner_threads = threads
        FLAGS_tuner_gpus = ','.join([str(g) for g in gpus]).encode('utf-8')
        FLAGS_tuner_proto = proto.encode('utf-8')
        FLAGS_tuner_gen_restore_from_proto = restore_from_proto
        FLAGS_tuner_gen_restore_number = restore_number
        FLAGS_tuner_gen_log_generations = log_generations
        FLAGS_tuner_min_launch_total_threads = tuner_min_launch_total_threads

        cdef GeneticAutotuner* tuner_ptr = new GeneticAutotuner(self.language)

        cdef vector[size_t] available_gpus = parseGpus()
        cdef unordered_map[size_t, ConstDLTensorVec] inputsPerGpu
        cdef pair[size_t, ConstDLTensorVec] inputs_pair

        outputs = self.prepareOutputs(self.name, inputs)
        cdef unordered_map[size_t, DLTensorVec] outputsPerGpu
        cdef pair[size_t, DLTensorVec] outputs_pair

        for gpu in available_gpus:
            with cupy.cuda.Device(gpu):
                for x in inputs:
                    x = cupy.asarray(x)
                inputs_pair = pair[size_t, ConstDLTensorVec](
                    gpu, self.toConstDlpackTensors(inputs))
                inputsPerGpu.insert(inputs_pair)

                for y in outputs:
                    y = cupy.asarray(y)
                outputs_pair = pair[size_t, DLTensorVec](
                    gpu, self.toDlpackTensors(outputs))
                outputsPerGpu.insert(outputs_pair)

        cdef MappingOptions* options_ptr = new MappingOptions(
            self.create_options(base_mapping))
        cdef vector[MappingOptions] options_vec
        options_vec.push_back(dereference(options_ptr))
        cdef TuningParameterFixer* params = new TuningParameterFixer()
        cdef Optional[MappingOptions] best_options = tuner_ptr.tune(
            cache_name.encode('utf-8'),
            self.name,
            inputsPerGpu,
            outputsPerGpu,
            dereference(options_ptr),
            options_vec,
            dereference(params)
        )

    def __call__(self, *inputs, **kwargs):
        profile = kwargs.pop('profile') if 'profile' in kwargs else False
        outputs = self.prepareOutputs(self.name, inputs)

        cdef vector[const DLTensor*] input_tensors = self.toConstDlpackTensors(inputs)
        cdef vector[DLTensor*] output_tensors = self.toDlpackTensors(outputs)
        cdef duration time = self.engine_ptr.run(
            self.handle,
            input_tensors,
            output_tensors,
            profile,
            <prunning_ftype>prunningFunction)
        
        return outputs


cdef bool prunningFunction(const ExecutorInfo* info):
    return False
    
