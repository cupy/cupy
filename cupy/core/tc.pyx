from cpython cimport pycapsule
from cython.operator cimport dereference
from libc.stdint cimport int64_t

import cupy

from cupy.core.core cimport ndarray
from cupy.core.dlpack cimport DLManagedTensor
from cupy.core.dlpack cimport DLTensor
from cupy.core.tc cimport duration
from cupy.core.tc cimport ExecutionEngine
from cupy.core.tc cimport ExecutorInfo
from cupy.core.tc cimport prunning_ftype
from cupy.core.tc cimport MappingOptions

try:
    import tensor_comprehensions as tc
except ImportError:
    pass


cdef class TCKernel:

    cdef ExecutionEngine* engine_ptr
    cdef size_t handle
    cdef string name

    def __init__(self, language, name, inputs, options='naive'):
        self.engine_ptr = new ExecutionEngine()
        self.compile(language, name, inputs, options=options)

    def compile(self, language, name, inputs, options):
        self.engine_ptr.define(language.encode('utf-8'))
        cdef vector[const DLTensor*] input_tensors
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            input_tensors.push_back(&dlm_tensor.dl_tensor)

        cdef size_t handle
        cdef string name_str = name.encode('utf-8')
        if options == 'pointwise':
            handle = self.engine_ptr.compile(
                name_str, input_tensors, MappingOptions.makePointwiseMappingOptions())
        elif options == 'mlp':
            handle = self.engine_ptr.compile(
                name_str, input_tensors, MappingOptions.makeMlpMappingOptions())
        elif options == 'conv':
            handle = self.engine_ptr.compile(
                name_str, input_tensors, MappingOptions.makeConvolutionMappingOptions())
        elif options == 'group_conv':
            handle = self.engine_ptr.compile(
                name_str, input_tensors, MappingOptions.makeGroupConvolutionMappingOptions())
        elif options == 'naive':
            handle = self.engine_ptr.compile(
                name_str, input_tensors, MappingOptions.makeNaiveMappingOptions())
        elif isinstance(options, tc.Options):
            handle = self.engine_ptr.compile(
                name_str, input_tensors, MappingOptions(options.serializeToProtobuf()))
        else:
            raise ValueError('Given options argument is invalid.')
        
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

    def autotune(self, inputs, pop_size, crossover_rate, mutation_rate, generations, number_elites, threads, gpus, proto, restore_from_proto, restore_number, log_generations, tuner_min_launch_total_threads, tune):
        pass

    def __call__(self, *inputs, **kwargs):
        profile = kwargs.pop('profile') if 'profile' in kwargs else False
        outputs = self.prepareOutputs(self.name, inputs)

        cdef vector[const DLTensor*] input_tensors
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            input_tensors.push_back(<const DLTensor*>(&dlm_tensor.dl_tensor))

        cdef vector[DLTensor*] output_tensors
        for array in outputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            output_tensors.push_back(&dlm_tensor.dl_tensor)

        cdef duration time = self.engine_ptr.run(
            self.handle,
            input_tensors,
            output_tensors,
            profile,
            <prunning_ftype>prunningFunction)
        
        return outputs

cdef bool prunningFunction(const ExecutorInfo* info):
    return False
    