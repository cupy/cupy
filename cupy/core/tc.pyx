from cpython cimport pycapsule
from cython.operator cimport dereference
from libc.stdint cimport int64_t

import cupy

from cupy.core.core cimport ndarray
from cupy.core.dlpack cimport DLManagedTensor
from cupy.core.dlpack cimport DLTensor
from cupy.core.tc cimport duration
from cupy.core.tc cimport ExecutionEngine
from cupy.core.tc cimport MappingOptions


cdef class CuPyCompiler:

    cdef ExecutionEngine* engine_ptr

    def __init__(self):
        self.engine_ptr = new ExecutionEngine()

    def define(self, language):
        self.engine_ptr.define(language.encode('utf-8'))

    def compile(self, name, inputs):
        cdef vector[const DLTensor*] input_tensors
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            input_tensors.push_back(&dlm_tensor.dl_tensor)

        cdef size_t handle = self.engine_ptr.compile(
            name.encode('utf-8'), input_tensors, MappingOptions.makeNaiveMappingOptions())

        return handle

    def prepareOutputs(self, name, inputs):
        cdef vector[const DLTensor*] input_tensors
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            input_tensors.push_back(&dlm_tensor.dl_tensor)

        cdef vector[const DLTensor*] output_tensors = \
            self.engine_ptr.inferOutputTensorInfo(name.encode(), input_tensors)

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

    def run(self, handle, inputs, outputs, profile=False):
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
            handle,
            input_tensors,
            output_tensors,
            profile)

