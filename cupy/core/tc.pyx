from cython.operator cimport dereference
from cpython cimport pycapsule
from cupy.core.tc cimport ExecutionEngine
from cupy.core.tc cimport MappingOptions
from cupy.core.dlpack cimport DLManagedTensor

from libc.stdint cimport int64_t

cdef class CuPyCompiler:

    cdef ExecutionEngine* engine_ptr

    def __init__(self):
        self.engine_ptr = new ExecutionEngine()

    def define(self, language):
        self.engine_ptr.define(language.encode('utf-8'))

    def compile(self, name, inputs):
        cdef vector[const DLTensor*] tensors
        #cdef DLManagedTensor* dlm_tensor

        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            print(
                <size_t>array.data.ptr,
                <size_t>dlm_tensor.dl_tensor.data,
                dlm_tensor.dl_tensor.ndim,
                dlm_tensor.dl_tensor.dtype.code,
                dlm_tensor.dl_tensor.dtype.bits,
                dlm_tensor.dl_tensor.dtype.lanes
            )

            tensors.push_back(&dlm_tensor.dl_tensor)
        return self.engine_ptr.compile(
            name.encode('utf-8'), tensors, MappingOptions.makeNaiveMappingOptions())
