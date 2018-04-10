from cython.operator cimport dereference
from cpython cimport pycapsule
from cupy.core.tc cimport ExecutionEngine
from cupy.core.tc cimport MappingOptions
from cupy.core.dlpack cimport DLManagedTensor


cdef class CuPyCompiler:

    cdef ExecutionEngine* engine_ptr

    def __init__(self):
        self.engine_ptr = new ExecutionEngine()

    def define(self, language):
        self.engine_ptr.define(language.encode('utf-8'))

    def compile(self, name, inputs):
        cdef vector[const DLTensor*] tensors
        cdef DLManagedTensor* dlm_tensor
        for array in inputs:
            tensor = array.toDlpack()
            dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')
            tensors.push_back(&(dlm_tensor.dl_tensor))
        return self.engine_ptr.compile(name.encode('utf-8'), tensors, MappingOptions.makeNaiveMappingOptions())
