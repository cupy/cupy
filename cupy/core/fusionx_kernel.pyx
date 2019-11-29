from cupy.core.core cimport ndarray, Indexer
from cupy.core._routines_manipulation cimport _broadcast_core
from cupy.core.fusionx import _FusionXVarArray
from cupy import util

class FusionXKernel(object):
    def __init__(self, fusionx):
        self.fusionx = fusionx

    def _reset_vals(self):
        for pvar in self.fusionx.param_list:
            if isinstance(pvar, _FusionXVarArray):
                pvar.ndarray = None            

    def _set_real_shape(self, shape_map):
        for pvar in self.fusionx.param_list:
            if isinstance(pvar, _FusionXVarArray):
                ndim = pvar.ndim
                real_shape = [None for _ in range(ndim)]
                for i in range(ndim):
                    assert pvar.abstracted_shape[i] in shape_map
                    real_shape[i] = shape_map[pvar.abstracted_shape[i]]
                pvar.real_shape = tuple(real_shape)
                pvar.set_size()

    def _set_ndarray(self, args):
        for pvar in self.fusionx.param_list_base:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.is_input:
                    pvar.ndarray = args[pvar.input_order]
                else:
                    pvar.ndarray = ndarray(pvar.real_shape, pvar.dtype)

    def _broadcast(self):
        for pvar in self.fusionx.param_list:
            if isinstance(pvar, _FusionXVarArray):
                if pvar.broadcasted_from is not None:
                    assert pvar.broadcasted_from.ndarray is not None
                    value, _ = _broadcast_core(pvar.broadcasted_from.ndarray)
                    pvar.ndarray = value

    def _reduce_dims(self):
        pass
    
    def __call__(self, shape_map, *args):
        self._reset_vals()
        self._set_real_shape(shape_map)
        self._set_ndarray(args)
        self._broadcast()
        self._reduce_dims()

        print(self.cuda_body)
