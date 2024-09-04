import cupy
from typing import Optional, List, Callable, Tuple
from abc import ABC, abstractmethod

_set_value_module = cupy.RawModule(code=r"""
#include <cuda_device_runtime_api.h>
extern "C" {
__global__ void set_value_bool(cudaGraphConditionalHandle handle, bool* ptr) {
    cudaGraphSetConditional(handle, *ptr);
}
}
""")
set_value_bool = _set_value_module.get_function("set_value_bool")

def _set_value_to_handle(handle, val: cupy.ndarray):
    if not val.dtype == cupy.bool_:
        # TODO: Implementation for other dtypes
        raise NotImplementedError(
            "_set_value_to_handle has not been implemented "
            "for dtypes other than bool."
        )
    set_value_bool((1,), (1,), (handle, val))

class GraphConverterInterface(ABC):
    @abstractmethod
    def graphify(self, func: Callable):
        pass
    @abstractmethod
    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        fn_args: Tuple=()
    ): pass
    @abstractmethod
    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        false_fn: Optional[Callable]=None,
        fn_args: Tuple=()
    ): pass

class GraphConverter(GraphConverterInterface):
    def __init__(self):
        self.streams: List[cupy.cuda.Stream] = []
        self.main_graph: Optional[cupy.cuda.Graph] = None
        self.root_stream = None

        # Temporal references to prevent evaluated arrays from being freed
        self.cond_outputs: List[cupy.ndarray] = []

    def graphify(self, func: Callable):
        def wrapped(*args, **kwargs):
            if not self.main_graph is None:
                self.main_graph.launch()

            # On initial call
            if self.root_stream is None:
                self.root_stream = cupy.cuda.Stream()
            self.streams.append(self.root_stream)
            with self.root_stream:
                try:
                    self.root_stream.begin_capture()
                    ret = func(*args, **kwargs)
                finally:
                    self.main_graph = self.root_stream.end_capture()
            self.streams.pop()
            self.main_graph.launch(
                stream=self.root_stream
            )
            return ret
        return wrapped

    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        fn_args: Tuple=()
    ):
        """
        equivalent to:
        def while_loop(cond_fn, body_fn, fn_args):
            val = fn_args
            while cond_fn(*val):
                val = body_fn(*val)
            return val
        """
        st = cupy.cuda.Stream()
        parent_st = self.streams[-1]
        self.streams.append(st)
        handle = parent_st.create_conditional_handle(default_value=True)
        cond_before_loop = cond_fn(*fn_args)
        self.cond_outputs.append(cond_before_loop)
        _set_value_to_handle(handle, cond_before_loop) # set value before the loop
        body_graph = parent_st.append_conditional_node(
            "while", handle
        )
        with st:
            try:
                st.begin_capture(dest_graph=body_graph)
                carry = body_fn(*fn_args)
                if len(carry) != len(fn_args):
                    raise ValueError("Argument and return value of body_fn must have same shape")
                for before, after in zip(fn_args, carry):
                    cupy.copyto(before, after) # Copy after -> before
                cond_in_loop = cond_fn(*carry)
                self.cond_outputs.append(cond_in_loop)
                _set_value_to_handle(handle, cond_in_loop) # set value in the loop
            finally:
                st.end_capture()
        self.streams.pop()
        return carry

    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        false_fn: Optional[Callable]=None,
        fn_args: Tuple=()
    ):
        if not false_fn is None:
            raise NotImplementedError(
                "Currently conditional node feature does not support `else` body."
                " `else` support may be added in the future release."
            )
        st = cupy.cuda.Stream()
        parent_st = self.streams[-1]
        self.streams.append(st)
        handle = parent_st.create_conditional_handle(default_value=False)

        cond = cond_fn(*fn_args)
        self.cond_outputs.append(cond)
        _set_value_to_handle(handle, cond) # set value before the loop
        body_graph = parent_st.append_conditional_node(
            "if", handle
        )
        with st:
            try:
                st.begin_capture(dest_graph=body_graph)
                true_fn(*fn_args)
            finally:
                st.end_capture()
        self.streams.pop()
        return None

class MockGraphConverter(GraphConverterInterface):
    def __init__(self):
        pass

    def graphify(self, func: Callable):
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped

    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        fn_args: Tuple=()
    ):
        carry = fn_args
        while cond_fn(*carry):
            carry = body_fn(*carry)
        return carry

    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        false_fn: Optional[Callable]=None,
        fn_args: Tuple=()
    ):
        if not false_fn is None:
            raise NotImplementedError(
                "Currently conditional node feature does not support `else` body."
                " `else` support may be added in the future release."
            )
        if cond_fn(*fn_args):
            true_fn(*fn_args)
        return None
