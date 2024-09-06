import cupy
from cupy import cuda
from typing import (
    Optional,
    List,
    Callable,
    Tuple,
    Union,
)
from collections import deque
from abc import ABC, abstractmethod

_set_value_module = cupy.RawModule(code=r"""
#include <cuda_device_runtime_api.h>
extern "C" {
__global__ void set_value_bool(cudaGraphConditionalHandle handle, bool* ptr) {
    cudaGraphSetConditional(handle, *ptr);
}

}
""")
_set_value_bool = _set_value_module.get_function("set_value_bool")

def _set_value_to_handle(handle, val: cupy.ndarray):
    if not val.dtype == cupy.bool_:
        # TODO: Implementation for other dtypes
        raise NotImplementedError(
            "_set_value_to_handle has not been implemented "
            "for dtypes other than bool."
        )
    if val.shape != ():
        raise ValueError("val must be 1d size 1 array")
    _set_value_bool((1,), (1,), (handle, val))

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
    ):
        pass
    @abstractmethod
    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        fn_args: Tuple=()
    ):
        pass
    @abstractmethod
    def multicond(
        self,
        branches: List[
            Tuple[Optional[Callable], Callable] or
            Tuple[Optional[Callable], Callable, Tuple]
        ],
    ):
        pass

class GraphConverter(GraphConverterInterface):
    def __init__(self):
        self.streams: List[cuda.Stream] = []
        self.main_graph: Optional[cuda.Graph] = None
        self.root_stream = None

        # Temporal references to prevent evaluated arrays from being freed
        self.cond_outputs: List[cupy.ndarray] = []

        self.memory_pool = cuda.MemoryPool()

    def graphify(self, func: Callable):
        def wrapped(*args, **kwargs):
            with cuda.using_allocator(self.memory_pool.malloc):
                if not self.main_graph is None:
                    self.main_graph.launch()

                # On initial call
                if self.root_stream is None:
                    self.root_stream = cuda.Stream()
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
        st = cuda.Stream()
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
        fn_args: Tuple=()
    ):
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

    def multicond(
        self,
        branches: List[
            Tuple[Optional[Callable], Callable] or
            Tuple[Optional[Callable], Callable, Tuple]
        ],
    ):
        """Multiconditional switch

        ```
        gc.multicond([
            (cond_fn0, fn0, args0),
            (cond_fn1, fn1, args1),
            (None, fn2, args2),
        ])
        ```

        The code above will be expanded to:

        ```
        if cond_fn0():
            fn0(*args0)
        else if cond_fn1():
            fn1(*args1)
        else:
            fn2(*args2)
        ```

        """
        if len(branches) == 1:
            cond_fn, true_fn, fn_args = branches[0]
            return self.cond(cond_fn, true_fn, fn_args)

        # We implement
        # ```
        # if cond0():
        #    fn0()
        # elif cond1():
        #    fn1()
        # ```
        # as
        # ```
        # val = cond0()
        # if val:
        #     fn0()
        # if not val and cond1():
        #     fn1()
        # ```
        # because of conditional node constraint.

        branches = deque(branches)
        self._multicond_inner(branches)

        return None

    def _multicond_inner(
        self,
        branches: deque,
    ):
        first = branches.popleft()
        cond_fn, body_fn, *_ = first
        args = () if len(first) == 2 else first[2]

        if cond_fn is None:
            body_fn(*args)
            return

        parent_st = self.streams[-1]
        handle = parent_st.create_conditional_handle()

        # True branch
        cond_val = cond_fn()
        self.cond_outputs.append(cond_val)
        _set_value_to_handle(handle, cond_val)
        body_graph_true = parent_st.append_conditional_node(
            "if", handle
        )

        stream = cuda.Stream()
        self.streams.append(stream)
        with stream:
            stream.begin_capture(dest_graph=body_graph_true)
            try:
                body_fn(*args)
            finally:
                stream.end_capture()
        self.streams.pop()

        # False branch
        if len(branches) > 0:
            stream = cuda.Stream()
            self.streams.append(stream)
            handle2 = parent_st.create_conditional_handle()
            _set_value_to_handle(handle2, ~cond_val)
            body_graph_false = parent_st.append_conditional_node(
                "if", handle2
            )
            with stream:
                stream.begin_capture(dest_graph=body_graph_false)
                try:
                    self._multicond_inner(branches)
                finally:
                    stream.end_capture()
            self.streams.pop()


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
        fn_args: Tuple=()
    ):
        if cond_fn(*fn_args):
            true_fn(*fn_args)
        return None

    def multicond(
        self,
        branches: List[
            Tuple[Optional[Callable], Callable] or
            Tuple[Optional[Callable], Callable, Tuple]
        ],
    ):
        for branch in branches:
            if len(branch) == 2:
                cond_fn, body_fn = branch
                args = ()
            elif len(branch) == 3:
                cond_fn, body_fn, args = branch
            else:
                raise ValueError("\n".join([
                    "branches must be an instance of",
                    "List[",
                    "    Tuple[Optional[Callable], Callable] or",
                    "    Tuple[Optional[Callable], Callable, Tuple]",
                    "]"
                ]))
            if cond_fn is None or cond_fn():
                body_fn(*args)
                break

class TempVariableStorage:
    special_attrs = {"_attributes", "_old_attributes"}
    def __init__(self):
        self._attributes = dict()
        self._old_attributes = []
    def __setattr__(self, name, value):
        if name in TempVariableStorage.special_attrs:
            super().__setattr__(name, value)
        else:
            if name in self._attributes.keys():
                self._old_attributes.append((name, self._attributes[name]))
            self._attributes[name] = value
    def __getattr__(self, name):
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    def __delattr__(self, name):
        raise AttributeError(f"Deleting attribute is not allowed")
