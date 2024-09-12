import cupy
from cupy import cuda
from cupy_backends.cuda import stream as backend_stream
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
__global__ void cupy_cudaGraphSetConditional_bool(
    cudaGraphConditionalHandle handle, bool* ptr
) {
    cudaGraphSetConditional(handle, *ptr);
}
}
""")
_set_value_bool = _set_value_module.get_function("cupy_cudaGraphSetConditional_bool")

def _set_value_to_handle(handle, val: cupy.ndarray):
    if not val.dtype == cupy.bool_:
        # TODO: Implementation for other dtypes
        raise NotImplementedError(
            "_set_value_to_handle has not been implemented "
            "for dtypes other than bool."
        )
    if val.size != 1:
        raise ValueError(
            "Conditional function must return array of size 1, "
            f"but got {val.size}.")
    _set_value_bool((1,), (1,), (handle, val))

class GraphBuilderInterface(ABC):
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

class GraphBuilder(GraphBuilderInterface):
    def __init__(self):
        self._streams: List[cuda.Stream] = []
        self.root_graph: Optional[cuda.Graph] = None
        self._memory_pool = cuda.MemoryPool()
        self._target_func: Optional[Callable] = None
        self._return_ref = None

    def graphify(self, func: Callable):
        self._target_func = func
        def wrapped(*args):
            if not self.root_graph is None:
                self.root_graph.launch()
                return self._return_ref

            self.capture(fn_args=args)
            self.root_graph.launch()
            return self._return_ref
        return wrapped

    def capture(self, fn_args=()):
        if self._target_func is None:
            raise RuntimeError(
                "Set graph target function before calling capture() "
                "by using graphify() method."
            )
        with cuda.using_allocator(self._memory_pool.malloc):
            self._allocate_cublas_workspace()
            # On initial call
            root_stream = cuda.Stream()
            self._streams.append(root_stream)
            with root_stream:
                try:
                    root_stream.begin_capture()
                    self._return_ref = self._target_func(*fn_args)
                finally:
                    self.root_graph = root_stream.end_capture()
            self._streams.pop()

    def _allocate_cublas_workspace(self):
        # Prepare cuBLAS workspace memory to avoid stream memory allocation
        # which is incompatible with child graphs.
        # See: https://docs.nvidia.com/cuda/cublas/#cuda-graphs-support
        # TODO: respect CUBLAS_WORKSPACE_CONFIG env var
        dev = cuda.Device()
        cc = int(dev.compute_capability)
        if cc >= 90: # Hopper or newer
            workspace_size = 32 * 1024 * 1024
        else:
            workspace_size = 4 * 1024 * 1024
        workspace = self._memory_pool.malloc(workspace_size)
        backend_stream.set_current_cublas_workspace(
            workspace.ptr, workspace_size, dev.id
        )

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
        parent_st = self._streams[-1]
        self._streams.append(st)
        handle = parent_st.create_conditional_handle(default_value=True)
        cond_before_loop = cond_fn(*fn_args)
        _set_value_to_handle(handle, cond_before_loop) # set value before the loop
        body_graph = parent_st.append_conditional_node(
            "while", handle
        )
        with st:
            try:
                st.begin_capture(to_graph=body_graph)
                carry = body_fn(*fn_args)
                if len(carry) != len(fn_args):
                    raise ValueError("Argument and return value of body_fn must have same shape")
                for before, after in zip(fn_args, carry):
                    # TODO: Skip copy when before pointer and after pointer are same
                    # and add type validation
                    cupy.copyto(before, after) # Copy after -> before
                cond_in_loop = cond_fn(*carry)
                _set_value_to_handle(handle, cond_in_loop) # set value in the loop
            finally:
                st.end_capture()
        self._streams.pop()
        return carry

    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        fn_args: Tuple=()
    ):
        st = cupy.cuda.Stream()
        parent_st = self._streams[-1]
        self._streams.append(st)
        handle = parent_st.create_conditional_handle(default_value=False)

        cond = cond_fn(*fn_args)
        _set_value_to_handle(handle, cond) # set value before the loop
        body_graph = parent_st.append_conditional_node(
            "if", handle
        )
        with st:
            try:
                st.begin_capture(to_graph=body_graph)
                true_fn(*fn_args)
            finally:
                st.end_capture()
        self._streams.pop()
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
            cond_fn, true_fn, fn_args = (*branches[0], ())
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

        parent_st = self._streams[-1]
        handle = parent_st.create_conditional_handle()

        # True branch
        cond_val = cond_fn()
        _set_value_to_handle(handle, cond_val)
        body_graph_true = parent_st.append_conditional_node(
            "if", handle
        )

        stream = cuda.Stream()
        self._streams.append(stream)
        with stream:
            stream.begin_capture(to_graph=body_graph_true)
            try:
                body_fn(*args)
            finally:
                stream.end_capture()
        self._streams.pop()

        # False branch
        if len(branches) > 0:
            stream = cuda.Stream()
            self._streams.append(stream)
            handle2 = parent_st.create_conditional_handle()
            _set_value_to_handle(handle2, ~cond_val)
            body_graph_false = parent_st.append_conditional_node(
                "if", handle2
            )
            with stream:
                stream.begin_capture(to_graph=body_graph_false)
                try:
                    self._multicond_inner(branches)
                finally:
                    stream.end_capture()
            self._streams.pop()


class MockGraphBuilder(GraphBuilderInterface):
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
