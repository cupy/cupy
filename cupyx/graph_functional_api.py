import cupy
from cupy import cuda, _util
from cupy.cuda.graph import (
    _append_conditional_node_to_stream,
    _create_conditional_handle_from_stream
)
from cupy_backends.cuda import stream as backend_stream
from typing import (
    Optional,
    List,
    Callable,
    Tuple,
    Union,
    cast
)
from collections import deque
from abc import ABC, abstractmethod

_util.experimental("cupyx.graph_functional_api")

# Conditional value kernel definition
_set_value_kernel_name = "cupy_cudaGraphSetConditional"
_set_value_cuda_types = {
    "bool": "bool",
    "int8": "char",
    "int16": "short",
    "int32": "int",
    "int64": "long long",
    "uint8": "unsigned char",
    "uint16": "unsigned short",
    "uint32": "unsigned int",
    "uint64": "unsigned long long",
    "float16": "half",
    "float32": "float",
    "float64": "double"
}
_set_value_kernel_module = cupy.RawModule(code=rf"""
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void {_set_value_kernel_name}(
    cudaGraphConditionalHandle handle, const T* ptr, bool invert
) {{
    unsigned int value = (unsigned int)*ptr;
    cudaGraphSetConditional(handle, (invert)? !value : value);
}}
""", name_expressions=[
    f"{_set_value_kernel_name}<{t}>" for t in _set_value_cuda_types.values()
])


def _set_value_to_handle(handle, val: cupy.ndarray, invert=False):
    dtype_name = val.dtype.name
    if dtype_name not in _set_value_cuda_types.keys():
        raise ValueError(
            "Conditional function must return any array of dtype " +
            str(tuple(_set_value_cuda_types.keys())) +
            f", but got `{dtype_name}`."
        )
    if val.size != 1:
        raise ValueError(
            "Conditional function must return array of size 1, "
            f"but got `{val.size}`.")
    cuda_type_name = _set_value_cuda_types[dtype_name]
    _set_value_fn = _set_value_kernel_module.get_function(
        f"{_set_value_kernel_name}<{cuda_type_name}>"
    )
    _set_value_fn((1,), (1,), (handle, val, invert))


class GraphBuilderInterface(ABC):
    @abstractmethod
    def graphify(self, func: Callable):
        pass

    @abstractmethod
    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        fn_args: Tuple = ()
    ):
        pass

    @abstractmethod
    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        fn_args: Tuple = ()
    ):
        pass

    @abstractmethod
    def multicond(
        self,
        branches: List[Union[
            Tuple[Optional[Callable], Callable],
            Tuple[Optional[Callable], Callable, Tuple]
        ]],
    ):
        pass


class GraphBuilder(GraphBuilderInterface):
    def __init__(self):
        self._streams: List[cuda.Stream] = []
        self._root_graph: Optional[cuda.Graph] = None
        self._memory_pool = cuda.MemoryPool()
        self._target_func: Optional[Callable] = None
        self._return_ref = None
        self._cublas_workspace = None
        self._prev_cublas_workspace_config = None

    def graphify(self, func: Callable):
        self._target_func = func

        def wrapped(*args):
            if self._root_graph is not None:
                self._root_graph.launch()
                return self._return_ref

            self.capture(fn_args=args)
            self._root_graph.launch()
            return self._return_ref
        return wrapped

    def capture(self, fn_args=()) -> cuda.Graph:
        if self._target_func is None:
            raise RuntimeError(
                "Set graph target function before calling capture() "
                "by using graphify() method."
            )
        if self._root_graph is not None:
            return self._root_graph

        with cuda.using_allocator(self._memory_pool.malloc):
            root_stream = cuda.Stream()
            self._streams.append(root_stream)
            with root_stream:
                self._allocate_cublas_workspace()
                root_stream.begin_capture()
                try:
                    self._return_ref = self._target_func(*fn_args)
                finally:
                    self._root_graph = root_stream.end_capture()
                    self._reset_cublas_workspace()
            self._streams.pop()

        # Setting ref to captured graph to avoid freeing memory
        self._root_graph._add_ref(self._memory_pool)
        if self._cublas_workspace is not None:
            self._root_graph._add_ref(self._cublas_workspace)

        return self._root_graph

    def _allocate_cublas_workspace(self):
        # Prepare cuBLAS workspace memory to avoid stream memory allocation
        # which is incompatible with child graphs.
        # See: https://docs.nvidia.com/cuda/cublas/#cuda-graphs-support
        # TODO: respect CUBLAS_WORKSPACE_CONFIG env var
        dev = cuda.Device()
        self._prev_cublas_workspace_config = \
            backend_stream.get_current_cublas_workspace(dev.id)

        cc = int(dev.compute_capability)
        if cc >= 90:  # Hopper or newer
            workspace_size = 32 * 1024 * 1024
        else:
            workspace_size = 4 * 1024 * 1024
        workspace = self._memory_pool.malloc(workspace_size)
        backend_stream.set_current_cublas_workspace(
            workspace.ptr, workspace_size, dev.id
        )
        self._cublas_workspace = (workspace, workspace_size)

    def _reset_cublas_workspace(self):
        dev = cuda.Device()
        ptr, size = self._prev_cublas_workspace_config
        backend_stream.set_current_cublas_workspace(
            ptr, size, dev.id
        )
        self._cublas_workspace = None

    def while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        fn_args: Tuple = ()
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
        handle = _create_conditional_handle_from_stream(
            parent_st, default_value=True)
        cond_before_loop = cond_fn(*fn_args)
        # set value before the loop
        _set_value_to_handle(handle, cond_before_loop)
        body_graph = _append_conditional_node_to_stream(
            parent_st, "while", handle
        )
        with st:
            st.begin_capture(to_graph=body_graph)
            try:
                carry = body_fn(*fn_args)
                if carry is not None:
                    if len(carry) != len(fn_args):
                        raise ValueError(
                            "Argument and return value of body_fn"
                            " must have the same shape"
                        )
                    for before, after in zip(fn_args, carry):
                        if before.shape != after.shape:
                            raise ValueError(
                                "Argument and return value of body_fn"
                                " must have the same shape"
                            )
                        if before.data.ptr != after.data.ptr:
                            # Copy after -> before
                            cupy.copyto(before, after)
                cond_in_loop = cond_fn(*fn_args)
                # set value in the loop
                _set_value_to_handle(handle, cond_in_loop)
            finally:
                st.end_capture()
        self._streams.pop()
        return carry

    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        fn_args: Tuple = ()
    ):
        st = cupy.cuda.Stream()
        parent_st = self._streams[-1]
        self._streams.append(st)
        handle = _create_conditional_handle_from_stream(
            parent_st, default_value=False)

        cond = cond_fn(*fn_args)
        _set_value_to_handle(handle, cond)  # set value before the loop
        body_graph = _append_conditional_node_to_stream(
            parent_st, "if", handle
        )
        with st:
            st.begin_capture(to_graph=body_graph)
            try:
                ret = true_fn(*fn_args)
            finally:
                st.end_capture()
        self._streams.pop()
        return ret

    def multicond(
        self,
        branches: List[Union[
            Tuple[Optional[Callable], Callable],
            Tuple[Optional[Callable], Callable, Tuple]
        ]],
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
            first = branches[0]
            if len(first) == 2:
                cond_fn, true_fn = cast(
                    Tuple[Optional[Callable], Callable], first)
                fn_args: Tuple = ()
            elif len(first) == 3:
                cond_fn, true_fn, fn_args = cast(
                    Tuple[Optional[Callable], Callable, Tuple], first)
            else:
                raise ValueError("\n".join([
                    "branches must be an instance of",
                    "List[",
                    "    Tuple[Optional[Callable], Callable] or",
                    "    Tuple[Optional[Callable], Callable, Tuple]",
                    "]"
                ]))

            if cond_fn is not None:
                self.cond(cond_fn, true_fn, fn_args)
            else:
                raise ValueError(
                    "`cond_fn` of the first element of"
                    " `branches` must not be `None`"
                )
            return

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

        branches_deque = deque(branches)
        self._multicond_inner(branches_deque)

        return None

    def _multicond_inner(
        self,
        branches: deque,
    ):
        first = branches.popleft()
        if len(first) == 2:
            cond_fn, body_fn = cast(
                Tuple[Optional[Callable], Callable], first)
            args: Tuple = ()
        elif len(first) == 3:
            cond_fn, body_fn, args = cast(
                Tuple[Optional[Callable], Callable, Tuple], first)
        else:
            raise ValueError("\n".join([
                "branches must be an instance of",
                "List[",
                "    Tuple[Optional[Callable], Callable] or",
                "    Tuple[Optional[Callable], Callable, Tuple]",
                "]"
            ]))

        if cond_fn is None:
            body_fn(*args)
            return

        parent_st = self._streams[-1]
        handle = _create_conditional_handle_from_stream(parent_st)

        # True branch
        cond_val = cond_fn()
        _set_value_to_handle(handle, cond_val)
        body_graph_true = _append_conditional_node_to_stream(
            parent_st, "if", handle
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
            handle2 = _create_conditional_handle_from_stream(parent_st)
            _set_value_to_handle(handle2, cond_val, invert=True)
            body_graph_false = _append_conditional_node_to_stream(
                parent_st, "if", handle2
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
        fn_args: Tuple = ()
    ):
        carry = fn_args
        while cond_fn(*carry):
            carry = body_fn(*carry)
        return carry

    def cond(
        self,
        cond_fn: Callable,
        true_fn: Callable,
        fn_args: Tuple = ()
    ):
        if cond_fn(*fn_args):
            true_fn(*fn_args)
        return None

    def multicond(
        self,
        branches: List[Union[
            Tuple[Optional[Callable], Callable],
            Tuple[Optional[Callable], Callable, Tuple]
        ]],
    ):
        for branch in branches:
            if len(branch) == 2:
                cond_fn, body_fn = cast(
                    Tuple[Optional[Callable], Callable], branch)
                args: Tuple = ()
            elif len(branch) == 3:
                cond_fn, body_fn, args = cast(
                    Tuple[Optional[Callable], Callable, Tuple], branch)
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
