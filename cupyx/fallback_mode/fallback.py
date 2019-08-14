"""
`fallback_mode` for cupy. Whenever a method is not yet implemented in CuPy,
it will fallback to corresponding NumPy method.
"""
import sys
import types

import numpy as np

import cupy as cp


class _RecursiveAttr(object):
    """
    RecursiveAttr class to catch all attributes corresponding to numpy,
    when user calls fallback_mode. numpy is an instance of this class.
    """

    def __init__(self, numpy_object, cupy_object, array=None):
        """
        _RecursiveAttr initializer.

        Args:
            numpy_object (method): NumPy method.
            cupy_method (method): Corresponding CuPy method.
            array (ndarray): Acts as flag to know if _RecursiveAttr object
            is called from ``ndarray`` class. Also, acts as container for
            modifying args in case it is called from ``ndarray``.
            None otherwise.
        """

        self._numpy_object = numpy_object
        self._cupy_object = cupy_object
        self._fallback_array = array

    def __instancecheck__(self, instance):
        """
        Enable support for isinstance(instance, _RecursiveAttr instance)
        by redirecting it to appropriate isinstance method.
        """

        if self._cupy_object is not None:
            return isinstance(instance, self._cupy_object)

        return isinstance(instance, self._numpy_object)

    def __getattr__(self, attr):
        """
        Catches attributes corresponding to numpy.

        Runs recursively till attribute gets called.
        Or numpy ScalarType is retrieved.

        Args:
            attr (str): Attribute of _RecursiveAttr class object.

        Returns:
            (_RecursiveAttr object, NumPy scalar):
            Returns_RecursiveAttr object with new numpy_object, cupy_object.
            Returns objects in cupy which is an alias of numpy object.
        """

        numpy_object = getattr(self._numpy_object, attr)
        cupy_object = getattr(self._cupy_object, attr, None)

        if numpy_object is np.ndarray:
            return ndarray

        if numpy_object is np.vectorize:
            return vectorize

        if numpy_object is cupy_object:
            return numpy_object

        return _RecursiveAttr(numpy_object, cupy_object)

    def __repr__(self):

        if isinstance(self._numpy_object, types.ModuleType):
            return "<numpy = module {}, cupy = module {}>".format(
                self._numpy_object.__name__,
                getattr(self._cupy_object, '__name__', None))

        return "<numpy = {}, cupy = {}>".format(
            self._numpy_object, self._cupy_object)

    @property
    def __doc__(self):
        return self._numpy_object.__doc__

    @staticmethod
    def _is_cupy_compatible(args, kwargs):
        """
        Returns False if ndarray is not compatible with CuPy.
        """
        for arg in args:
            if isinstance(arg, ndarray) and arg._class is not cp.ndarray:
                return False

        for key in kwargs:
            if isinstance(kwargs[key], ndarray) and \
               kwargs[key]._class is not cp.ndarray:
                return False

        _compatible_dtypes = [
            np.bool, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64,
            np.complex64, np.complex128
        ] + list('?bhilqBHILQefdFD')

        dtype = kwargs.get('dtype', None)
        if dtype is not None and dtype not in _compatible_dtypes:
            return False

        return True

    def __call__(self, *args, **kwargs):
        """
        Gets invoked when last attribute of _RecursiveAttr class gets called.
        Calls _cupy_object if not None else call _numpy_object.

        Args:
            args (tuple): Arguments.
            kwargs (dict): Keyword arguments.

        Returns:
            (res, ndarray): Returns of methods call_cupy or call_numpy
        """

        if not callable(self._numpy_object):
            raise TypeError("'{}' object is not callable".format(
                type(self._numpy_object).__name__))

        # _RecursiveAttr gets called from ndarray
        if self._fallback_array is not None:
            args = ((self._fallback_array,) + args)

        if self._cupy_object is not None and \
           _RecursiveAttr._is_cupy_compatible(args, kwargs):
            return _call_cupy(self._cupy_object, args, kwargs)

        return _call_numpy(self._numpy_object, args, kwargs)


numpy = _RecursiveAttr(np, cp)


# -----------------------------------------------------------------------------
# proxying of ndarray magic methods and wrappers
# -----------------------------------------------------------------------------


class ndarray(object):
    """
    Wrapper around cupy.ndarray
    Supports cupy.ndarray.__init__ as well as,
    gets initialized with a cupy ndarray.
    """

    def __new__(cls, *args, **kwargs):
        """
        If `_stored` and `_class` are arguments, initialize cls(ndarray).
        Else get cupy.ndarray from provided arguments,
        then initialize cls(ndarray).
        """
        _stored = kwargs.get('_stored', None)
        if _stored is not None:
            return object.__new__(cls)

        cupy_ndarray_init = cp.ndarray(*args, **kwargs)
        return cls(_stored=cupy_ndarray_init, _class=cp.ndarray)

    def __init__(self, *args, **kwargs):
        """
        Args:
            _stored (None, ndarray): If _stored is None, object is not
            initialized. Otherwise, _stored (ndarray) would be set to
            _cupy_array or _numpy_array depending upon _class.
            _class (ndarray type): If _class is `cp.ndarray`, _stored is
            set as _cupy_array. Otherwise, _stored is set as _numpy_array.
            Intended values for _class are `np.ndarray`, `np.ma.MaskedArray`,
            `np.matrix`, `np.chararray`, `np.recarray`.

        Attributes:
            _cupy_array (cp.ndarray): ndarray fully compatible with CuPy.
            This will be always set to ndarray in GPU.
            _numpy_array (np.ndarray and variants): ndarray not supported by
            CuPy. Such as np.ndarray(where dtype is not in '?bhilqBHILQefdFD')
            and it's variants. This will be always set to ndarray in CPU.
            _latest (str): If 'cupy', latest change may be in _cupy_array, but
            certainly not in _numpy_array. If 'numpy', latest change may be in
            _numpy_array but certainly not in _cupy_array.
            _class (ndarray type): If _class is `cp.ndarray`, data of array
            will contain in _cupy_array or _numpy_array (only if fallback
            occurs). In all other cases _numpy_array will have the data.
        """

        _class = kwargs.pop('_class', None)
        _stored = kwargs.pop('_stored', None)
        if _stored is None:
            return

        self._cupy_array = None
        self._numpy_array = None
        self._class = _class

        assert isinstance(_stored, (cp.ndarray, np.ndarray))
        if _class is cp.ndarray:
            self._cupy_array = _stored
            self._latest = 'cupy'
        else:
            self._numpy_array = _stored

    @classmethod
    def _store_array_from_cupy(cls, array):
        return cls(_stored=array, _class=cp.ndarray)

    @classmethod
    def _store_array_from_numpy(cls, array):
        if type(array) is np.ndarray and \
           array.dtype.kind in '?bhilqBHILQefdFD':
            return cls(_stored=cp.array(array), _class=cp.ndarray)

        return cls(_stored=array, _class=array.__class__)

    def __getattr__(self, attr):
        """
        Catches attributes corresponding to ndarray.

        Args:
            attr (str): Attribute of ndarray class.

        Returns:
            (_RecursiveAttr object, self._array.attr):
            Returns_RecursiveAttr object with numpy_object, cupy_object.
            Returns self._array.attr if attr is not callable.
        """

        if self._class is cp.ndarray:
            cupy_object = getattr(cp.ndarray, attr, None)
            numpy_object = getattr(np.ndarray, attr)
        else:
            cupy_object = None
            numpy_object = getattr(self._class, attr)

        if not callable(numpy_object):
            if self._class is cp.ndarray:
                return getattr(self._cupy_array, attr)
            return getattr(self._numpy_array, attr)

        return _RecursiveAttr(numpy_object, cupy_object, self)

    def _get_cupy_array(self):
        """
        Returns _cupy_array (cupy.ndarray) of ndarray object.
        """
        return self._cupy_array

    def _get_numpy_array(self):
        """
        Returns _numpy_array (ex: np.ndarray, numpy.ma.MaskedArray,
        numpy.chararray etc.) of ndarray object.
        """
        return self._numpy_array

    @classmethod
    def _cupy_dispatch(cls, array):
        """
        If _cupy_array is not latest, update it.
        """
        if array._latest != 'cupy':
            array._cupy_array = cp.array(array._numpy_array)
            array._latest = 'cupy'

    @classmethod
    def _numpy_dispatch(cls, array):
        """
        If _numpy_array is not latest, update it.
        """
        if array._class is cp.ndarray and array._latest != 'numpy':
            array._numpy_array = cp.asnumpy(array._cupy_array)
            array._latest = 'numpy'


def _create_magic_methods():
    """
    Set magic methods of cupy.ndarray as methods of fallback.ndarray.
    """

    # Decorator for ndarray magic methods
    def make_method(name):
        def method(self, *args, **kwargs):
            _method = getattr(self._class, name)
            args = ((self,) + args)
            if self._class is cp.ndarray:
                return _call_cupy(_method, args, kwargs)
            return _call_numpy(_method, args, kwargs)
        return method

    _common = [

        # Comparison operators:
        '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',

        # Unary operations:
        '__neg__', '__pos__', '__abs__', '__invert__',

        # Arithmetic:
        '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
        '__mod__', '__divmod__', '__pow__', '__lshift__', '__rshift__',
        '__and__', '__or__', '__xor__',

        # Arithmetic, in-place:
        '__iadd__', '__isub__', '__imul__', '__itruediv__', '__ifloordiv__',
        '__imod__', '__ipow__', '__ilshift__', '__irshift__',
        '__iand__', '__ior__', '__ixor__',

        # reflected-methods:
        '__radd__', '__rsub__', '__rmul__', '__rtruediv__', '__rfloordiv__',
        '__rmod__', '__rdivmod__', '__rpow__', '__rlshift__', '__rrshift__',
        '__rand__', '__ror__', '__rxor__',

        # For standard library functions:
        '__copy__', '__deepcopy__', '__reduce__',

        # Container customization:
        '__iter__', '__len__', '__getitem__', '__setitem__',

        # Conversion:
        '__int__', '__float__', '__complex__',

        # String representations:
        '__repr__', '__str__'
    ]

    _py3 = [
        '__matmul__', '__rmatmul__', '__bool__'
    ]

    _py2 = [
        '__div__', '__rdiv__', '__idiv__', '__nonzero__',
        '__long__', '__hex__', '__oct__'
    ]

    _specific = _py3
    if sys.version_info[0] == 2:
        _specific = _py2

    for method in _common + _specific:
        setattr(ndarray, method, make_method(method))


_create_magic_methods()


class vectorize(object):

    def __init__(self, *args, **kwargs):
        # NumPy will raise error if pyfunc is a cupy method
        if isinstance(args[0], _RecursiveAttr):
            args = (args[0]._numpy_object,) + args[1:]
        self.__dict__['vec_obj'] = np.vectorize(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.__dict__['vec_obj'], attr)

    def __setattr__(self, name, value):
        return setattr(self.vec_obj, name, value)

    @property
    def __doc__(self):
        return self.vec_obj.__doc__

    def __call__(self, *args, **kwargs):
        return _call_numpy(self.vec_obj, args, kwargs)


# -----------------------------------------------------------------------------
# Data Transfer methods
# -----------------------------------------------------------------------------


def _get_xp_args(ndarray_instance, to_xp, arg):
    """
    Converts ndarray_instance type object to target object using to_xp.

    Args:
        ndarray_instance (numpy.ndarray, cupy.ndarray or fallback.ndarray):
        Objects of type `ndarray_instance` will be converted using `to_xp`.
        to_xp (FunctionType): Method to convert ndarray_instance type objects.
        arg (object): `ndarray_instance`, `tuple`, `list` and `dict` type
        objects will be returned by either converting the object or it's
        elements, if object is iterable.
        Objects of other types is returned as it is.

    Returns:
        Return data structure will be same as before after converting ndarrays.
    """

    if isinstance(arg, ndarray_instance):
        return to_xp(arg)

    if isinstance(arg, tuple):
        return tuple([_get_xp_args(ndarray_instance, to_xp, x) for x in arg])

    if isinstance(arg, dict):
        return {x_name: _get_xp_args(ndarray_instance, to_xp, x)
                for x_name, x in arg.items()}

    if isinstance(arg, list):
        return [_get_xp_args(ndarray_instance, to_xp, x) for x in arg]

    return arg


def _convert_numpy_to_fallback(numpy_res):
    return _get_xp_args(np.ndarray, ndarray._store_array_from_numpy, numpy_res)


def _convert_fallback_to_numpy(args, kwargs):
    return _get_xp_args(ndarray, ndarray._get_numpy_array, (args, kwargs))


def _convert_fallback_to_cupy(args, kwargs):
    return _get_xp_args(ndarray, ndarray._get_cupy_array, (args, kwargs))


def _convert_cupy_to_fallback(cupy_res):
    return _get_xp_args(cp.ndarray, ndarray._store_array_from_cupy, cupy_res)


def _prepare_for_cupy_dispatch(args, kwargs):
    return _get_xp_args(ndarray, ndarray._cupy_dispatch, (args, kwargs))


def _prepare_for_numpy_dispatch(args, kwargs):
    return _get_xp_args(ndarray, ndarray._numpy_dispatch, (args, kwargs))

# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------


def _call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs and
    does necessary data transfers.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func and performing data transfers.
    """

    _prepare_for_cupy_dispatch(args, kwargs)
    cupy_args, cupy_kwargs = _convert_fallback_to_cupy(args, kwargs)
    cupy_res = func(*cupy_args, **cupy_kwargs)

    cupy_out = cupy_kwargs.get('out', None)
    if cupy_out is not None and cupy_out is cupy_res:
        return kwargs.get('out')

    return _convert_cupy_to_fallback(cupy_res)


def _call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs and
    does necessary data transfers.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func and performing data transfers.
    """

    _prepare_for_numpy_dispatch(args, kwargs)
    numpy_args, numpy_kwargs = _convert_fallback_to_numpy(args, kwargs)
    numpy_res = func(*numpy_args, **numpy_kwargs)

    # Sometimes reference to out(which was inplace updated) is returned
    # ex: numpy.add, numpy.nanmean
    # Therefore, to avoid creating separate ndarrays out is returned
    numpy_out = numpy_kwargs.get('out', None)
    if numpy_out is not None and numpy_out is numpy_res:
        return kwargs.get('out')

    return _convert_numpy_to_fallback(numpy_res)
