"""
`fallback_mode` for cupy. Whenever a method is not yet implemented in CuPy,
it will fallback to corresponding NumPy method.
"""
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

        if self._cupy_object is not None:
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
        If `_stored` is an argument, initialize cls(ndarray).
        Else get cupy.ndarray from provided arguments,
        then initialize cls(ndarray).
        """
        _stored = kwargs.get('_stored', None)
        if _stored is not None:
            return object.__new__(cls)

        cupy_ndarray_init = cp.ndarray(*args, **kwargs)
        return cls(_stored=cupy_ndarray_init)

    def __init__(self, *args, **kwargs):
        _stored = kwargs.pop('_stored', None)
        if _stored is not None:
            self._array = _stored

    @classmethod
    def _store(cls, array):
        return cls(_stored=array)

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

        cupy_object = getattr(cp.ndarray, attr, None)
        numpy_object = getattr(np.ndarray, attr)

        if not callable(numpy_object):
            return getattr(self._array, attr)

        return _RecursiveAttr(numpy_object, cupy_object, self)

    def _get_array(self):
        """
        Returns _array (cupy.ndarray) of ndarray object.
        """
        return self._array


def _create_magic_methods():
    """
    Set magic methods of cupy.ndarray as methods of fallback.ndarray.
    """

    # Decorator for ndarray magic methods
    def make_method(name):
        def method(self, *args, **kwargs):
            args, kwargs = _get_cupy_args(args, kwargs)
            cupy_method = getattr(cp.ndarray, name)
            res = cupy_method(self._array, *args, **kwargs)
            return _get_fallback_result(res)
        return method

    for method in (
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
        '__matmul__',

        # reflected-methods:
        '__radd__', '__rsub__', '__rmul__', '__rtruediv__', '__rfloordiv__',
        '__rmod__', '__rdivmod__', '__rpow__', '__rlshift__', '__rrshift__',
        '__rand__', '__ror__', '__rxor__',
        '__rmatmul__',

        # For standard library functions:
        '__copy__', '__deepcopy__', '__reduce__',

        # Container customization:
        '__iter__', '__len__', '__getitem__', '__setitem__',

        # Conversion:
        '__bool__', '__int__', '__float__', '__complex__',

        # String representations:
        '__repr__', '__str__'
    ):
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


def _get_cupy_result(numpy_res):
    return _get_xp_args(np.ndarray, cp.array, numpy_res)


def _get_numpy_args(args, kwargs):
    return _get_xp_args(cp.ndarray, cp.asnumpy, (args, kwargs))


def _get_cupy_args(args, kwargs):
    return _get_xp_args(ndarray, ndarray._get_array, (args, kwargs))


def _get_fallback_result(cupy_res):
    return _get_xp_args(cp.ndarray, ndarray._store, cupy_res)


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

    args, kwargs = _get_cupy_args(args, kwargs)
    res = func(*args, **kwargs)

    return _get_fallback_result(res)


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

    args, kwargs = _get_cupy_args(args, kwargs)
    numpy_args, numpy_kwargs = _get_numpy_args(args, kwargs)
    numpy_res = func(*numpy_args, **numpy_kwargs)
    cupy_res = _get_cupy_result(numpy_res)

    return _get_fallback_result(cupy_res)
