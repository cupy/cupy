"""
`fallback_mode` for cupy. Whenever a method is not yet implemented in CuPy,
it will fallback to corresponding NumPy method.
"""
import types

import numpy as np

import cupy as cp


from cupyx.fallback_mode import notification


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
                Returns_RecursiveAttr object with new numpy_object,
                cupy_object. OR
                Returns objects in cupy which is an alias
                of numpy object. OR
                Returns wrapper objects, `ndarray`, `vectorize`.
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
    def _is_cupy_compatible(arg):
        """
        Returns False if CuPy's functions never accept the arguments as
        parameters due to the following reasons.
        - The inputs include an object of a NumPy's specific class other than
          `np.ndarray`.
        - The inputs include a dtype which is not supported in CuPy.
        """

        if isinstance(arg, ndarray):
            if not arg._supports_cupy:
                return False

        if isinstance(arg, (tuple, list)):
            return all(_RecursiveAttr._is_cupy_compatible(i) for i in arg)

        if isinstance(arg, dict):
            bools = [_RecursiveAttr._is_cupy_compatible(arg[i]) for i in arg]
            return all(bools)

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
           _RecursiveAttr._is_cupy_compatible((args, kwargs)):
            try:
                return _call_cupy(self._cupy_object, args, kwargs)
            except Exception:
                return _call_numpy(self._numpy_object, args, kwargs)

        notification._dispatch_notification(self._numpy_object)
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
        If `_initial_array` and `_supports_cupy` are arguments,
        initialize cls(ndarray).
        Else get cupy.ndarray from provided arguments,
        then initialize cls(ndarray).
        """
        _initial_array = kwargs.get('_initial_array', None)
        if _initial_array is not None:
            return object.__new__(cls)

        cupy_ndarray_init = cp.ndarray(*args, **kwargs)
        return cls(_initial_array=cupy_ndarray_init, _supports_cupy=True)

    def __init__(self, *args, **kwargs):
        """
        Args:
            _initial_array (None, cp.ndarray/np.ndarray(including variants)):
                If _initial_array is None, object is not initialized.
                Otherwise, _initial_array (ndarray) would be set to
                _cupy_array and/or _numpy_array depending upon _supports_cupy.
            _supports_cupy (bool): If _supports_cupy is True, _initial_array
                is set as _cupy_array and _numpy_array.
                Otherwise, _initial_array is set as only _numpy_array.

        Attributes:
            _cupy_array (None or cp.ndarray): ndarray fully compatible with
                CuPy. This will be always set to a ndarray in GPU.
            _numpy_array (None or np.ndarray(including variants)): ndarray not
                supported by CuPy. Such as np.ndarray (where dtype is not in
                '?bhilqBHILQefdFD') and it's variants. This will be always set
                to a ndarray in CPU.
            _supports_cupy (bool): If _supports_cupy is True, data of array
                will contain in _cupy_array and _numpy_array.
                Else only _numpy_array will have the data.
        """

        _supports_cupy = kwargs.pop('_supports_cupy', None)
        _initial_array = kwargs.pop('_initial_array', None)
        if _initial_array is None:
            return

        self._cupy_array = None
        self._numpy_array = None
        self.base = None
        self._supports_cupy = _supports_cupy

        assert isinstance(_initial_array, (cp.ndarray, np.ndarray))
        if _supports_cupy:
            if type(_initial_array) is cp.ndarray:
                # _initial_array is in GPU memory
                # called by _store_array_from_cupy
                self._cupy_array = _initial_array
                self._remember_numpy = False
            else:
                # _initial_array is in CPU memory
                # called by _store_array_from_numpy
                self._numpy_array = _initial_array
                self._remember_numpy = True
        else:
            self._numpy_array = _initial_array

    @classmethod
    def _store_array_from_cupy(cls, array):
        return cls(_initial_array=array, _supports_cupy=True)

    @classmethod
    def _store_array_from_numpy(cls, array):
        if type(array) is np.ndarray and \
           array.dtype.kind in '?bhilqBHILQefdFD':
            return cls(_initial_array=array, _supports_cupy=True)

        return cls(_initial_array=array, _supports_cupy=False)

    @property
    def dtype(self):
        if self._supports_cupy and not self._remember_numpy:
            return self._cupy_array.dtype
        return self._numpy_array.dtype

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

        if self._supports_cupy:
            cupy_object = getattr(cp.ndarray, attr, None)
            numpy_object = getattr(np.ndarray, attr)
        else:
            cupy_object = None
            numpy_object = getattr(self._numpy_array.__class__, attr)

        if not callable(numpy_object):
            if self._supports_cupy:
                if self._remember_numpy:
                    self._update_cupy_array()
                return getattr(self._cupy_array, attr)
            return getattr(self._numpy_array, attr)

        return _RecursiveAttr(numpy_object, cupy_object, self)

    def _get_cupy_array(self):
        """
        Returns _cupy_array (cupy.ndarray) of ndarray object. And marks
        self(ndarray) and it's base (if exist) as numpy not up-to-date.
        """
        base = self.base
        if base is not None:
            base._remember_numpy = False
        self._remember_numpy = False
        return self._cupy_array

    def _get_numpy_array(self):
        """
        Returns _numpy_array (ex: np.ndarray, numpy.ma.MaskedArray,
        numpy.chararray etc.) of ndarray object. And marks self(ndarray)
        and it's base (if exist) as numpy up-to-date.
        """
        base = self.base
        if base is not None and base._supports_cupy:
            base._remember_numpy = True
        if self._supports_cupy:
            self._remember_numpy = True
        return self._numpy_array

    def _update_numpy_array(self):
        """
        Updates _numpy_array from _cupy_array.
        To be executed before calling numpy function.
        """
        base = self.base
        _type = np.ndarray if self._supports_cupy \
            else self._numpy_array.__class__

        if self._supports_cupy:
            # cupy-compatible
            if base is None:
                if not self._remember_numpy:
                    if self._numpy_array is None:
                        self._numpy_array = cp.asnumpy(self._cupy_array)
                    else:
                        self._cupy_array.get(out=self._numpy_array)
            else:
                if not base._remember_numpy:
                    base._update_numpy_array()
                    if self._numpy_array is None:
                        self._numpy_array = base._numpy_array.view(type=_type)
                        self._numpy_array.shape = self._cupy_array.shape
                        self._numpy_array.strides = self._cupy_array.strides
        else:
            # not cupy-compatible
            if base is not None:
                assert base._supports_cupy
                if not base._remember_numpy:
                    base._update_numpy_array()

    def _update_cupy_array(self):
        """
        Updates _cupy_array from _numpy_array.
        To be executed before calling cupy function.
        """
        base = self.base

        if base is None:
            if self._remember_numpy:
                if self._cupy_array is None:
                    self._cupy_array = cp.array(self._numpy_array)
                else:
                    self._cupy_array[:] = self._numpy_array
        else:
            if base._remember_numpy:
                base._update_cupy_array()


def _create_magic_methods():
    """
    Set magic methods of cupy.ndarray as methods of fallback.ndarray.
    """

    # Decorator for ndarray magic methods
    def make_method(name):
        def method(self, *args, **kwargs):
            CLASS = cp.ndarray if self._supports_cupy \
                else self._numpy_array.__class__
            _method = getattr(CLASS, name)
            args = ((self,) + args)
            if self._supports_cupy:
                return _call_cupy(_method, args, kwargs)
            return _call_numpy(_method, args, kwargs)
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
        self.__dict__['_is_numpy_pyfunc'] = False
        self.__dict__['_cupy_support'] = False
        if isinstance(args[0], _RecursiveAttr):
            self.__dict__['_is_numpy_pyfunc'] = True
            if args[0]._cupy_object:
                self.__dict__['_cupy_support'] = True
            args = (args[0]._numpy_object,) + args[1:]
        notification._dispatch_notification(np.vectorize)
        self.__dict__['vec_obj'] = np.vectorize(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.__dict__['vec_obj'], attr)

    def __setattr__(self, name, value):
        return setattr(self.vec_obj, name, value)

    @property
    def __doc__(self):
        return self.vec_obj.__doc__

    def __call__(self, *args, **kwargs):
        if self._is_numpy_pyfunc:
            notification._dispatch_notification(
                self.vec_obj.pyfunc, self._cupy_support)
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
            elements, if object is iterable. Objects of other types is
            returned as it is.

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


def _update_numpy_args(args, kwargs):
    return _get_xp_args(ndarray, ndarray._update_numpy_array, (args, kwargs))


def _update_cupy_args(args, kwargs):
    return _get_xp_args(ndarray, ndarray._update_cupy_array, (args, kwargs))


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

    _update_cupy_args(args, kwargs)
    cupy_args, cupy_kwargs = _convert_fallback_to_cupy(args, kwargs)
    cupy_res = func(*cupy_args, **cupy_kwargs)

    # If existing argument is being returned
    ext_res = _get_same_reference(
        cupy_res, cupy_args, cupy_kwargs, args, kwargs)
    if ext_res is not None:
        return ext_res

    if isinstance(cupy_res, cp.ndarray):
        if cupy_res.base is None:
            # Don't share memory
            fallback_res = _convert_cupy_to_fallback(cupy_res)
        else:
            # Share memory with one of the arguments
            base_arg = _get_same_reference(
                cupy_res.base, cupy_args, cupy_kwargs, args, kwargs)
            fallback_res = _convert_cupy_to_fallback(cupy_res)
            fallback_res.base = base_arg
        return fallback_res
    return cupy_res


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

    _update_numpy_args(args, kwargs)
    numpy_args, numpy_kwargs = _convert_fallback_to_numpy(args, kwargs)
    numpy_res = func(*numpy_args, **numpy_kwargs)

    # If existing argument is being returned
    ext_res = _get_same_reference(
        numpy_res, numpy_args, numpy_kwargs, args, kwargs)
    if ext_res is not None:
        return ext_res

    if isinstance(numpy_res, np.ndarray):
        if numpy_res.base is None:
            # Don't share memory
            fallback_res = _convert_numpy_to_fallback(numpy_res)
        else:
            # Share memory with one of the arguments
            base_arg = _get_same_reference(
                numpy_res.base, numpy_args, numpy_kwargs, args, kwargs)
            fallback_res = _convert_numpy_to_fallback(numpy_res)
            fallback_res.base = base_arg
        return fallback_res
    return numpy_res


def _get_same_reference(res, args, kwargs, ret_args, ret_kwargs):
    """
    Returns object corresponding to res in (args, kwargs)
    from (ret_args, ret_kwargs)
    """
    for i in range(len(args)):
        if res is args[i]:
            return ret_args[i]

    for key in kwargs:
        if res is kwargs[key]:
            return ret_kwargs[key]

    return
