"""
Main fallback class and helper functions. Also numpy is a instance of Fallback class.

TODO: Capturing function with __getattribute__() OK??
"""
from types import FunctionType, ModuleType, MethodType, BuiltinFunctionType, BuiltinMethodType
import numpy as np
import cupy as cp

attr_list = []


class FallbackUtil:

    notifications = True

    @property
    @classmethod
    def notification_status(cls):
        return cls.notifications

    @classmethod
    def get_notification_status(cls):
        print("Notification status is currently {}".format(cls.notifications))

    @classmethod
    def set_notification_status(cls, status):
        cls.notifications = status
        print("Notification status is now {}".format(cls.notifications))

    @classmethod
    def create_fallback_object(cls, name):
        """
        Not being used. I implemented it for another approach.
        """
        exec(name + ' = FallbackUtil()')
        return eval(name)

    @classmethod
    def search_and_setattr(cls, set_from, module):
        """
        Currently not being used anywhere. Ran into recursion limit exceeded.
        """
        for attr_name, attr in set_from.__dict__.items():

            if isinstance(attr, FunctionType):
                setattr(module, attr_name, attr)

            elif isinstance(attr, ModuleType):
                sub_module = cls.create_fallback_object(attr_name)
                cls.search_and_setattr(getattr(set_from, attr_name), sub_module)
                setattr(module, attr_name, sub_module)

    @classmethod
    def search_and_getattr(cls, name, get_from, primary=True):
        """
        Currently not being used anywhere. Ran into recursion limit exceeded.
        """
        if hasattr(get_from, '__dict__'):
            for attr_name, attr in get_from.__dict__.items():
                if name == attr_name:
                    return attr

                elif isinstance(attr, ModuleType):
                    cls.search_and_getattr(name, attr, primary=False)

        if primary:
            return AttributeError

    @classmethod
    def clear_attrs(cls):
        """
        Initializes new attr_list.
        """
        global attr_list
        attr_list = []

    @classmethod
    def add_attrs(cls, attr):
        """
        Add given attr to attr_list
        """
        global attr_list
        attr_list.append(attr)

    @classmethod
    def join_attrs(cls):
        global attr_list
        path = ".".join(attr_list)
        return path

    @classmethod
    def get_last_and_rest(cls):
        """
        Provides sub-module and function name using attr_list
        """
        global attr_list
        path = ".".join(attr_list[:-1])
        return path, attr_list[-1]


class Fallback(FallbackUtil):

    def __getattribute__(self, attr):
        """
        All operations of fallback_mode will be done in this area.
        Currently it just supports finding appropriate function.
        
        Gets called by Recursive_attr object if function definition is found.
        """
        numpy_func = None
        cupy_func = None
        sub_module, func_name = FallbackUtil.get_last_and_rest()
        FallbackUtil.clear_attrs()

        # trying cupy
        try:
            if sub_module == '':
                cupy_path = 'cp'
            else:
                cupy_path = 'cp' + '.' + sub_module
            cupy_func = getattr(eval(cupy_path), func_name)

        except AttributeError:
            # trying numpy
            if FallbackUtil.notifications:
                print("no attribute '{}.{}' found in cupy. Falling back to numpy".format(sub_module, func_name))
            if sub_module == '':
                numpy_path = 'np'
            else:
                numpy_path = 'np' + '.' + sub_module
            numpy_func = getattr(eval(numpy_path), func_name)

        print("performing other fallback steps")

        if cupy_func is not None:
            return cupy_func
        elif numpy_func is not None:
            return numpy_func


class Recursive_attr:

    def __getattribute__(self, attr):
        """
        Helps in creating attr_list and finding apporiate function.
        Once function definition is found, calls Fallback object now_fallback.
        """
        FallbackUtil.add_attrs(attr)
        sub_module, func_name = FallbackUtil.get_last_and_rest()
        try:
            if sub_module == '':
                path = 'np'
            else:
                path = 'np' + '.' + sub_module
            func = getattr(eval(path), func_name)
        except AttributeError as error:
            FallbackUtil.clear_attrs()
            raise error
        if isinstance(func, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)):
            return now_fallback.attr
        elif isinstance(func, ModuleType):
            exec(attr + " = Recursive_attr()")
            return eval(attr)
        else:
            raise AttributeError("neither FunctionType nor ModuleType")


numpy = Recursive_attr()

now_fallback = Fallback()
