#from libc.string cimport memcpy
cimport cpython
from numpy cimport (#dtype, PyArray_ArrFuncs,
                    PyArray_Descr, PyArray_DescrFromType)# PyArray_RegisterDataType)
cimport numpy

from numpy cimport import_array

import_array()


cdef extern from *:
    '''
    typedef struct {
        unsigned short x;
        unsigned short y;
    } _complex32;

    typedef struct {
        PyObject_HEAD
        _complex32 x;
    } _complex32_obj;

    static PyTypeObject _complex32_type = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "cupy.complex32",
        .tp_basicsize = sizeof(_complex32_obj),
        .tp_doc = "a complex number consisting of two float16",
    };

    static PyObject* PyComplex32_FromComplex32(_complex32 c) {
        printf("\\n\\n\\nI am here!!!!\\n\\n\\n"); fflush(stdout);
        _complex32_obj* p = (_complex32_obj*)_complex32_type.tp_alloc(&_complex32_type, 0);
        if (p) {
            p->x.x = c.x;
            p->x.y = c.y;
        }
        return (PyObject*)p;
    }
    
    static PyObject* complex32_getitem(void* data, void* arr) {
        printf("\\n\\n\\n getitem !!!!\\n\\n\\n");  fflush(stdout);
        _complex32 c;
        memcpy(&c, data, sizeof(_complex32));
        return PyComplex32_FromComplex32(c);
    }

    static int complex32_setitem(PyObject* item, void* data, void* arr) {
        printf("\\n\\n\\n setitem !!!!\\n\\n\\n");  fflush(stdout);
        _complex32 c;
        memcpy(data, &c, sizeof(_complex32));
        return 0;
    }

    static NPY_INLINE void byteswap(npy_half* x) {
        char* p = (char*)x;
        for (size_t i = 0; i < sizeof(*x)/2; i++) {
            size_t j = sizeof(*x)-1-i;
            char t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
    }

    static void complex32_copyswap(void* dst, void* src, int swap, void* arr) {
        _complex32* c;
        if (!src) {
            return;
        }
        c = (_complex32*)dst;
        memcpy(c, src, sizeof(_complex32));
        if (swap) {
            byteswap(&c->x);
            byteswap(&c->y);
        }
    }

    // minimal requirement to make dtype registration work
    // TODO: use PyArray_InitArrFuncs to initialize the struct
    static PyArray_ArrFuncs _complex32_arrfuncs = {
        .getitem = complex32_getitem,
        .setitem = complex32_setitem,
        .copyswap = complex32_copyswap,
    };

    static PyArray_Descr _complex32_dtype = {
        PyObject_HEAD_INIT(0)
        .typeobj = &_complex32_type,  // (PyTypeObject*)(&_complex32_type),
        .kind = 'c',
        .type = 'E',  // lowercase for fp16
        .byteorder = '=',  // native order
        .flags = NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
        .elsize = 4,
        .alignment = 4,
        //.names = {'real', 'imag'},
        .f = &_complex32_arrfuncs,
    };

    int register_dtype (void) {
        int is_ready = PyType_Ready((PyTypeObject*)(&_complex32_type));
        printf("is _complex32_type ready? %i\\n", is_ready);

        //((PyObject*)(&_complex32_dtype)) -> ob_refcnt = 1;
        //((PyObject*)(&_complex32_dtype)) -> ob_type = PyArray_Descr*;
        int _complex32_num = PyArray_RegisterDataType(&_complex32_dtype);
        return _complex32_num;
    }
    '''

    #ctypedef struct _complex32:
    #    pass
    #ctypedef struct _complex32_obj:
    #    pass
    cpython.PyTypeObject _complex32_type
    #PyArray_ArrFuncs _complex32_arrfuncs
    PyArray_Descr _complex32_dtype
    int register_dtype()


#import_ufunc()

#    struct _complex32_obj:
#        pass
#
#    cpython.PyTypeObject _complex32_type
#
#
#cdef class _complex32(dtype):
#
#    def __cinit__(self):
#        self.typeobj = <cpython.PyTypeObject*>(&_complex32_type)
#        self.kind = b'c'
#        self.type = b'E'  # lowercase for fp16
#        self.byteorder = b'='  # native order
#        self.itemsize = 4
#        self.subarray = NULL
#        self.names = (b'real', b'imag')

complex32_num = None
complex32 = None  #_complex32_type


def init():
    global complex32, complex32_num

    print("before registering...", flush=True)
    cdef int _complex32_num = register_dtype()
    print("after registering...", flush=True)
    complex32_num = _complex32_num
    print(_complex32_num)
    complex32 = PyArray_DescrFromType(_complex32_num)
