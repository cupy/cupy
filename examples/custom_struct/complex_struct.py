import sys
import numpy
import cupy

struct_definition = '''
struct complex_struct {
    int4 a;
    char b;
    double c[2];
    short1 d;
    unsigned long long int e[3];
};
'''

struct_layout_code = '''
{struct_definition}

extern "C" __global__ void get_struct_layout(
                                unsigned long long *itemsize,
                                unsigned long long *sizes,
                                unsigned long long *offsets) {{
    const complex_struct* ptr = nullptr;

    itemsize[0] = sizeof(complex_struct);

    sizes[0] = sizeof(ptr->a);
    sizes[1] = sizeof(ptr->b);
    sizes[2] = sizeof(ptr->c);
    sizes[3] = sizeof(ptr->d);
    sizes[4] = sizeof(ptr->e);

    offsets[0] = (unsigned long long)&ptr->a;
    offsets[1] = (unsigned long long)&ptr->b;
    offsets[2] = (unsigned long long)&ptr->c;
    offsets[3] = (unsigned long long)&ptr->d;
    offsets[4] = (unsigned long long)&ptr->e;
}}
'''.format(struct_definition=struct_definition)


kernel_code = '''
{struct_definition}

extern "C" __global__ void test_kernel(const complex_struct s,
                                       double* out) {{
    int i = threadIdx.x;
    double sum = 0.0;
    sum += s.a.x + s.a.y + s.a.z + s.a.w;
    sum += s.b;
    sum += s.c[0] + s.c[1];
    sum += s.d.x;
    sum += s.e[0] + s.e[1] + s.e[2];
    out[i] = i * sum;
}}
'''.format(struct_definition=struct_definition)


def make_packed(basetype, N, itemsize):
    # A small utility function to make packed structs
    # Can represent simple packed vectors such as float4 or double[3].
    assert 0 < N <= 4, N
    names = list('xyzw')[:N]
    formats = [basetype]*N
    return numpy.dtype(dict(names=names,
                            formats=formats,
                            itemsize=itemsize))


def main():
    # This program demonstrate how to build a hostside
    # representation of device structure 'complex_struct'
    # defined in variable 'struct_definition' that can be
    # used as a RawKernel argument.

    # First step is to determine structure memory layout
    #  itemsize -> overall struct size
    #  sizes    -> individual struct member sizes, determined with sizeof
    #  offsets  -> individual struct member offsets, determined with offsetof
    # Results (in terms of bytes) are copied to host after kernel launch.
    # Note that 'complex_struct' has 5 members named a, b, c, d and e.
    itemsize = cupy.ndarray(shape=(1,), dtype=numpy.uint64)
    sizes = cupy.ndarray(shape=(5,), dtype=numpy.uint64)
    offsets = cupy.ndarray(shape=(5,), dtype=numpy.uint64)

    kernel = cupy.RawKernel(
        struct_layout_code, 'get_struct_layout', options=('--std=c++11',))
    kernel((1,), (1,), (itemsize, sizes, offsets))

    (itemsize, sizes, offsets) = map(cupy.asnumpy, (itemsize, sizes, offsets))
    print("Overall structure itemsize: {} bytes".format(itemsize.item()))
    print("Structure members itemsize: {}".format(sizes))
    print("Structure members offsets: {}".format(offsets))

    # Second step: build a numpy dtype for each struct member
    atype = make_packed(numpy.int32,   4, sizes[0])
    btype = make_packed(numpy.int8,    1, sizes[1])
    ctype = make_packed(numpy.float64, 2, sizes[2])
    dtype = make_packed(numpy.int16,   1, sizes[3])
    etype = make_packed(numpy.uint64,  3, sizes[4])

    # Third step: create the complex struct representation with
    #  the right offsets
    names = list('abcde')
    formats = [atype, btype, ctype, dtype, etype]
    complex_struct = numpy.dtype(dict(names=names,
                                      formats=formats,
                                      offsets=offsets,
                                      itemsize=itemsize.item()))

    # Build a complex_struct kernel argument
    s = numpy.empty(shape=(1,), dtype=complex_struct)
    s['a'] = numpy.arange(0, 4).astype(numpy.int32).view(atype)
    s['b'] = numpy.arange(4, 5).astype(numpy.int8).view(btype)
    s['c'] = numpy.arange(5, 7).astype(numpy.float64).view(ctype)
    s['d'] = numpy.arange(7, 8).astype(numpy.int16).view(dtype)
    s['e'] = numpy.arange(8, 11).astype(numpy.uint64).view(etype)
    print("Complex structure value:\n  {}".format(s))

    # Setup test kernel
    N = 8
    out = cupy.empty(shape=(N,), dtype=numpy.float64)
    kernel = cupy.RawKernel(kernel_code, 'test_kernel')
    kernel((1,), (N,), (s, out))

    # the sum of all members of our complex struct instance is 55.0
    expected = cupy.arange(N) * 55.0

    cupy.testing.assert_array_almost_equal(expected, out)
    print("Kernel output matches expected value.")


if __name__ == '__main__':
    sys.exit(main())
