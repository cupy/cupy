A tool to generate files for C extensions of CUDA-relatged libraries for CuPy. Currently covered are cuBLAS, cuSPARSE, and cuSOLVER, which have so many APIs to write their extensions by hands.

## Usage

###### Generate files for all of the libraries

```
./gen.sh
```

###### Generate files for a specific library

```
python gen_pyx.py <directive-file> <tempalte-file>       # Cython .pyx files
python gen_pxd.py <directive-file> <tempalte-file>       # Cython .pxd files
python gen_stub.py <directive-file> <tempalte-file>      # stub for Read the Docs
python gen_compat.py <directive-file> <tempalte-file>    # stub for CUDA version compatibility
```

Follow `gen.sh` for combinations of the directive and template files.

## Depencency

- pycparser(https://github.com/eliben/pycparser)
- C preprocessor - `cpp`

## Directive reference

### Directives

```
<directives> ::= <cuda-versions-directive>
                 <headers-directive>
                 <patterns-directive>
                 [<special-types-directive>]
                 {<comment-directive> | <function-directive>}*
```
The directives consist of a CUDA VERSIONS directive, a HEADERS directive, a PATTERNS directive, and an optional SPECIAL TYPES directive in this order, followed by multiple COMMENT directives and/or FUNCTION directives. The following shows the beginning part of the directives for generating files for cuBLAS.

```Python
[
    # Setting
    ('CudaVersions', ['11.2', '11.1', '11.0', '10.2', '10.0', '9.2']),
    ('Headers', ['cublas_v2.h']),
    ('Patterns', {
        'func': r'cublas([A-Z][^_]*)(:?_v2|)',
        'type': r'cublas([A-Z].*)_t',
    }),
    # cuBLAS Helper Function
    ('Comment', 'cuBLAS Helper Function'),
    ('cublasCreate_v2', {
        'out': 'handle',
        'except?': 0,
        'use_stream': False,
    }),
    ...
]
```

### CUDA VERSIONS directive

```
<cuda-versions-directive> ::= ('CudaVersions', [<cuda-version>, ...])
```

CUDA VERSIONS directive specifies the CUDA Toolkit versions from which this tool generates files for CuPy to call CUDA-related libraries. `<cuda-version>`s are CUDA Toolkit versions as strings and they have to be sorted in descending order. The following shows a CUDA VERSIONS directive.

```Python
('CudaVersions', ['11.2', '11.1', '11.0', '10.2', '10.1', '10.0', '9.2'])
```

### HEADERS directive

```
<headers-directive> ::= ('Headers', [<header-filename>, ...])
```

HEADERS directive specifies the names of the headers of the CUDA-related library for which this tool generates files with the directives. The following shows the HEADERS directive for cuSOLVER.

```Python
('Headers', ['cusolverDn.h', 'cusolverSp.h'])
```

### PATTERNS directive

```
<patterns-directive> ::= ('Patterns', {
    'func': <function-pattern>,
    'type': <type-pattern>
})
```

PATTERNS directive specifies patterns to map the names of CUDA-related libraries' APIs in C to those of CuPy's wrapper functions and related types in Python, represented in regular expressions. The value of `'func'` is for functions, and the value of `'type'` is for types including opaque pointers and enums.

The following shows the PATTERNS directive for cuSPARSE. The contents of the groups are used as the names in CuPy, e.g. `cusparseCreate` to `create`, and `cusparseHandle_t` to `Handle`

```Python
('Patterns', {
    'func': r'cusparse([A-Z].*)',
    'type': r'cusparse([A-Z].*)_t',
})
```

### SPECIAL TYPES directive

```
<speical-types-directive> ::= ('SpecialTypes', {
    <cuda-type-name>: {
        'transpiled': <transpiled-type-name>,
        'erased': <erased-type-name>,
        'conversion': <erased-to-transpiled-conversion-template>,
    },
    ...
})
```
SPECIAL TYPES directive specifies special rules that does not follow the PATTERN directive to map the names in CUDA-related librarys' types to those in CuPy. `<cuda-type-name>` is a type name in the C headers and `<transpiled-type-name>` is its counterpart transpiled into in the generated Cython files. `<erased-type-name>` is the name of its representative type used in the signatures of the generated wrapper functions. `<erased-to-transpiled-conversion-template>` is a template string to be formatted into a Cython code that converts a value of the 'transpiled' type to a value of the 'erased' type. `cudaDataType_t`, `cudaStream_t`, and some others are pre-defined as built-in ones.

The following shows a SPECIAL TYPES directive and a generated wrapper function for a C declaration uses `cuComplex`.

###### Directive for cuComplex

```Python
('SpecialTypes', {
    'cuComplex': {
        'transpiled': 'cuComplex',
        'erased': 'complex',
        'conversion': 'complex_to_cuda({var})',
    },
    ...
})
```

###### C Declaration using cuComplex

```C
cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csr_compress(cusparseHandle_t         handle,
                          int                      m,
                          int                      n,
                          const cusparseMatDescr_t descrA,
                          const cuComplex*         csrSortedValA,
                          const int*               csrSortedColIndA,
                          const int*               csrSortedRowPtrA,
                          int                      nnzA,
                          const int*               nnzPerRow,
                          cuComplex*               csrSortedValC,
                          int*                     csrSortedColIndC,
                          int*                     csrSortedRowPtrC,
                          cuComplex                tol);
```

###### Generated wrapper

```Cython
cpdef ccsr2csr_compress(intptr_t handle, int m, int n, size_t descrA, intptr_t csrSortedValA, intptr_t csrSortedColIndA, intptr_t csrSortedRowPtrA, int nnzA, intptr_t nnzPerRow, intptr_t csrSortedValC, intptr_t csrSortedColIndC, intptr_t csrSortedRowPtrC, complex tol):
    ...
    status = cusparseCcsr2csr_compress(<Handle>handle, m, n, <const MatDescr>descrA, <const cuComplex*>csrSortedValA, <const int*>csrSortedColIndA, <const int*>csrSortedRowPtrA, nnzA, <const int*>nnzPerRow, <cuComplex*>csrSortedValC, <int*>csrSortedColIndC, <int*>csrSortedRowPtrC, complex_to_cuda(tol))
    check_status(status)
```

### COMMENT directive

```
<comment-directive> ::= ('Comment', <comment>)
```

COMMENT directive specifies a comment line emitted in the place it appears in the order of the directives. The following shows a COMMENT directive followed by `cublasCreate` FUNCTION directive, and its generation in the .pyx file.

###### Directive

```Python
('Comment', 'cuBLAS Helper Function'),
('cublasCreate_v2', {
    ...
}),
```

###### Generated .pyx file

```Cython
########################################
# cuBLAS Helper Function

cpdef intptr_t create() except? 0:
    ...
```

### FUNCTION directive
```
<function-directive> ::= (<function-name>, {
  'transpiled': <transpiled-option>,
  'out': <out-option>,
  'except?': <except?-option>,
  'except': <except-option>,
  'use_stream': <use-stream-option>
})
```

FUNCTION directive specifies the CUDA function to generate the wrapper function and gives it some configuration for its generation process. It consists of a tuple with two elements that a string `<function-name>` and a dictionary that has some options for the configuration.

`<function-name>` is the name of the CUDA function to generate. We can use some convenient notations to make a FUNCTION directive match multiple CUDA functions:

- `<t>` for the data types, e.g. `cublasI<t>amax`
- `<t1><t2>` for the datatype precision and the internal lower precision, e.g. `cusolverDn<t1><t2>gesv`
- `{foo,bar}` like brace expansion for arbitrary strings in the function name, e.g. `cublas<t>dot{,u,c}`

The configuration dictionary's options are described in the subsections.

The following is `cublasCreate`'s directive, C declaration in its header, and generated wrapper function.

###### Directive for cublasCreate

```Python
('cublasCreate_v2', {
    'out': 'handle',
    'except?': 0,
    'use_stream': False,
})
```

###### C declaration for cublasCreate

```C
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCreate_v2 (cublasHandle_t *handle);
```

###### Generated wrapper for cublasCreate

```Cython
cpdef intptr_t create() except? 0:
    cdef Handle handle
    status = cublasCreate_v2(&handle)
    check_status(status)
    return <intptr_t>handle
```

#### TRANSPILED option

```
<transpiled-option> ::= <wrapper-function-name>
```

TRANSPILED option exceptionally gives the name of the wrapper function, `<wrapper-function-name>`, that does not follow the pattern specified in PATTERNS directive. TRANSPILED option is optional. The following shows `cusolverSpCreate`'s directive, C declaration in its header, and generated wrapper function.

###### Directive for cusolverSpCreate

```Python
('cusolverSpCreate', {
    'transpiled': 'spCreate',
    ...
})
```

###### C declaration for cusolverSpCreate

```C
cusolverStatus_t CUSOLVERAPI
cusolverSpCreate(cusolverSpHandle_t *handle);
```

###### Generated wrapper for cusolverSpCreate

```Cython
cpdef intptr_t spCreate() except? 0:
    cdef SpHandle handle
    status = cusolverSpCreate(&handle)
    check_status(status)
    return <intptr_t>handle
```

#### OUT option

```
<out-option> ::= None                                                   (* NONE-OUT case *)
               | 'Returned'                                             (* RETURNED-OUT case *)
               | <var-name>                                             (* SINGLE-OUT case *)
               | (<var-name>, ...) | (<class-name>, (<var-name>, ...))  (* MULTI-OUT case *)
```

OUT option gives if the wrapper function returns a value or not and what value it returns. It has three cases: NONE-OUT case, SINGLE-OUT case, and MULTI-OUT case. OUT option is required.

##### NONE-OUT case

When `None` is specified, the generated wrapper returns no value. The following shows `cublasI<t>amax`'s directive, declaration in its C header and generated wrapper function. 

###### Directive for cublasI\<t\>amax

```Python
('cublasI<t>amax_v2', {
    'out': None,
    ...
})
```

###### C declaration for cublasI\<t\>amax

```C
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasIsamax_v2(cublasHandle_t handle,
                int n,
                const float *x,
                int incx,
                int *result);
```

###### Generated wrapper for cublasI\<t\>amax

```Cython
cpdef isamax(intptr_t handle, int n, intptr_t x, int incx, intptr_t result):
    ...
    status = cublasIsamax_v2(<Handle>handle, n, <const float*>x, incx, <int*>result)
    check_status(status)
```

##### RETURNED-OUT case

When `'Returned'` is specified, the generated wrapper returns the returned value of the CUDA function directly.

##### SINGLE-OUT case

When a string `<var-name>` is specified, the generated wrapper returns a value set by an output parameter whose name is `<var-name>` in the CUDA function. The following shows `cusparseCreate`'s directive, declaration in its C header, and generated wrapper function.

###### Directive for cusparseCreate

```Python
('cusparseCreate', {
    'out': 'handle',
    ...
})
```

###### C declaration for cusparseCreate

```C
cusparseStatus_t CUSPARSEAPI
cusparseCreate(cusparseHandle_t* handle);
```

###### Generated wrapper for cusparseCreate

```Cython
cpdef intptr_t create() except? 0:
    cdef Handle handle
    status = cusparseCreate(&handle)
    check_status(status)
    return <intptr_t>handle
```

##### MULTI-OUT case

When a tuple of strings is specified, the generated wrapper returns a tuple of values set by several output parameters in the CUDA function. The specified tuple's strings determine the output parameters to use by their names.

MULTI-OUT case has another form: a tuple with two elements that a string `<class-name>` and a tuple of strings. With this form, the generated wrapper returns values contained by a helper class whose name is the given `<class-name>`. The definition of the helper class is also generated automatically. The following shows `cusparseSpVecGet`'s directive, declaration in its C header, and generated wrapper function.

###### Directive for cusparseSpVecGet

```Python
('cusparseSpVecGet', {
    'out': ('SpVecAttributes',
            ('size', 'nnz', 'indices', 'values', 'idxType', 'idxBase',
             'valueType')),
    ...
})
```

###### C declaration for cusparseSpVecGet

```C
cusparseStatus_t CUSPARSEAPI
cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr,
                 int64_t*             size,
                 int64_t*             nnz,
                 void**               indices,
                 void**               values,
                 cusparseIndexType_t* idxType,
                 cusparseIndexBase_t* idxBase,
                 cudaDataType*        valueType);
```

###### Generated wrapper for cusparseSpVecGet

```Cython
cpdef SpVecAttributes spVecGet(size_t spVecDescr):
    cdef int64_t size
    cdef int64_t nnz
    cdef void* indices
    cdef void* values
    cdef IndexType idxType
    cdef IndexBase idxBase
    cdef DataType valueType
    status = cusparseSpVecGet(<SpVecDescr>spVecDescr, &size, &nnz, &indices, &values, &idxType, &idxBase, &valueType)
    check_status(status)
    return SpVecAttributes(size, nnz, <intptr_t>indices, <intptr_t>values, idxType, idxBase, valueType)
```

#### EXCEPT? and EXCEPT options

```
<except?-option> ::= <exception-return-value>
<except-option> ::= <exception-return-value>
```

EXCEPT? and EXCEPT options give if the wrapper function has exception value declaration in its `cpdef` statement to propagatge exceptions that occur inside it. They correspond to `except?` and `except` declaration respectively. EXCEPT? and EXCEPT options are required only when the directive's OUT option is set to SINGLE-OUT case. Otherwise, they must not be specified. The following shows `cublasCreate`'s directive, C declaration in its header, and generated wrapper function.

###### Directive for cublasCreate

```Python
('cublasCreate_v2', {
    'out': 'handle',
    'except?': 0,
    ...
})
```

###### C declaration for cublasCreate

```C
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasCreate_v2 (cublasHandle_t *handle);
```

###### Generated wrapper for cublasCreate

```Cython
cpdef intptr_t create() except? 0:
    ...
```

#### USE_STREAM option

```
<use-stream-option> ::= False                                  (* DONT-USE case *)
                      | 'set' | ('set', <stream-setter-name>)  (* SET case *)
                      | 'pass'                                 (* PASS case *)
```

USE_STREAM option gives if the wrapper function uses the current stream on calling the CUDA function or not and how the CUDA function takes the stream. It has three cases: DONT-USE case, SET case, and PASS case. USE_STREAM option is required.

##### DONT-USE case

When `False` is specified, the generated wrapper does nothing for the stream.

##### SET case

When `'set'` is specified, the generated wrapper sets the current stream before it calls the CUDA function. The following shows `cublas<t>axpy`'s directive, declaration in its C header and generated wrapper function.

###### Directive for cublas\<t\>axpy

```Python
('cublas<t>axpy_v2', {
    ...
    'use_stream': 'set',
})
```

###### C declaration for cublas\<t\>axpy

```C
CUBLASAPI cublasStatus_t CUBLASWINAPI
cublasSaxpy_v2 (cublasHandle_t handle,
                int n,
                const float *alpha,
                const float *x,
                int incx,
                float *y,
                int incy);
```

###### Generated wrapper for cublas\<t\>axpy

```Cython
cpdef saxpy(intptr_t handle, int n, intptr_t alpha, intptr_t x, int incx, intptr_t y, int incy):
    if stream_module.enable_current_stream:
        setStream(handle, stream_module.get_current_stream_ptr())
    status = cublasSaxpy_v2(<Handle>handle, n, <const float*>alpha, <const float*>x, incx, <float*>y, incy)
    check_status(status)
```

SET case has another form: a tuple with two elements that `'set'` and a string `<stream-setter-name>`. With this form, the generated wrapper sets the current stream before it calls the CUDA function as the first form except that `<stream-setter-name>` is used for the name of the function that sets the stream. The following shows `cusolverSp<t>csrlsvchol`'s directive, declaration in its C header, and generated wrapper function.

###### Directive for cusolverSp\<t\>csrlsvchol

```Python
('cusolverSp<t>csrlsvchol', {
    ...
    'use_stream': ('set', 'spSetStream'),
})
```

###### C header for cusolverSp\<t\>csrlsvchol

```C
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);
```

###### Generated wrapper for cusolverSp\<t\>csrlsvchol

```Cython
cpdef scsrlsvchol(intptr_t handle, int m, int nnz, size_t descrA, intptr_t csrVal, intptr_t csrRowPtr, intptr_t csrColInd, intptr_t b, float tol, int reorder, intptr_t x, intptr_t singularity):
    if stream_module.enable_current_stream:
        spSetStream(handle, stream_module.get_current_stream_ptr())
    status = cusolverSpScsrlsvchol(<SpHandle>handle, m, nnz, <const MatDescr>descrA, <const float*>csrVal, <const int*>csrRowPtr, <const int*>csrColInd, <const float*>b, tol, reorder, <float*>x, <int*>singularity)
    check_status(status)
```

##### PASS case

When `'pass'` is specified, the generated wrapper passes the current stream to the CUDA function as one of its parameters. The CUDA function that takes `'pass'` for USE_STREAM option should have a parameter typed `cudaStream_t`.
