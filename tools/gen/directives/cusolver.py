[
    # Setting
    ('Headers', ['cusolverDn.h', 'cusolverSp.h']),
    ('Regexes', {
        'func': r'cusolver(?:Dn|Sp|)([A-Z].*)',
        'type': r'(?:cusolver|cublas|cusparse)([A-Z].*)_t',  # uses some enums in cuBLAS and cuSPARSE
    }),
    ('SpecialTypes', {
        'cusolver_int_t': {
            'transpiled': 'int',
            'erased': 'int',
            'conversion': '{}',
        },
        'cuComplex': {
            'transpiled': 'cuComplex',
            'erased': 'size_t',  # should be `intptr_t`?
            'conversion': '(<cuComplex*>{})[0]',
        },
        'cuDoubleComplex': {
            'transpiled': 'cuDoubleComplex',
            'erased': 'size_t',  # should be `intptr_t`?
            'conversion': '(<cuDoubleComplex*>{})[0]',
        },
    }),
    # Library Attributes
    ('Comment', 'Library Attributes'),
    ('cusolverGetProperty', {
        'out': 'value',
        'except?': -1,
        'use_stream': False,
    }),
    ('Raw', '''cpdef tuple _getVersion():
    return (getProperty(MAJOR_VERSION),
            getProperty(MINOR_VERSION),
            getProperty(PATCH_LEVEL))'''),
    # cuSOLVER Dense LAPACK Function - Helper Function
    ('Comment', 'cuSOLVER Dense LAPACK Function - Helper Function'),
    ('cusolverDnCreate', {
        #'transpiled': 'dnCreate',
        'out': 'handle',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnDestroy', {
        #'transpiled': 'dnDestroy',
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnSetStream', {
        #'transpiled': 'dnSsetStream',
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnGetStream', {
        #'transpiled': 'dnGetStream',
        'out': 'streamId',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnCreateSyevjInfo', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnDestroySyevjInfo', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXsyevjSetTolerance', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXsyevjSetMaxSweeps', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXsyevjSetSortEig', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXsyevjGetResidual', {
        'out': 'residual',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnXsyevjGetSweeps', {
        'out': 'executed_sweeps',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnCreateGesvdjInfo', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnDestroyGesvdjInfo', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXgesvdjSetTolerance', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXgesvdjSetMaxSweeps', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXgesvdjSetSortEig', {
        'out': None,
        'use_stream': False,
    }),
    ('cusolverDnXgesvdjGetResidual', {
        'out': 'residual',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDnXgesvdjGetSweeps', {
        'out': 'executed_sweeps',
        'except?': 0,
        'use_stream': False,
    }),
    # cuSOLVER Dense LAPACK Function - Dense Linear Solver
    ('Comment', 'cuSOLVER Dense LAPACK Function - Dense Linear Solver'),
    ('cusolverDn<t>potrf_bufferSize', {
        'out': 'Lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>potrf', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>potrs', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>potrfBatched', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>potrsBatched', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>getrf_bufferSize', {
        'out': 'Lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>getrf', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>getrs', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>geqrf_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>geqrf', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>{or,un}gqr_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>{or,un}gqr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>{or,un}mqr_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>{or,un}mqr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>sytrf_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>sytrf', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t1><t2>gesv_bufferSize', {
        'out': 'lwork_bytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t1><t2>gesv', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t1><t2>gels_bufferSize', {
        'out': 'lwork_bytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t1><t2>gels', {
        'out': None,
        'use_stream': True,
    }),
    # cuSOLVER Dense LAPACK Function - Dense Eigenvalue Solver
    ('Comment', 'cuSOLVER Dense LAPACK Function - Dense Eigenvalue Solver'),
    ('cusolverDn<t>gebrd_bufferSize', {
        'out': 'Lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>gebrd', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvd_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvd', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvdj_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvdj', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvdjBatched_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvdjBatched', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>gesvdaStridedBatched_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDn<t>gesvdaStridedBatched', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>{sy,he}evd_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverDn<t>{sy,he}evd', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>{sy,he}evj_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>{sy,he}evj', {
        'out': None,
        'use_stream': True,
    }),
    ('cusolverDn<t>{sy,he}evjBatched_bufferSize', {
        'out': 'lwork',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusolverDn<t>{sy,he}evjBatched', {
        'out': None,
        'use_stream': True,
    }),
    # cuSOLVER Sparse LAPACK Function - Helper Function
    ('Comment', 'cuSOLVER Sparse LAPACK Function - Helper Function'),
    ('cusolverSpCreate', {
        'transpiled': 'spCreate',
        'out': 'handle',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusolverSpDestroy', {
        'transpiled': 'spDestroy',
        'out': None,
        'use_stream': False,
    }),
    ('cusolverSpSetStream', {
        'transpiled': 'spSetStream',
        'out': None,
        'use_stream': False,
    }),
    ('cusolverSpGetStream', {
        'transpiled': 'spGetStream',
        'out': 'streamId',
        'except?': 0,
        'use_stream': False,
    }),
    # cuSOLVER Sparse LAPACK Function - High Level Function
    ('Comment', 'cuSOLVER Sparse LAPACK Function - High Level Function'),
    ('cusolverSp<t>csrlsvchol', {
        'out': None,
        'use_stream': 'spSetStream',
    }),
    ('cusolverSp<t>csrlsvqr', {
        'out': None,
        'use_stream': 'spSetStream',
    }),
    ('cusolverSp<t>csreigvsi', {
        'out': None,
        'use_stream': 'spSetStream',
    }),
]
