# functions, orders, configurations
# The reference and the header have different orders, even different sections.
# https://docs.nvidia.com/cuda/cusparse/index.html
[
    # cuSPARSE Management Function
    ('Comment', 'cuSPARSE Management Function'),
    ('cusparseCreate', {
        'out': 'handle',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroy', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseGetVersion', {
        'out': 'version',
        'except': -1,
        'use_stream': False,
    }),
    ('cusparseSetPointerMode', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseGetStream', {
        'out': 'streamId',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSetStream', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Helper Function
    ('Comment', 'cuSPARSE Helper Function'),
    ('cusparseCreateMatDescr', {
        'out': 'descrA',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyMatDescr', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatDiagType', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatFillMode', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatIndexBase', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSetMatType', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrsv2Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrsv2Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrsm2Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrsm2Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsric02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsric02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrilu02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrilu02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateBsric02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyBsric02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateBsrilu02Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyBsrilu02Info', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCreateCsrgemm2Info', {
        'out': 'info',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyCsrgemm2Info', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Level 1 Function
    ('Comment', 'cuSPARSE Level 1 Function'),
    ('cusparse<t>gthr', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Level 2 Function
    ('Comment', 'cuSPARSE Level 2 Function'),
    ('cusparse<t>csrmv', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparseCsrmvEx_bufferSize', {
        'out': 'bufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparseCsrmvEx', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrsv2_bufferSize', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csrsv2_analysis', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrsv2_solve', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsrsv2_zeroPivot', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Level 3 Function
    ('Comment', 'cuSPARSE Level 3 Function'),
    ('cusparse<t>csrmm', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrmm2', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrsm2_bufferSizeExt', {
        'out': 'pBufferSize',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csrsm2_analysis', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrsm2_solve', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsrsm2_zeroPivot', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Extra Function
    ('Comment', 'cuSPARSE Extra Function'),
    ('cusparseXcsrgeamNnz', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrgeam', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrgeam2_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparseXcsrgeam2Nnz', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrgeam2', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsrgemmNnz', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrgemm', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrgemm2_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparseXcsrgemm2Nnz', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrgemm2', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Preconditioners - Incomplete Cholesky Factorization: level 0
    ('Comment', ('cuSPARSE Preconditioners - '
                 'Incomplete Cholesky Factorization: level 0')),
    ('cusparse<t>csric02_bufferSize', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csric02_analysis', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csric02', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsric02_zeroPivot', {
        'out': 'position',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>bsric02_bufferSize', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>bsric02_analysis', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>bsric02', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXbsric02_zeroPivot', {
        'out': 'position',
        'except?': 0,
        'use_stream': True,
    }),
    # cuSPARSE Preconditioners - Incomplete LU Factorization: level 0
    ('Comment', ('cuSPARSE Preconditioners - '
                 'Incomplete LU Factorization: level 0')),
    ('cusparse<t>csrilu02_numericBoost', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrilu02_bufferSize', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csrilu02_analysis', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csrilu02', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsrilu02_zeroPivot', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>bsrilu02_numericBoost', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>bsrilu02_bufferSize', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>bsrilu02_analysis', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>bsrilu02', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXbsrilu02_zeroPivot', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Preconditioners - Tridiagonal Solve
    ('Comment', 'cuSPARSE Preconditioners - Tridiagonal Solve'),
    ('cusparse<t>gtsv2_bufferSizeExt', {
        'out': 'bufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>gtsv2', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>gtsv2_nopivot_bufferSizeExt', {
        'out': 'bufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>gtsv2_nopivot', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Preconditioners - Batched Tridiagonal Solve
    ('Comment', 'cuSPARSE Preconditioners - Batched Tridiagonal Solve'),
    ('cusparse<t>gtsv2StridedBatch_bufferSizeExt', {
        'out': 'bufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>gtsv2StridedBatch', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>gtsvInterleavedBatch_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>gtsvInterleavedBatch', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Preconditioners - Batched Pentadiagonal Solve
    ('Comment', 'cuSPARSE Preconditioners - Batched Pentadiagonal Solve'),
    ('cusparse<t>gpsvInterleavedBatch_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>gpsvInterleavedBatch', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Reordering
    ('Comment', 'cuSPARSE Reorderings'),
    # cuSPARSE Format Conversion
    ('Comment', 'cuSPARSE Format Conversion'),
    ('cusparseXcoo2csr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csc2dense', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsr2coo', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csr2csc', {  # REMOVED
        'out': None,
        'use_stream': True,
    }),
    ('cusparseCsr2cscEx2_bufferSize', {
        'out': 'bufferSize',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseCsr2cscEx2', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>csr2dense', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>nnz_compress', {
        'out': 'nnzC',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparse<t>csr2csr_compress', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>dense2csc', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>dense2csr', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparse<t>nnz', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseCreateIdentityPermutation', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcoosort_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparseXcoosortByRow', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcoosortByColumn', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcsrsort_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparseXcsrsort', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseXcscsort_bufferSizeExt', {
        'out': 'pBufferSizeInBytes',
        'except?': 0,
        'use_stream': True,
    }),
    ('cusparseXcscsort', {
        'out': None,
        'use_stream': True,
    }),
    # cuSPARSE Generic API - Sparse Vector APIs
    ('Comment', 'cuSPARSE Generic API - Sparse Vector APIs'),
    ('cusparseCreateSpVec', {
        'out': 'spVecDescr',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroySpVec', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSpVecGet', {
        'out': ('SpVecAttributes',
                ('size', 'nnz', 'indices', 'values', 'idxType', 'idxBase',
                 'valueType')),
        'use_stream': False,
    }),
    ('cusparseSpVecGetIndexBase', {
        'out': 'idxBase',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpVecGetValues', {
        'out': 'values',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpVecSetValues', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Generic API - Sparse Matrix APIs
    ('Comment', 'cuSPARSE Generic API - Sparse Matrix APIs'),
    ('cusparseCreateCoo', {
        'out': 'spMatDescr',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseCreateCooAoS', {
        'out': 'spMatDescr',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseCreateCsr', {
        'out': 'spMatDescr',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroySpMat', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseCooGet', {
        'out': ('CooAttributes',
                ('rows', 'cols', 'nnz', 'cooRowInd', 'cooColInd', 'cooValues',
                 'idxType', 'idxBase', 'valueType')),
        'use_stream': False,
    }),
    ('cusparseCooAoSGet', {
        'out': ('CooAoSAttributes',
                ('rows', 'cols', 'nnz', 'cooInd', 'cooValues', 'idxType',
                 'idxBase', 'valueType')),
        'use_stream': False,
    }),
    ('cusparseCsrGet', {
        'out': ('CsrAttributes',
                ('rows', 'cols', 'nnz', 'csrRowOffsets', 'csrColInd',
                 'csrValues', 'csrRowOffsetsType', 'csrColIndType', 'idxBase',
                 'valueType')),
        'use_stream': False,
    }),
    ('cusparseSpMatGetFormat', {
        'out': 'format',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpMatGetIndexBase', {
        'out': 'idxBase',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpMatGetValues', {
        'out': 'values',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpMatSetValues', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseSpMatGetStridedBatch', {
        'out': 'batchCount',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpMatSetStridedBatch', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Generic API - Dense Vector APIs
    ('Comment', 'cuSPARSE Generic API - Dense Vector APIs'),
    ('cusparseCreateDnVec', {
        'out': 'dnVecDescr',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyDnVec', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseDnVecGet', {
        'out': ('DnVecAttributes',
                ('size', 'values', 'valueType')),
        'use_stream': False,
    }),
    ('cusparseDnVecGetValues', {
        'out': 'values',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDnVecSetValues', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Generic API - Dense Matrix APIs
    ('Comment', 'cuSPARSE Generic API - Dense Matrix APIs'),
    ('cusparseCreateDnMat', {
        'out': 'dnMatDescr',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDestroyDnMat', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseDnMatGet', {
        'out': ('DnMatAttributes',
                ('rows', 'cols', 'ld', 'values', 'type', 'order')),
        'use_stream': False,
    }),
    ('cusparseDnMatGetValues', {
        'out': 'values',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseDnMatSetValues', {
        'out': None,
        'use_stream': False,
    }),
    ('cusparseDnMatGetStridedBatch', {
        'out': ('DnMatBatchAttributes', ('batchCount', 'batchStride')),
        'use_stream': False,
    }),
    ('cusparseDnMatSetStridedBatch', {
        'out': None,
        'use_stream': False,
    }),
    # cuSPARSE Generic API - Generic API Functions
    ('Comment', 'cuSPARSE Generic API - Generic API Functions'),
    ('cusparseSpVV_bufferSize', {
        'out': 'bufferSize',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpVV', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseSpMV_bufferSize', {
        'out': 'bufferSize',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpMV', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseSpMM_bufferSize', {
        'out': 'bufferSize',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseSpMM', {
        'out': None,
        'use_stream': True,
    }),
    ('cusparseConstrainedGeMM_bufferSize', {
        'out': 'bufferSize',
        'except?': 0,
        'use_stream': False,
    }),
    ('cusparseConstrainedGeMM', {
        'out': None,
        'use_stream': True,
    }),
]
