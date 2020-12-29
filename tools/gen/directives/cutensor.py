[
    # Setting
    ('CudaVersions', ['11.0']),
    ('Headers', ['cutensor.h']),
    ('Patterns', {
        'func': r'cutensor([A-Z].*)',
        'type': r'cutensor([A-Z].*)_t',
    }),
    ('SpecialTypes', {
        'cutensorHandle_t*': {
            'transpiled': 'cutensorHandle_t*',
            'erased': 'Handle',
            'conversion': '<{quals}cutensorHandle_t*>{var}._ptr',
        },
        'cutensorTensorDescriptor_t*': {
            'transpiled': 'cutensorTensorDescriptor_t*',
            'erased': 'TensorDescriptor',
            'conversion': '<{quals}cutensorTensorDescriptor_t*>{var}._ptr',
        },
        'cutensorContractionDescriptor_t*': {
            'transpiled': 'cutensorContractionDescriptor_t*',
            'erased': 'ContractionDescriptor',
            'conversion': (
                '<{quals}cutensorContractionDescriptor_t*>{var}._ptr'),
        },
        'cutensorContractionPlan_t*': {
            'transpiled': 'cutensorContractionPlan_t*',
            'erased': 'ContractionPlan',
            'conversion': '<{quals}cutensorContractionPlan_t*>{var}._ptr',
        },
        'cutensorContractionFind_t*': {
            'transpiled': 'cutensorContractionFind_t*',
            'erased': 'ContractionFind',
            'conversion': '<{quals}cutensorContractionFind_t*>{var}._ptr',
        },
    }),
    # cuTENSOR Helper Functions
    ('Comment', 'cuTENSOR Helper Functions'),
    ('cutensorInit', {
        'out': None,
        'use_stream': False,
    }),
    ('cutensorInitTensorDescriptor', {
        'out': None,
        'use_stream': False,
    }),
    ('cutensorGetAlignmentRequirement', {
        'out': 'alignmentRequirement',
        'except?': 0,
        'use_stream': False,
    }),
    # cuTENSOR Element-wise Operations
    ('Comment', 'cuTENSOR Element-wise Operations'),
    ('cutensorElementwiseTrinary', {
        'out': None,
        'use_stream': 'pass',
    }),
    ('cutensorElementwiseBinary', {
        'out': None,
        'use_stream': 'pass',
    }),
    # cuTENSOR Contraction Operations
    ('Comment', 'cuTENSOR Contraction Operations'),
    ('cutensorInitContractionDescriptor', {
        'out': None,
        'use_stream': False,
    }),
    ('cutensorInitContractionFind', {
        'out': None,
        'use_stream': False,
    }),
    ('cutensorInitContractionPlan', {
        'out': None,
        'use_stream': False,
    }),
    ('cutensorContraction', {
        'out': None,
        'use_stream': 'pass',
    }),
    ('cutensorContractionGetWorkspace', {
        'out': 'workspaceSize',
        'except?': 0,
        'use_stream': False,
    }),
    ('cutensorContractionMaxAlgos', {
        'out': 'maxNumAlgos',
        'except?': 0,
        'use_stream': False,
    }),
    # cuTENSOR Reduction Operations
    ('Comment', 'cuTENSOR Reduction Operations'),
    ('cutensorReduction', {
        'out': None,
        'use_stream': 'pass',
    }),
    ('cutensorReductionGetWorkspace', {
        'out': 'workspaceSize',
        'except?': 0,
        'use_stream': False,
    }),
]
