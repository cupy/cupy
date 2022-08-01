from typing import List, Mapping, Tuple, Union
import os


def get_cuda_source_data(source_root: str) \
        -> Mapping[str, Mapping[str, Union[str, Tuple[str, List[str]]]]]:
    # if not all dtypes are supported, the list of supported dtypes is returned
    return {
        'thrust': {
            'argsort': f'{source_root}/cupy/cuda/detail/cupy_thrust_argsort.template',  # noqa: E501
            'lexsort': f'{source_root}/cupy/cuda/detail/cupy_thrust_lexsort.template',  # noqa: E501
            'sort': f'{source_root}/cupy/cuda/detail/cupy_thrust_sort.template',  # noqa: E501
        },
        'cub': {
            'sum': f'{source_root}/cupy/cuda/detail/cupy_cub_device_reduce_sum.template',  # noqa: E501
            'prod': f'{source_root}/cupy/cuda/detail/cupy_cub_device_reduce_prod.template',  # noqa: E501
            'min': f'{source_root}/cupy/cuda/detail/cupy_cub_device_reduce_min.template',  # noqa: E501
            'max': f'{source_root}/cupy/cuda/detail/cupy_cub_device_reduce_max.template',  # noqa: E501
            'argmin': f'{source_root}/cupy/cuda/detail/cupy_cub_device_reduce_argmin.template',  # noqa: E501
            'argmax': f'{source_root}/cupy/cuda/detail/cupy_cub_device_reduce_argmax.template',  # noqa: E501
            's_sum': f'{source_root}/cupy/cuda/detail/cupy_cub_device_segmented_reduce_sum.template',  # noqa: E501
            's_prod': f'{source_root}/cupy/cuda/detail/cupy_cub_device_segmented_reduce_prod.template',  # noqa: E501
            's_min': f'{source_root}/cupy/cuda/detail/cupy_cub_device_segmented_reduce_min.template',  # noqa: E501
            's_max': f'{source_root}/cupy/cuda/detail/cupy_cub_device_segmented_reduce_max.template',  # noqa: E501
            'cumsum': f'{source_root}/cupy/cuda/detail/cupy_cub_device_scan_cumsum.template',  # noqa: E501
            'cumprod': f'{source_root}/cupy/cuda/detail/cupy_cub_device_scan_cumprod.template',  # noqa: E501
            'spmv': f'{source_root}/cupy/cuda/detail/cupy_cub_device_spmv.template',  # noqa: E501
            'hist_range': (f'{source_root}/cupy/cuda/detail/cupy_cub_device_histogram_range.template',  # noqa: E501
                           ['char', 'short', 'int', 'int64_t',
                            'unsigned char', 'unsigned short', 'unsigned int', 'uint64_t',  # noqa: E501
                            '__half', 'float', 'double',
                            'bool']),
            'hist_even': (f'{source_root}/cupy/cuda/detail/cupy_cub_device_histogram_even.template',  # noqa: E501
                          ['char', 'short', 'int', 'int64_t',
                           'unsigned char', 'unsigned short', 'unsigned int', 'uint64_t',  # noqa: E501
                           'bool']),
        }
    }


# TODO(leofang): use exact bit-width names (ex: int8_t instead of char)
cuda_type_to_code = {
    'char': 'CUPY_TYPE_INT8',
    'short': 'CUPY_TYPE_INT16',
    'int': 'CUPY_TYPE_INT32',
    'int64_t': 'CUPY_TYPE_INT64',
    'unsigned char': 'CUPY_TYPE_UINT8',
    'unsigned short': 'CUPY_TYPE_UINT16',
    'unsigned int': 'CUPY_TYPE_UINT32',
    'uint64_t': 'CUPY_TYPE_UINT64',
    '__half': 'CUPY_TYPE_FLOAT16',
    'float': 'CUPY_TYPE_FLOAT32',
    'double': 'CUPY_TYPE_FLOAT64',
    'complex<float>': 'CUPY_TYPE_COMPLEX64',
    'complex<double>': 'CUPY_TYPE_COMPLEX128',
    'bool': 'CUPY_TYPE_BOOL',
}


def generate_translation_unit(
        func_name, type_name, code_name, source_path) -> None:
    with open(source_path) as f:
        func_template = f.read()
        func_template = func_template.replace('<CODENAME>', code_name)
        func_template = func_template.replace('<TYPENAME>', type_name)
    base_name = os.path.basename(source_path).split('.')[0]
    full_path = f'{os.path.dirname(source_path)}/{base_name}_{code_name}.cu'
    with open(full_path, 'w') as f:
        f.write(func_template)
    print(f'generated {full_path}')


if __name__ == '__main__':
    source_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '../../..'))

    for mod, funcs in get_cuda_source_data(source_root).items():
        for func_name, template in funcs.items():
            if isinstance(template, str):
                template_path = template
                supported_types = list(cuda_type_to_code.keys())  # all supported
            else:
                template_path, supported_types = template
            for type_name in supported_types:
                code_name = cuda_type_to_code[type_name]
                generate_translation_unit(
                    func_name, type_name, code_name, template_path)
