from typing import Mapping
import os


def get_cuda_source_data(source_root: str) -> Mapping[str, Mapping[str, str]]:
    return {
        'thrust': {
            'argsort': f'{source_root}/cupy/cuda/detail/cupy_thrust_argsort.template',  # noqa: E501
            'lexsort': f'{source_root}/cupy/cuda/detail/cupy_thrust_lexsort.template',  # noqa: E501
            'sort': f'{source_root}/cupy/cuda/detail/cupy_thrust_sort.template',  # noqa: E501
        },
        'cub': {
        }
    }


# TODO(leofang): some functions only support a subset of this list
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
        for func_name, template_path in funcs.items():
            for type_name, code_name in cuda_type_to_code.items():
                generate_translation_unit(
                    func_name, type_name, code_name, template_path)
