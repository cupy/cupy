# mypy: ignore-errors

import os


def print_warning(*lines):
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')


def get_path(key):
    return os.environ.get(key, '').split(os.pathsep)


def search_on_path(filenames):
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


def generate_translation_unit(func_name, type_name, source_path):
    with open(source_path) as f:
        func_template = f.read()
    base_name = os.path.basename(source_path).split('.')[0]
    # both filename and function name cannot contain space, <, >
    type_name_fixed = type_name.replace(' ', '_').replace('<', '_').replace('>', '_')
    full_path = f'{os.path.dirname(source_path)}/{base_name}_{type_name_fixed}.cu'
    with open(full_path, 'w') as f:
        # TODO(leofang): come up with a better hack?
        func_template = func_template.replace('<TYPENAME>', type_name_fixed, 1)
        func_template = func_template.replace('<TYPENAME>', type_name)
        f.write(func_template)
    return full_path
