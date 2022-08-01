# mypy: ignore-errors

import os
import threading


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


def generate_translation_unit(func_name, type_name, code_name, source_path):
    with open(source_path) as f:
        func_template = f.read()
        func_template = func_template.replace('<CODENAME>', code_name)
        func_template = func_template.replace('<TYPENAME>', type_name)
    base_name = os.path.basename(source_path).split('.')[0]
    full_path = f'{os.path.dirname(source_path)}/{base_name}_{code_name}.cu'
    with open(full_path, 'w') as f:
        f.write(func_template)
    return full_path


class ThreadWorker(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self._exception = e

    @property
    def exception(self):
        return self._exception
