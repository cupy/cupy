import os


def get_cuda_requirements_path():
    return os.path.join(os.path.dirname(__file__), 'cuda-requirements.txt')


def get_cuda_requirements():
    with open(get_cuda_requirements_path()) as f:
        return f.read()
