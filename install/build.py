import distutils
import os
import shutil
import subprocess
import tempfile

import setuptools

from install import utils


minimum_cuda_version = 6050
minimum_cudnn_version = 2000


def check_cuda_version(compiler, settings):
    out = build_and_run(compiler, '''
    #include <cuda.h>
    #include <stdio.h>
    int main(int argc, char* argv[]) {
      printf("%d", CUDA_VERSION);
      return 0;
    }
    ''', include_dirs=settings['include_dirs'])

    if out is None:
        utils.print_warning('Cannot check CUDA version')
        return False

    cuda_version = int(out)

    if cuda_version < minimum_cuda_version:
        utils.print_warning(
            'CUDA version is too old: %d' % cuda_version,
            'CUDA v6.5 or newer is required')
        return False

    return True


def check_cudnn_version(compiler, settings):
    out = build_and_run(compiler, '''
    #include <cudnn.h>
    #include <stdio.h>
    int main(int argc, char* argv[]) {
      printf("%d", CUDNN_VERSION);
      return 0;
    }
    ''', include_dirs=settings['include_dirs'])

    if out is None:
        utils.print_warning('Cannot check cuDNN version')
        return False

    cudnn_version = int(out)

    if cudnn_version < minimum_cudnn_version:
        utils.print_warning(
            'cuDNN version is too old: %d' % cudnn_version,
            'cuDNN v2 or newer is required')
        return False

    return True


def build_and_run(compiler, source, libraries=[],
                  include_dirs=[], library_dirs=[]):
    temp_dir = tempfile.mkdtemp()

    try:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)

        try:
            objects = compiler.compile([fname], output_dir=temp_dir,
                                       include_dirs=include_dirs)
        except distutils.errors.CompileError:
            return None

        try:
            postargs = ['/MANIFEST'] if sys.platform == 'win32' else []
            compiler.link_executable(objects,
                                     os.path.join(temp_dir, 'a'),
                                     libraries=libraries,
                                     library_dirs=library_dirs,
                                     extra_postargs=postargs)
        except (distutils.errors.LinkError, TypeError):
            return None

        try:
            out = subprocess.check_output(os.path.join(temp_dir, 'a'))
            return out

        except Exception:
            return None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
