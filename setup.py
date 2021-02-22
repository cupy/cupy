#!/usr/bin/env python

import glob
import os
from setuptools import setup, find_packages
import sys

import cupy_setup_build


for submodule in ('cupy/core/include/cupy/cub/',
                  'cupy/core/include/cupy/jitify'):
    if len(os.listdir(submodule)) == 0:
        msg = '''
        The folder %s is a git submodule but is
        currently empty. Please use the command

            git submodule update --init

        to populate the folder before building from source.
        ''' % submodule
        print(msg, file=sys.stderr)
        sys.exit(1)


requirements = {
    'setup': [
        'fastrlock>=0.5',
    ],
    'install': [
        'numpy>=1.17',
        'fastrlock>=0.5',
    ],
    'all': [
        'scipy>=1.4',
        'optuna>=2.0',
    ],

    'stylecheck': [
        'autopep8==1.4.4',
        'flake8==3.7.9',
        'pbr==4.0.4',
        'pycodestyle==2.5.0',
    ],
    'test': [
        # 4.2 <= pytest < 6.2 is slow collecting tests and times out on CI.
        'pytest>=6.2',
    ],
    'appveyor': [
        '-r test',
    ],
    'jenkins': [
        '-r test',
        'pytest-timeout',
        'pytest-cov',
        'coveralls',
        'codecov',
        'coverage<5',  # Otherwise, Python must be built with sqlite
    ],
}


def reduce_requirements(key):
    # Resolve recursive requirements notation (-r)
    reqs = requirements[key]
    resolved_reqs = []
    for req in reqs:
        if req.startswith('-r'):
            depend_key = req[2:].lstrip()
            reduce_requirements(depend_key)
            resolved_reqs += requirements[depend_key]
        else:
            resolved_reqs.append(req)
    requirements[key] = resolved_reqs


for k in requirements.keys():
    reduce_requirements(k)


extras_require = {k: v for k, v in requirements.items() if k != 'install'}


setup_requires = requirements['setup']
install_requires = requirements['install']
tests_require = requirements['test']

# List of files that needs to be in the distribution (sdist/wheel).
# Notes:
# - Files only needed in sdist should be added to `MANIFEST.in`.
# - The following glob (`**`) ignores items starting with `.`.
cupy_package_data = [
    'cupy/cuda/cupy_thrust.cu',
    'cupy/cuda/cupy_cub.cu',
    'cupy/cuda/cupy_cufftXt.cu',  # for cuFFT callback
    'cupy/cuda/cupy_cufftXt.h',  # for cuFFT callback
    'cupy/cuda/cupy_cufft.h',  # for cuFFT callback
    'cupy/cuda/cufft.pxd',  # for cuFFT callback
    'cupy/cuda/cufft.pyx',  # for cuFFT callback
    'cupy/random/cupy_distributions.cu',
    'cupy/random/cupy_distributions.cuh',
] + [
    x for x in glob.glob('cupy/core/include/cupy/**', recursive=True)
    if os.path.isfile(x)
]

package_data = {
    'cupy': [
        os.path.relpath(x, 'cupy') for x in cupy_package_data
    ],
}

package_data['cupy'] += cupy_setup_build.prepare_wheel_libs()

package_name = cupy_setup_build.get_package_name()
long_description = cupy_setup_build.get_long_description()
ext_modules = cupy_setup_build.get_ext_modules()
build_ext = cupy_setup_build.custom_build_ext
sdist = cupy_setup_build.sdist_with_cython

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3 :: Only
Programming Language :: Cython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Microsoft :: Windows
"""


with open('VERSION') as f:
    version = f.read()

use_scm_version = {
    'write_to': 'cupy/_version.py',
    'write_to_template': "__version__ = '{version}'",
}
if os.environ.get('CUPY_RELEASE_BUILD', False):
    # setuptools-scm assumes that sdist/wheels are built *after* git-tagging, but
    # that conflicts with our workflow that builds release assets before the tagging.
    # The sdist/wheel build process must be done with this environment variable
    # to disable setuptools-scm and generate sdist/wheels versioned without commit hash.
    # Note that when a user is installing an sdist, verseion in PKG-INFO file (which is
    # bundled with the sdist tarball) is used and setuptools-scm does nothing.
    with open(use_scm_version['write_to'], 'w') as f:
        f.write(use_scm_version['write_to_template'].format(version=version))
    use_scm_version = None


setup(
    name=package_name,
    version=version,
    use_scm_version=use_scm_version,
    description='CuPy: A NumPy-compatible array library accelerated by CUDA',
    long_description=long_description,
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='https://cupy.dev/',
    license='MIT License',
    project_urls={
        "Bug Tracker": "https://github.com/cupy/cupy/issues",
        "Documentation": "https://docs.cupy.dev/",
        "Source Code": "https://github.com/cupy/cupy",
    },
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    packages=find_packages(exclude=['install', 'tests']),
    package_data=package_data,
    zip_safe=False,
    python_requires='>=3.6.0',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext,
              'sdist': sdist},
)
