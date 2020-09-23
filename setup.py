#!/usr/bin/env python

import glob
import os
from setuptools import setup, find_packages
import sys

import cupy_setup_build


if len(os.listdir('cupy/core/include/cupy/cub/')) == 0:
    msg = '''
    The folder cupy/core/include/cupy/cub/ is a git submodule but is
    currently empty. Please use the command

        git submodule update --init

    to populate the folder before building from source.
    '''
    print(msg, file=sys.stderr)
    sys.exit(1)


if sys.version_info[:3] == (3, 5, 0):
    if not int(os.getenv('CUPY_PYTHON_350_FORCE', '0')):
        msg = """
CuPy does not work with Python 3.5.0.

We strongly recommend to use another version of Python.
If you want to use CuPy with Python 3.5.0 at your own risk,
set 1 to CUPY_PYTHON_350_FORCE environment variable."""
        print(msg)
        sys.exit(1)


requirements = {
    'setup': [
        'fastrlock>=0.3',
    ],
    'install': [
        'numpy>=1.15',
        'fastrlock>=0.3',
    ],
    'stylecheck': [
        'autopep8==1.4.4',
        'flake8==3.7.9',
        'pbr==4.0.4',
        'pycodestyle==2.5.0',
    ],
    'test': [
        'pytest<4.2.0',  # 4.2.0 is slow collecting tests and times out on CI.
        'attrs<19.2.0',  # pytest 4.1.1 does not run with attrs==19.2.0
    ],
    'doctest': [
        'matplotlib',
        'optuna',
    ],
    'docs': [
        'sphinx==3.0.4',
        'sphinx_rtd_theme',
    ],
    'travis': [
        '-r stylecheck',
        '-r docs',
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

here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, 'cupy', '_version.py')).read())

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
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


setup(
    name=package_name,
    version=__version__,  # NOQA
    description='CuPy: NumPy-like API accelerated with CUDA',
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
    python_requires='>=3.5.0',
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext,
              'sdist': sdist},
)
