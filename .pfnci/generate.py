#!/usr/bin/env python3

import argparse
import os
import sys

import yaml

from typing import Any, Dict, List


SchemaType = Dict[str, Any]


class Matrix:
    def __init__(self, record: Dict[str, str]):
        self._rec = record

    def __getattr__(self, key):
        return self._rec[key]


class LinuxGenerator:
    def __init__(self, schema: SchemaType, matrix: Matrix):
        assert matrix.system == 'linux'
        self.schema = schema
        self.matrix = matrix

    def generate_dockerfile(self) -> str:
        matrix = self.matrix
        lines = [
            '# AUTO GENERATED: DO NOT EDIT!',
        ]

        os_name, os_version = matrix.os.split(':')
        if matrix.cuda is not None:
            full_ver = self.schema['cuda'][matrix.cuda]['full_version']
            base_image = f'nvidia/cuda:{full_ver}-devel-{os_name}{os_version}'
        elif matrix.rocm is not None:
            full_ver = self.schema['rocm'][matrix.rocm]['full_version']
            base_image = f'rocm/dev-{os_name}-{os_version}:{full_ver}'
        else:
            raise AssertionError

        lines += [
            f'ARG BASE_IMAGE="{base_image}"',
            'FROM ${BASE_IMAGE}',
            '',
        ]

        # Install tools and additional libraries.
        if os_name == 'ubuntu':
            lines += [
                'RUN export DEBIAN_FRONTEND=noninteractive && \\',
                '    apt-get -qqy update && \\',

                # pyenv: https://github.com/pyenv/pyenv/wiki
                '    apt-get -qqy install \\',
                '       make build-essential libssl-dev zlib1g-dev \\',
                '       libbz2-dev libreadline-dev libsqlite3-dev wget \\',
                '       curl llvm libncursesw5-dev xz-utils tk-dev \\',
                '       libxml2-dev libxmlsec1-dev libffi-dev \\',
                '       liblzma-dev && \\',
                '    apt-get -qqy install ccache git curl && \\',
                '    apt-get -qqy --allow-change-held-packages install {}'.format(
                    ' '.join(self._additional_packages('apt'))
                ),
                '',
            ]
        elif os_name == 'centos':
            lines += [
                # pyenv: https://github.com/pyenv/pyenv/wiki
                'RUN yum -y install \\',
                '       zlib-devel bzip2 bzip2-devel readline-devel sqlite \\',
                '       sqlite-devel openssl-devel tk-devel libffi-devel \\',
                '       xz-devel && \\',
                '    yum -y install "@Development Tools" ccache git curl && \\',  # NOQA
                '    yum -y install {}'.format(
                    ' '.join(self._additional_packages('yum'))
                ),
                '',
            ]
        else:
            raise AssertionError

        # Enable ccache for gcc/g++.
        lines += [
            'ENV PATH "/usr/lib/ccache:${PATH}"',
            '',
        ]

        # Set environment variables for ROCm.
        if matrix.rocm is not None:
            lines += [
                'ENV ROCM_HOME "/opt/rocm"',
                'ENV LD_LIBRARY_PATH "${ROCM_HOME}/lib"',
                'ENV CPATH "${ROCM_HOME}/include"',
                'ENV LDFLAGS "-L${ROCM_HOME}/lib"',
                '',
            ]

        # Setup Python.
        if matrix.python is None:
            raise ValueError('Python cannot be null')

        py_spec = self.schema['python'][matrix.python]['spec']
        lines += [
            'RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv',
            'ENV PYENV_ROOT "/opt/pyenv"',
            'ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"',
            f'RUN pyenv install {py_spec} && \\',
            f'    pyenv global {py_spec} && \\',
            '    pip install -U setuptools pip',
            '',
        ]

        # Setup Python libraries.
        pip_args = []
        for pylib in ('numpy', 'scipy', 'optuna', 'cython'):
            pylib_ver = getattr(matrix, pylib)
            if pylib_ver is None:
                continue
            pip_spec = self.schema[pylib][pylib_ver]['spec']
            pip_args.append(f'{pylib}{pip_spec}')
        lines += [
            f'RUN pip install -U {" ".join(pip_args)}',
            '',
        ]

        return '\n'.join(lines)

    def _additional_packages(self, kind: str) -> List[str]:
        assert kind in ('apt', 'yum')
        matrix = self.matrix
        if matrix.cuda is not None:
            packages = []
            apt = kind == 'apt'
            cuda = matrix.cuda
            nccl = matrix.nccl
            cutensor = matrix.cutensor
            cusparselt = matrix.cusparselt
            cudnn = matrix.cudnn
            if nccl is not None:
                spec = self.schema['nccl'][nccl]['spec']
                major = nccl.split('.')[0]
                if apt:
                    packages.append(f'libnccl{major}={spec}+cuda{cuda}')
                    packages.append(f'libnccl-dev={spec}+cuda{cuda}')
                else:
                    packages.append(f'libnccl-devel-{spec}+cuda{cuda}')
            if cutensor is not None:
                spec = self.schema['cutensor'][cutensor]['spec']
                packages.append(
                    f'libcutensor-dev={spec}' if apt else
                    f'libcutensor-devel-{spec}')
            if cusparselt is not None:
                spec = self.schema['cusparselt'][cusparselt]['spec']
                packages.append(
                    f'libcusparselt-dev={spec}' if apt else
                    f'libcusparselt-devel-{spec}')
            if cudnn is not None:
                spec = self.schema['cudnn'][cudnn]['spec']
                major = cudnn.split('.')[0]
                packages.append(
                    f'libcudnn{major}-dev={spec}+cuda{cuda}' if apt else
                    f'libcudnn{major}-devel-{spec}+cuda{cuda}')
            return packages
        elif matrix.rocm is not None:
            return ['rocm-dev', 'hipblas', 'hipfft', 'hipsparse', 'hipcub',
                    'rocsparse', 'rocrand', 'rocthrust', 'rocsolver', 'rocfft',
                    'rocprim', 'rccl']
        raise AssertionError

    def generate_script(self) -> str:
        matrix = self.matrix
        lines = [
            '#!/bin/bash',
            '',
            '# AUTO GENERATED: DO NOT EDIT!',
            '',
            'set -uex',
            '',
            'ACTIONS="$(dirname $0)/actions"',
            '. "$ACTIONS/_environment.sh"',
            '',
        ]

        if matrix.cuda is not None:
            lines += [
                'export NVCC="ccache nvcc"',
                '',
            ]
        elif matrix.rocm is not None:
            lines += [
                'export CUPY_INSTALL_USE_HIP=1',
                '',
            ]
        else:
            raise AssertionError

        if matrix.accelerators is not None:
            lines += [
                f'export CUPY_ACCELERATORS="{matrix.accelerators}"',
                '',
            ]

        lines += ['"$ACTIONS/build.sh"']
        if matrix.test in ('unit', 'slow'):
            # pytest marker
            spec = 'not slow' if matrix.test == 'unit' else 'slow'
            lines += [f'"$ACTIONS/unittest.sh" "{spec}"']
        elif matrix.test == 'doctest':
            lines += ['"$ACTIONS/doctest.sh"']
        elif matrix.test == 'example':
            lines += ['"$ACTIONS/example.sh"']
        else:
            raise AssertionError

        lines += [
            '"$ACTIONS/cleanup.sh"',
            ''
        ]

        return '\n'.join(lines)


class CoverageGenerator:
    def __init__(self, schema: SchemaType, matrixes: List[Matrix]):
        self.schema = schema
        self.matrixes = matrixes

    def generate_markdown(self) -> str:
        # Generate a matrix table.
        table = [
            ['Param', '', 'Test'] + [''] * (len(self.matrixes) - 1) + ['#'],
            ['', 'System'] + [m.system for m in self.matrixes] + [''],
            ['', 'Target'] + [
                f'[{m.target}][t{i}]' for i, m in enumerate(self.matrixes)
            ] + [''],
            [''] * (len(self.matrixes) + 3)
        ]
        for key, key_schema in self.schema.items():
            possible_values = key_schema.keys()
            matrix_values = [getattr(m, key) for m in self.matrixes]
            key_header = key
            for value in possible_values:
                count = matrix_values.count(value)
                table += [
                    [key_header, value if value else 'null'] + [
                        '✅' if mv == value else '' for mv in matrix_values
                    ] + [
                        str(count) if count != 0 else '0 🚨'
                    ],
                ]
                key_header = ''

        # Prepare markdown output.
        lines = [
            '<!-- AUTO GENERATED: DO NOT EDIT! -->',
            '',
            '# CuPy CI Test Coverage',
            '',
        ]

        # Render the matrix table as markdown.
        widths = [
            max([len(row[col_idx]) for row in table])
            for col_idx in range(len(table[0]))
        ]
        for row_idx, row in enumerate(table):
            lines += [
                '| ' + ' | '.join([
                    ('{:<' + str(widths[col_idx]) + '}').format(row[col_idx])
                    for col_idx in range(len(row))
                ]) + ' |',
            ]
            if row_idx == 0:
                lines += [
                    '| ' + ' | '.join([
                        '-' * widths[col_idx]
                        for col_idx in range(len(row))
                    ]) + ' |',
                ]

        # Add links to FlexCI projects.
        lines += [''] + [
            f'[t{i}]:https://ci.preferred.jp/{m.project}/'
            for i, m in enumerate(self.matrixes)
        ] + ['']

        return '\n'.join(lines)


def validate_schema(schema: SchemaType):
    # Validate schema consistency
    for key, key_schema in schema.items():
        if key == 'os':
            for value, value_schema in key_schema.items():
                system = value_schema.get('system', None)
                if system is None:
                    raise ValueError(
                            f'system is missing '
                            f'while parsing schema os:{value}')
                if system not in schema['system'].keys():
                    raise ValueError(
                        f'unknown system: {system} '
                        f'while parsing schema os:{value}')
        if key in ('nccl', 'cutensor', 'cusparselt', 'cudnn'):
            for value, value_schema in key_schema.items():
                for cuda in value_schema.get('cuda', []):
                    if cuda not in schema['cuda'].keys():
                        raise ValueError(
                            f'unknown CUDA version: {cuda} '
                            f'while parsing schema {key}:{value}')
        elif key in ('numpy', 'scipy'):
            for value, value_schema in key_schema.items():
                for python in value_schema.get('python', []):
                    if python not in schema['python'].keys():
                        raise ValueError(
                            f'unknown Python version: {python} '
                            f'while parsing schema {key}:{value}')
                for numpy in value_schema.get('numpy', []):
                    if numpy not in schema['numpy'].keys():
                        raise ValueError(
                            f'unknown NumPy version: {numpy} '
                            f'while parsing schema {key}:{value}')


def validate_matrixes(schema: SchemaType, matrixes: List[Matrix]):
    # Validate overall consistency
    project_seen = set()
    system_target_seen = set()
    for matrix in matrixes:
        if not hasattr(matrix, 'project'):
            raise ValueError(f'matrix must have a project: {matrix}')

        if matrix.project in project_seen:
            raise ValueError(f'{matrix.project}: duplicate project name')
        project_seen.add(matrix.project)

        if not hasattr(matrix, 'target'):
            raise ValueError(f'{matrix.project}: target is missing')

        if (matrix.system, matrix.target) in system_target_seen:
            raise ValueError(
                f'{matrix.project}: duplicate system/target combination: '
                f'{matrix.system}/{matrix.target}')
        system_target_seen.add((matrix.system, matrix.target))

    # Validate consistency for each matrix
    for matrix in matrixes:
        if matrix.cuda is None and matrix.rocm is None:
            raise ValueError(
                f'{matrix.project}: Either cuda nor rocm must be non-null')

        if matrix.cuda is not None and matrix.rocm is not None:
            raise ValueError(
                f'{matrix.project}: cuda and rocm are mutually exclusive')

        for key, key_schema in schema.items():
            possible_values = list(key_schema.keys())
            if not hasattr(matrix, key):
                raise ValueError(f'{matrix.project}: {key} is missing')
            value = getattr(matrix, key)
            if value not in possible_values:
                raise ValueError(
                    f'{matrix.project}: {key} must be one of '
                    f'{possible_values} but got {value}')

            if key in ('nccl', 'cutensor', 'cusparselt', 'cudnn'):
                supports = schema[key][value].get('cuda', None)
                if supports is not None and matrix.cuda not in supports:
                    raise ValueError(
                        f'{matrix.project}: CUDA {matrix.cuda} '
                        f'not supported by {key} {value}')
            elif key in ('numpy', 'scipy'):
                supports = schema[key][value].get('python', None)
                if supports is not None and matrix.python not in supports:
                    raise ValueError(
                        f'{matrix.project}: Python {matrix.python} '
                        f'not supported by {key} {value}')
                supports = schema[key][value].get('numpy', None)
                if supports is not None and matrix.numpy not in supports:
                    raise ValueError(
                        f'{matrix.project}: NumPy {matrix.numpy} '
                        f'not supported by {key} {value}')


def log(msg: str) -> None:
    print(msg)


def parse_args(argv: List[str]) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--schema', type=str, required=True)
    parser.add_argument('-m', '--matrix', type=str, required=True)
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('-D', '--dry-run', action='store_true', default=False)
    return parser.parse_args()


def main(argv: List[str]) -> int:
    options = parse_args(argv)

    log(f'Loading schema: {options.schema}')
    with open(options.schema) as f:
        schema = yaml.load(f, Loader=yaml.Loader)
    validate_schema(schema)

    log(f'Loading project matrixes: {options.matrix}')
    matrixes = []
    with open(options.matrix) as f:
        for matrix_record in yaml.load(f, Loader=yaml.Loader):
            matrixes.append(Matrix(matrix_record))
    validate_matrixes(schema, matrixes)

    output = {}
    for matrix in matrixes:
        log(
            f'Processing project matrix: {matrix.project} '
            f'(system: {matrix.system}, target: {matrix.target})')
        if matrix.system == 'linux':
            gen = LinuxGenerator(schema, matrix)
            output[f'linux/tests/{matrix.target}.Dockerfile'] = \
                gen.generate_dockerfile()
            output[f'linux/tests/{matrix.target}.sh'] = \
                gen.generate_script()
        elif matrix.system == 'windows':
            raise ValueError('Windows is not supported yet')
        else:
            raise AssertionError

    covgen = CoverageGenerator(schema, matrixes)
    output['coverage.md'] = covgen.generate_markdown()
    # Write output files.
    base_dir = (
        options.directory if options.directory else
        os.path.abspath(os.path.dirname(argv[0])))
    retval = 0
    for filename, content in output.items():
        filepath = f'{base_dir}/{filename}'
        if options.dry_run:
            with open(filepath) as f:
                if f.read() != content:
                    log(f'{filepath} needs to be updated')
                    retval = 1
        else:
            log(f'Writing {filepath}')
            with open(filepath, 'w') as f:
                f.write(content)
    return retval


if __name__ == '__main__':
    sys.exit(main(sys.argv))
