#!/usr/bin/env python3

import argparse
import os
import shlex  # requires Python 3.8
import sys

import yaml

from typing import Any, Dict, List, Tuple


SchemaType = Dict[str, Any]


class Matrix:
    def __init__(self, record: Dict[str, str]):
        self._rec = record

    def env(self):
        envvars = {}
        for k, v in self._rec.items():
            if not k.startswith('env:') or v is None:
                continue
            envvars[k.split(':', 2)[1]] = v
        return envvars

    def __getattr__(self, key):
        if key in self._rec:
            return self._rec[key]
        raise AttributeError

    def copy(self):
        return Matrix(self._rec.copy())

    def update(self, matrix: 'Matrix'):
        self._rec.update(matrix._rec)


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
            if matrix.rocm is not None:
                # GPG key has expired in ROCm 4.2 (or earlier) docker images
                lines += [
                    'RUN export DEBIAN_FRONTEND=noninteractive && \\',
                    '    ( apt-get -qqy update || true ) && \\',
                    '    apt-get -qqy install ca-certificates && \\',
                    '    curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -',  # NOQA
                ]
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
                '    apt-get -qqy --allow-change-held-packages \\',
                '            --allow-downgrades install {}'.format(
                    shlex.join(self._additional_packages('apt'))
                ),
                '',
                'ENV PATH "/usr/lib/ccache:${PATH}"',
                '',
            ]
        elif os_name == 'centos':
            assert os_version in ('7', '8')
            if os_version == '7':
                lines += [
                    'RUN yum -y install centos-release-scl && \\',
                    '    yum -y install devtoolset-7-gcc-c++',
                    'ENV PATH "/opt/rh/devtoolset-7/root/usr/bin:${PATH}"',
                    'ENV LD_LIBRARY_PATH "/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:${LD_LIBRARY_PATH}"',  # NOQA
                    '',
                ]

            lines += [
                # pyenv: https://github.com/pyenv/pyenv/wiki
                'RUN yum -y install \\',
                '       zlib-devel bzip2 bzip2-devel readline-devel sqlite \\',
                '       sqlite-devel openssl-devel tk-devel libffi-devel \\',
                '       xz-devel && \\',
                '    yum -y install epel-release && \\',
                '    yum -y install "@Development Tools" ccache git curl && \\',  # NOQA
                '    yum -y install {}'.format(
                    shlex.join(self._additional_packages('yum'))
                ),
                '',
                'ENV PATH "/usr/lib64/ccache:${PATH}"',
                '',
            ]
        else:
            raise AssertionError

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
        for pylib in ('numpy', 'scipy', 'optuna', 'cython', 'cuda-python'):
            pylib_ver = getattr(matrix, pylib)
            if pylib_ver is None:
                continue
            pip_spec = self.schema[pylib][pylib_ver]['spec']
            pip_args.append(f'{pylib}{pip_spec}')
        lines += [
            f'RUN pip install -U {shlex.join(pip_args)}',
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
                    packages.append(f'libnccl-{spec}-*+cuda{cuda}')
                    packages.append(f'libnccl-devel-{spec}-*+cuda{cuda}')
            if cutensor is not None:
                spec = self.schema['cutensor'][cutensor]['spec']
                major = cutensor.split('.')[0]
                if apt:
                    packages.append(f'libcutensor{major}={spec}')
                    packages.append(f'libcutensor-dev={spec}')
                else:
                    packages.append(f'libcutensor{major}-{spec}')
                    packages.append(f'libcutensor-devel-{spec}')
            if cusparselt is not None:
                spec = self.schema['cusparselt'][cusparselt]['spec']
                major = cusparselt.split('.')[0]
                if apt:
                    packages.append(f'libcusparselt{major}={spec}')
                    packages.append(f'libcusparselt-dev={spec}')
                else:
                    packages.append(f'libcusparselt{major}-{spec}')
                    packages.append(f'libcusparselt-devel-{spec}')
            if cudnn is not None:
                spec = self.schema['cudnn'][cudnn]['spec']
                cudnn_cuda_schema = self.schema['cudnn'][cudnn]['cuda'][cuda]
                alias = cuda
                if cudnn_cuda_schema is not None:
                    alias = cudnn_cuda_schema['alias']
                major = cudnn.split('.')[0]
                if apt:
                    packages.append(f'libcudnn{major}={spec}+cuda{alias}')
                    packages.append(f'libcudnn{major}-dev={spec}+cuda{alias}')
                else:
                    packages.append(
                        f'libcudnn{major}-{spec}-*.cuda{alias}')
                    packages.append(
                        f'libcudnn{major}-devel-{spec}-*.cuda{alias}')
            return packages
        elif matrix.rocm is not None:
            return self.schema['rocm'][matrix.rocm]['packages']
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
                '# TODO(kmaehashi): Tentatively sparsen parameterization to make test run complete.',  # NOQA
                'export CUPY_TEST_FULL_COMBINATION="0"',
                'export CUPY_INSTALL_USE_HIP=1',
                '',
            ]
        else:
            raise AssertionError

        for key, value in matrix.env().items():
            lines += [
                f'export {key}="{value}"',
                '',
            ]

        lines += ['"$ACTIONS/build.sh"']
        if matrix.test.startswith('unit'):
            if matrix.test == 'unit':
                spec = 'not slow and not multi_gpu'
            elif matrix.test == 'unit-multi':
                spec = 'not slow and multi_gpu'
            elif matrix.test == 'unit-slow':
                # Slow tests may use multiple GPUs.
                spec = 'slow'
            else:
                assert False
            lines += [f'"$ACTIONS/unittest.sh" "{spec}"']
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

    def generate_markdown(self) -> Tuple[str, List[str]]:
        # Generate a matrix table.
        table = [
            ['Param', '', 'Test'] + [''] * (len(self.matrixes) - 1) + ['#'],
            ['', 'System'] + [m.system for m in self.matrixes] + [''],
            ['', 'Target'] + [
                f'[{m.target}][t{i}][🐳][d{i}][📜][s{i}]'
                for i, m in enumerate(self.matrixes)
            ] + [''],
            [''] * (len(self.matrixes) + 3)
        ]
        coverage_warns = []
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
                if count == 0:
                    coverage_warns.append(f'Uncovered axis: {key} = {value}')

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
        lines += ['']
        for i, m in enumerate(self.matrixes):
            url = f'https://ci.preferred.jp/{m.project}/'
            if hasattr(m, '_url'):
                url = m._url
            lines += [
                f'[t{i}]:{url}',
                f'[d{i}]:{m.system}/tests/{m.target}.Dockerfile',
                f'[s{i}]:{m.system}/tests/{m.target}.sh',
            ]
        lines += ['']

        return '\n'.join(lines), coverage_warns


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
                for cuda, _ in value_schema.get('cuda', {}).items():
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


def validate_matrixes(schema: SchemaType, matrixes: List[Matrix]) -> None:
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


def expand_inherited_matrixes(matrixes: List[Matrix]) -> None:
    prj2mat = {m.project: m for m in matrixes}
    for matrix in [m for m in matrixes if hasattr(m, '_inherits')]:
        parent = prj2mat[matrix._inherits]
        log(f'Project {matrix.project} inherits from {parent.project}')
        assert not hasattr(parent, '_inherits'), 'no nested inheritance'
        # Fill values missing in the matrix with parent's values
        inherited = parent.copy()
        inherited.update(matrix)
        matrix.update(inherited)


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
    expand_inherited_matrixes(matrixes)
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
    covout, warns = covgen.generate_markdown()
    output['coverage.md'] = covout
    if len(warns) != 0:
        log('----------------------------------------')
        log('Test coverage warnings:')
        for w in warns:
            log(f'* {w}')
        log('----------------------------------------')

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
