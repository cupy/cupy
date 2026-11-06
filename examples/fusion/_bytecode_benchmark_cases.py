"""Numerical workloads and explicit-out reference implementations.

Factories allocate scratch outside timing. Constructor baselines still fill
their reused arrays on every call. Returned reused outputs are overwritten by
subsequent calls; the allocating reference preserves ordinary return ownership.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cupy as cp


def score(x):
    energy = (cp.exp(-cp.absolute(x)) * cp.sin(x)**2).sum(axis=1)
    mass = (1 / (1 + cp.exp(-x))).sum(axis=1)
    return cp.sqrt(energy) / (mass + 1)


def row_norm(x):
    return cp.sqrt((x * x).sum(axis=1) + 0.001)


def map_chain(x):
    return cp.tanh(cp.sin(x) * cp.cos(x) + x * 0.5)


def envelope(x):
    return cp.exp(-cp.absolute(x))


def wave(x, scale):
    return cp.sin(x) * scale + cp.cos(x)


def nested_chain(x, scale):
    return cp.sqrt(envelope(x) + wave(x, scale)**2)


def boolean_branch(x, flag):
    if flag:
        return cp.sin(x) * cp.cos(x) + x * 2
    return cp.tanh(x) * cp.exp(-x)


def shape_branch(x):
    if x.shape[1] <= 128:
        return cp.sin(x) * cp.cos(x) + x
    return cp.tanh(x) * envelope(x)


def conditional_allocation(x, flag):
    if flag:
        bias = cp.full(x.shape, 0.25, dtype=x.dtype)
        return cp.sin(x + bias) * cp.cos(x) + x * x
    return cp.tanh(x) * cp.exp(-x)


def allocation_mid(x):
    mapped = cp.sin(x) * x
    bias = cp.ones_like(x)
    return cp.tanh(mapped + bias) * cp.cos(x)


def shaped_allocation(x):
    bias = cp.zeros((x.shape[0], 1), dtype=x.dtype)
    return cp.sin(x + bias) * cp.cos(x) + x


def multiple_outputs(x):
    sine = cp.sin(x)
    return sine * sine + sine, cp.cos(x) - sine


class Transform:
    def __init__(self, scale, bias):
        self.scale, self.bias = scale, bias

    def helper(self, x):
        return cp.sin(x) * self.scale

    def apply(self, x):
        return cp.tanh(self.helper(x) + self.bias) * cp.cos(x)


def buffers(x, count):
    # Dense C-order scratch, including for noncontiguous input profiles.
    return [cp.empty(x.shape, dtype=x.dtype) for _ in range(count)]


def reuse_score(x):
    a, b = buffers(x, 2)
    energy, mass, out = [cp.empty(x.shape[0], dtype=x.dtype)
                         for _ in range(3)]

    def run(x):
        cp.absolute(x, out=a)
        cp.negative(a, out=a)
        cp.exp(a, out=a)
        cp.sin(x, out=b)
        cp.square(b, out=b)
        cp.multiply(a, b, out=a)
        cp.sum(a, axis=1, out=energy)
        cp.negative(x, out=a)
        cp.exp(a, out=a)
        cp.add(a, 1, out=a)
        cp.true_divide(1, a, out=a)
        cp.sum(a, axis=1, out=mass)
        cp.sqrt(energy, out=energy)
        cp.add(mass, 1, out=mass)
        cp.true_divide(energy, mass, out=out)
        return out

    return run


def reuse_row_norm(x):
    a, = buffers(x, 1)
    out = cp.empty(x.shape[0], dtype=x.dtype)

    def run(x):
        cp.multiply(x, x, out=a)
        cp.sum(a, axis=1, out=out)
        cp.add(out, 0.001, out=out)
        cp.sqrt(out, out=out)
        return out

    return run


def reuse_map_chain(x):
    a, b = buffers(x, 2)

    def run(x):
        cp.sin(x, out=a)
        cp.cos(x, out=b)
        cp.multiply(a, b, out=a)
        cp.multiply(x, 0.5, out=b)
        cp.add(a, b, out=a)
        cp.tanh(a, out=a)
        return a

    return run


def reuse_nested(x):
    a, b, c = buffers(x, 3)

    def run(x, scale):
        cp.absolute(x, out=a)
        cp.negative(a, out=a)
        cp.exp(a, out=a)
        cp.sin(x, out=b)
        cp.multiply(b, scale, out=b)
        cp.cos(x, out=c)
        cp.add(b, c, out=b)
        cp.square(b, out=b)
        cp.add(a, b, out=a)
        cp.sqrt(a, out=a)
        return a

    return run


def reuse_boolean(x):
    a, b = buffers(x, 2)

    def run(x, flag):
        if flag:
            cp.sin(x, out=a)
            cp.cos(x, out=b)
            cp.multiply(a, b, out=a)
            cp.multiply(x, 2, out=b)
            cp.add(a, b, out=a)
        else:
            cp.tanh(x, out=a)
            cp.negative(x, out=b)
            cp.exp(b, out=b)
            cp.multiply(a, b, out=a)
        return a

    return run


def reuse_shape(x):
    a, b = buffers(x, 2)

    def run(x):
        if x.shape[1] <= 128:
            cp.sin(x, out=a)
            cp.cos(x, out=b)
            cp.multiply(a, b, out=a)
            cp.add(a, x, out=a)
        else:
            cp.tanh(x, out=a)
            cp.absolute(x, out=b)
            cp.negative(b, out=b)
            cp.exp(b, out=b)
            cp.multiply(a, b, out=a)
        return a

    return run


def reuse_conditional_allocation(x, *, preinitialized=False):
    a, b, bias = buffers(x, 3)
    if preinitialized:
        bias.fill(0.25)

    def run(x, flag):
        if flag:
            if not preinitialized:
                bias.fill(0.25)
            cp.add(x, bias, out=a)
            cp.sin(a, out=a)
            cp.cos(x, out=b)
            cp.multiply(a, b, out=a)
            cp.multiply(x, x, out=b)
            cp.add(a, b, out=a)
        else:
            cp.tanh(x, out=a)
            cp.negative(x, out=b)
            cp.exp(b, out=b)
            cp.multiply(a, b, out=a)
        return a

    return run


def constants_conditional_allocation(x):
    return reuse_conditional_allocation(x, preinitialized=True)


def reuse_allocation_mid(x, *, preinitialized=False):
    a, b = buffers(x, 2)
    bias = buffers(x, 1)[0] if preinitialized else b
    if preinitialized:
        bias.fill(1)

    def run(x):
        cp.sin(x, out=a)
        cp.multiply(a, x, out=a)
        if not preinitialized:
            bias.fill(1)
        cp.add(a, bias, out=a)
        cp.tanh(a, out=a)
        cp.cos(x, out=b)
        cp.multiply(a, b, out=a)
        return a

    return run


def constants_allocation_mid(x):
    return reuse_allocation_mid(x, preinitialized=True)


def reuse_shaped_allocation(x, *, preinitialized=False):
    a, b = buffers(x, 2)
    bias = cp.empty((x.shape[0], 1), dtype=x.dtype)
    if preinitialized:
        bias.fill(0)

    def run(x):
        if not preinitialized:
            bias.fill(0)
        cp.add(x, bias, out=a)
        cp.sin(a, out=a)
        cp.cos(x, out=b)
        cp.multiply(a, b, out=a)
        cp.add(a, x, out=a)
        return a

    return run


def constants_shaped_allocation(x):
    return reuse_shaped_allocation(x, preinitialized=True)


def reuse_multiple_outputs(x):
    a, b, c = buffers(x, 3)

    def run(x):
        cp.sin(x, out=a)
        cp.multiply(a, a, out=b)
        cp.add(b, a, out=b)
        cp.cos(x, out=c)
        cp.subtract(c, a, out=c)
        return b, c

    return run


def reuse_transform(x):
    a, b = buffers(x, 2)

    def run(self, x):
        cp.sin(x, out=a)
        cp.multiply(a, self.scale, out=a)
        cp.add(a, self.bias, out=a)
        cp.tanh(a, out=a)
        cp.cos(x, out=b)
        cp.multiply(a, b, out=a)
        return a

    return run


@dataclass
class Case:
    name: str
    function: object
    factory: object
    arguments: tuple
    feature: str
    x_index: int = 0
    constant_factory: object = None

    def reused(self, *, constants=False):
        x_index = self.x_index
        factory = self.constant_factory if constants else self.factory
        by_shape = {}
        for args in self.arguments:
            x = args[x_index]
            if x.shape not in by_shape:
                by_shape[x.shape] = factory(x)
        if len(by_shape) == 1:
            return next(iter(by_shape.values()))

        def dispatch(*args):
            return by_shape[args[x_index].shape](*args)

        return dispatch


def cases(x):
    """All variants are finite and are validated/warmed before measurement."""
    yield Case('score', score, reuse_score, ((x,),), 'multiple reductions')
    yield Case('row_norm', row_norm, reuse_row_norm, ((x,),),
               'single reduction with pre/post maps')
    yield Case('map_chain', map_chain, reuse_map_chain, ((x,),), 'elementwise')
    for label, typ in (('numpy', np.float32), ('python', float)):
        values = tuple((x, typ(s)) for s in (0.25, 0.75, 0.5))
        yield Case(f'nested_{label}_scalars', nested_chain, reuse_nested,
                   values, 'nested callees and changing scalar values')
    for label, values in (('true', (True,)), ('false', (False,)),
                          ('alternating', (True, False))):
        arguments = tuple((x, flag) for flag in values)
        yield Case(f'boolean_{label}', boolean_branch, reuse_boolean,
                   arguments, 'boolean branch')
        yield Case(f'allocation_{label}', conditional_allocation,
                   reuse_conditional_allocation, arguments,
                   'conditional allocation and boolean branch',
                   constant_factory=constants_conditional_allocation)
    yield Case('shape_branch', shape_branch, reuse_shape, ((x,),),
               'shape-dependent branch')
    if x.size >= 512 and x.size % 512 == 0 and x.flags.c_contiguous:
        narrow, wide = x.reshape(-1, 64), x.reshape(-1, 512)
        yield Case('shape_alternating', shape_branch, reuse_shape,
                   ((narrow,), (wide,)),
                   'alternating metadata branch outcomes')
    yield Case('allocation_mid', allocation_mid, reuse_allocation_mid,
               ((x,),), 'allocation between map regions',
               constant_factory=constants_allocation_mid)
    yield Case('shaped_allocation', shaped_allocation, reuse_shaped_allocation,
               ((x,),), 'metadata-derived allocation and broadcasting',
               constant_factory=constants_shaped_allocation)
    bias = cp.linspace(-0.1, 0.2, x.shape[1], dtype=x.dtype)
    models = (Transform(np.float32(0.5), bias),
              Transform(np.float32(0.75), -bias))
    yield Case('instance_method', Transform.apply, reuse_transform,
               ((models[0], x),),
               'nested method and scalar/array attributes', 1)
    yield Case('instance_alternating', Transform.apply, reuse_transform,
               tuple((model, x) for model in models),
               'shared method analysis across changing instances', 1)
    yield Case('multiple_outputs', multiple_outputs, reuse_multiple_outputs,
               ((x,),), 'shared producer and tuple outputs')
