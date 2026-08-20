"""This module contains a GPU implementation of the BVLS algorithm in
    scipy.optimize.lsq_linear"""
from .gpu_lsq_linear import gpu_lsq_linear

__all__ = ['gpu_lsq_linear']
