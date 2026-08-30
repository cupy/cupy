"""Functions used by least-squares algorithms."""
import numpy as np
import cupy as cp

from scipy.sparse.linalg import LinearOperator, aslinearoperator


# Utility functions to work with bound constraints.
def in_bounds(x, lb, ub, arrtype=None):
    """Check if a point lies within bounds."""
    if arrtype is None:
        arrtype = cp
    return arrtype.all((x >= lb) & (x <= ub),axis=1)

def all_in_bounds(x, lb, ub):
    """Check if a point lies within bounds."""
    return cp.all((x >= lb) & (x <= ub))


# Functions to display algorithm's progress.


def print_header_nonlinear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality"))


def print_iteration_nonlinear(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print("{:^15}{:^15}{:^15.4e}{}{}{:^15.2e}"
          .format(iteration, nfev, cost, cost_reduction,
                  step_norm, optimality))


def print_header_linear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Cost", "Cost reduction", "Step norm",
                  "Optimality"))


def print_iteration_linear(iteration, cost, cost_reduction, step_norm,
                           optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print(f"{iteration:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}")


# Simple helper functions.


def compute_grad(J, f):
    """Compute gradient of the least-squares cost function."""
    if isinstance(J, LinearOperator):
        return J.rmatvec(f)
    else:
        return cp.einsum("ikj,ij->ik", J.swapaxes(1,2), f)
        #return J.T.dot(f)

