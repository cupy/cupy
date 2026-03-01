"""GPU implementation of Bounded-variable least-squares algorithm."""
import numpy as np
import cupy as cp
#from numpy.linalg import norm, lstsq
from cupy.linalg import norm, lstsq
from scipy.optimize import OptimizeResult

from .gpu_common import print_header_linear, print_iteration_linear, compute_grad

def compute_kkt_optimality(g, on_bound):
    """Compute the maximum violation of KKT conditions."""
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = cp.abs(g[free_set])
    return cp.max(g_kkt, axis=1)


def bvls(A, b, x_lsq, lb, ub, ib, tol, max_iter, verbose, rcond=None, return_cost=True, return_optimality=True):
    d, m, n = A.shape

    #x = x_lsq.copy()
    x = x_lsq
    on_bound = cp.zeros((d,n), dtype=cp.int32)

    mask = x <= lb
    x[mask] = lb[mask]
    on_bound[mask] = -1

    mask = x >= ub
    x[mask] = ub[mask]
    on_bound[mask] = 1

    free_set = on_bound == 0
    active_set = ~free_set

    if return_cost or return_optimality:
        r = cp.einsum("ijk,ik->ij", A, x) - b
        #r = A.dot(x) - b
        cost = 0.5*cp.einsum("ij,ij->i", r, r)
        #cost = 0.5 * np.dot(r, r)
        initial_cost = cost
        if return_optimality:
            g = compute_grad(A,r)
            #g = A.T.dot(r)
    else:
        cost = None
        initial_cost = None

    cost_change = None
    step_norm = None
    iteration = 0

    if verbose == 2:
        print_header_linear()

    # This is the initialization loop. The requirement is that the
    # least-squares solution on free variables is feasible before BVLS starts.
    # One possible initialization is to set all variables to lower or upper
    # bounds, but many iterations may be required from this state later on.
    # The implemented ad-hoc procedure which intuitively should give a better
    # initial state: find the least-squares solution on current free variables,
    # if its feasible then stop, otherwise, set violating variables to
    # corresponding bounds and continue on the reduced set of free variables.

    while free_set.any():
        if verbose == 2 and return_optimality:
            optimality = compute_kkt_optimality(g, on_bound)
            print_iteration_linear(iteration, cost, cost_change, step_norm,
                                   optimality)

        iteration += 1
        if verbose == 2:
            #Only need x_old if verbose == 2
            x_old = x.copy()

        b_free = b - cp.einsum("ijk,ik->ij", A, x*active_set)
        A_free = A.swapaxes(1,2).copy()
        A_free[active_set] = 0
        A_free = A_free.swapaxes(1,2)
        z = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A_free), b_free)

        #A_free = A[:, free_set]
        #b_free = b - A.dot(x * active_set)
        #z = lstsq(A_free, b_free, rcond=rcond)[0]

        lbv = (z < lb)*free_set
        ubv = (z > ub)*free_set
        #lbv = z < lb[free_set]
        #ubv = z > ub[free_set]
        v = lbv | ubv

        if cp.any(lbv):
            #ind = free_set[lbv]
            #x[ind] = lb[ind]
            x[lbv] = lb[lbv]
            #active_set[ind] = True
            active_set[lbv] = True
            #on_bound[ind] = -1
            on_bound[lbv] = -1

        if cp.any(ubv):
            #ind = free_set[ubv]
            #x[ind] = ub[ind]
            x[ubv] = ub[ubv]
            #active_set[ind] = True
            active_set[ubv] = True
            #on_bound[ind] = 1
            on_bound[ubv] = 1

        #ind = free_set[~v]
        ind = free_set*~v
        #x[ind] = z[~v]
        x[ind] = z[ind]

        r = cp.einsum("ijk,ik->ij", A, x) - b
        #r = A.dot(x) - b
        #cost_new = 0.5 * np.dot(r, r)
        cost_new = 0.5*cp.einsum("ij,ij->i", r, r)
        if return_cost:
            cost_change = cost - cost_new
        cost = cost_new
        #g = A.T.dot(r)
        g = compute_grad(A,r)
        if verbose == 2:
            #step_norm = norm(x[free_set] - x_free_old)
            step_norm = norm(x-x_old, axis=1)

        if cp.any(v):
            free_set[v] = False
            #free_set = free_set[~v]
        else:
            break

    if max_iter is None:
        max_iter = n
    max_iter += iteration

    termination_status = None

    # Main BVLS loop.
    optimality = compute_kkt_optimality(g, on_bound)
    idx = np.arange(d, dtype=np.int32)
    for iteration in range(iteration, max_iter):  # BVLS Loop A
        if verbose == 2 and return_optimality:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, optimality)

        if cp.all(optimality < tol):
            termination_status = 1

        if termination_status is not None:
            break

        g_on = g * on_bound
        move_to_free = cp.argmax(g_on, axis=1)
        #on_bound[idx,move_to_free] = cp.minimum(0, on_bound[idx,move_to_free])
        on_bound[idx,move_to_free] = (g_on[idx,move_to_free] < 0)*on_bound[idx,move_to_free]
        
        while True:   # BVLS Loop B
            free_set = on_bound == 0
            active_set = ~free_set
    
            if verbose == 2:
                #Only need x_old if verbose == 2
                x_old = x.copy()

            b_free = b - cp.einsum("ijk,ik->ij", A, x*active_set)
            #Set active_set to 0 in A_free
            A_free = A.swapaxes(1,2).copy()
            A_free[active_set] = 0
            A_free = A_free.swapaxes(1,2)

            z = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A_free), b_free)

            #A_free = A[:, free_set]
            #b_free = b - A.dot(x * active_set)
            #z = lstsq(A_free, b_free, rcond=rcond)[0]

            lbv = (z < lb)*free_set
            ubv = (z > ub)*free_set
            v = (lbv | ubv)

            #lbv, = np.nonzero(z < lb_free)
            #ubv, = np.nonzero(z > ub_free)
            #v = np.hstack((lbv, ubv))

            #if v.size > 0:
            if v.any():
                #alphas = cp.zeros(x.shape)
                #av = v.any(axis=1)
                alphas = cp.full(x.shape, cp.inf)
                alphas[lbv] = (lb[lbv]-x[lbv]) / (z[lbv]-x[lbv])
                alphas[ubv] = (ub[ubv]-x[ubv]) / (z[ubv]-x[ubv])

                i = cp.argmin(alphas, axis=1)
                alpha = alphas[idx,i].reshape((d,1))
                #x_free = x[v] * (1-alpha[v])
                #x_free += alpha[v]*z[v]

                #x_free = x[v] * (1-alpha[av])
                #x_free += alpha[av]*z[v]
                #x[v] = x_free

                x_free = x*(1-alpha)
                x_free += alpha*z
                x[v] = x_free[v]

                #Update on_bound
                on_bound[cp.where(lbv[idx,i]), i[cp.where(lbv[idx,i])]] = -1
                on_bound[cp.where(ubv[idx,i]), i[cp.where(ubv[idx,i])]] = 1

                #alphas = np.hstack((
                #    lb_free[lbv] - x_free[lbv],
                #    ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])

                #i = np.argmin(alphas)
                #i_free = v[i]
                #alpha = alphas[i]

                #x_free *= 1 - alpha
                #x_free += alpha * z
                #x[free_set] = x_free

                #if i < lbv.size:
                #    on_bound[free_set[i_free]] = -1
                #else:
                #    on_bound[free_set[i_free]] = 1
            else:
                x_free = z[free_set]
                x[free_set] = x_free
                break

        if verbose == 2:
            #step_norm = norm(x_free - x_free_old, axis=1)
            step_norm = norm(x - x_old, axis=1)

        r = cp.einsum("ijk,ik->ij", A, x) - b
        #r = A.dot(x) - b
        #cost_new = 0.5 * np.dot(r, r)
        cost_new = 0.5*cp.einsum("ij,ij->i", r, r)
        cost_change = cost - cost_new

        if cp.all(cost_change < tol * cost):
            termination_status = 2
        cost = cost_new

        #g = A.T.dot(r)
        g = compute_grad(A,r)
        optimality = compute_kkt_optimality(g, on_bound)
        x_lsq = x

    if termination_status is None:
        termination_status = 0

    return OptimizeResult(
        x=x_lsq, fun=r, cost=cost, optimality=optimality, active_mask=on_bound,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
