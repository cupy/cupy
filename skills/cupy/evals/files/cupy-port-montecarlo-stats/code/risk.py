# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portfolio risk Monte Carlo with bootstrap and a calibration step. NumPy/SciPy version."""

import time

import numpy as np
from scipy import optimize, stats

N_PATHS = 400_000
N_ASSETS = 16


def simulate(rng, mu, sigma, weights):
    shocks = rng.normal(loc=mu, scale=sigma, size=(N_PATHS, N_ASSETS))
    scenario = rng.choice(3, size=N_PATHS, p=[0.7, 0.2, 0.1])  # regime per path
    shocks *= np.array([1.0, 1.5, 3.0])[scenario][:, None]
    return shocks @ weights


def var_es(pnl, level=0.99):
    q = np.quantile(pnl, 1 - level)
    return -q, -pnl[pnl <= q].mean()


def bootstrap_var(rng, pnl, n_boot=200, level=0.99):
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.permutation(pnl.size)[: pnl.size // 10]
        out[b] = -np.quantile(pnl[idx], 1 - level)
    return out.std()


def main():
    rng = np.random.default_rng(42)
    mu = np.full(N_ASSETS, 0.0002)
    sigma = np.linspace(0.01, 0.03, N_ASSETS)
    weights = np.full(N_ASSETS, 1.0 / N_ASSETS)

    t0 = time.perf_counter()
    pnl = simulate(rng, mu, sigma, weights)
    var, es = var_es(pnl)
    var_se = bootstrap_var(rng, pnl)

    # Normal-approximation VaR and how well it matches the simulated tail
    z = stats.norm.ppf(0.01)
    var_normal = -(pnl.mean() + z * pnl.std())
    tail_prob = stats.norm.cdf((-var - pnl.mean()) / pnl.std())
    r, _ = stats.pearsonr(pnl[:-1], pnl[1:])

    # Calibrate the weight scale so that VaR hits a 2% target
    def gap(scale):
        return -np.quantile(pnl * scale, 0.01) - 0.02

    scale = optimize.brentq(gap, 0.1, 10.0)
    elapsed = time.perf_counter() - t0

    print(f"VaR99={var:.5f} ES99={es:.5f} VaR_se={var_se:.6f} VaR_normal={var_normal:.5f}")
    print(f"tail_prob={tail_prob:.4f} lag1_corr={r:.4f} scale_for_2pct={scale:.4f}")
    print(f"time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
