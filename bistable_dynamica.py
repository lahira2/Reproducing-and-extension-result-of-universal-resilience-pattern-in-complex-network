#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regulatory NuRsE-style pipeline with non-Laplacian coupling and CSV adjacency.

Dynamics:
    dx_i/dt = a * x_i - b * x_i^3 + sigma * sum_j A_ij * x_j

Implements:
- Preprocessing: a=2; delete_no_input; find_components (SCC size >= 2)
- Random node removal
- Random link removal
- Targeted node attack (degree-biased, stochastic but with fixed step schedule)
- iteration_real_R: integrates from x0=2 to t=400, outputs [mean, x_eff, beta_eff, f]

Usage example:
  python figure_3_laplacian.py --data network.csv --nnlros 1 --num_reali 20 --plot
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

# --------------------------- Model parameters ----------------------------

A_PARAM = 1.0      # a
B_PARAM = 1.0      # b
SIGMA_PARAM = 0.3  # sigma


# ---------------------------- I/O + Preprocess ----------------------------

def load_A_only(path: str) -> np.ndarray:
    """
    Load adjacency matrix from a CSV file.
    """
    A = np.loadtxt(path, delimiter=",")
    return A.astype(float)


def delete_no_input(A: np.ndarray) -> np.ndarray:
    AT = A.T.copy()
    n_prev = AT.shape[0]
    while AT.size and n_prev > 0:
        d = np.asarray(AT.sum(axis=0)).ravel()
        notin = np.where(d == 0)[0]
        if notin.size == 0:
            break
        keep = np.setdiff1d(np.arange(AT.shape[0]), notin, assume_unique=False)
        AT = AT[np.ix_(keep, keep)]
        n_now = AT.shape[0]
        if n_now == n_prev:
            break
        n_prev = n_now
    return AT.T


def delete_no_output(A: np.ndarray) -> np.ndarray:
    A2 = A.copy()
    n_prev = A2.shape[0]
    while A2.size and n_prev > 0:
        d = np.asarray(A2.sum(axis=1)).ravel()
        notout = np.where(d == 0)[0]
        if notout.size == 0:
            break
        keep = np.setdiff1d(np.arange(A2.shape[0]), notout, assume_unique=False)
        A2 = A2[np.ix_(keep, keep)]
        n_now = A2.shape[0]
        if n_now == n_prev:
            break
        n_prev = n_now
    return A2


def find_components(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return np.array([], dtype=int)
    G = csr_matrix(A != 0)
    n_comp, labels = connected_components(G, directed=True, connection="strong")
    counts = np.bincount(labels)
    keep_labels = np.where(counts >= 2)[0]
    keep_idx = (
        np.concatenate([np.where(labels == lab)[0] for lab in keep_labels])
        if keep_labels.size
        else np.array([], dtype=int)
    )
    return np.sort(keep_idx)


def preprocess_main(A: np.ndarray, scale: float = 2.0) -> np.ndarray:
    A2 = A * scale
    A2 = delete_no_input(A2)
    cluster = find_components(A2)
    if cluster.size == 0:
        return np.zeros((0, 0))
    return A2[np.ix_(cluster, cluster)]


# ------------------------------- Dynamics --------------------------------

def R_system_rhs(t: float, x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Non-Laplacian coupling:
        dx_i/dt = a*x_i - b*x_i^3 + sigma * sum_j A_ij * x_j
    """
    a = A_PARAM
    b = B_PARAM
    sigma = SIGMA_PARAM

    x_local = a * x - b * (x ** 3)
    Ax = A.dot(x)
    dxdt = x_local + sigma * Ax
    return dxdt


def integrate_to_tf(
    A: np.ndarray,
    x0: np.ndarray,
    t0: float = 0.0,
    tf: float = 400.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> np.ndarray:
    if x0.size == 0:
        return x0
    fun = lambda t, x: R_system_rhs(t, x, A)
    sol = solve_ivp(fun, (t0, tf), x0, method="RK45", rtol=rtol, atol=atol, vectorized=False)
    if not sol.success:
        sol = solve_ivp(fun, (t0, tf), x0, method="BDF", rtol=rtol, atol=atol)
    return sol.y[:, -1]


def betaspace(A: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    if A.size == 0 or x.size == 0:
        return 0.0, 0.0
    sAout = np.asarray(A.sum(axis=0)).ravel()
    x_nss = A.dot(x)
    denom_A = A.sum()
    if denom_A == 0:
        return 0.0, 0.0
    beta = float((A @ A).sum() / denom_A)
    x_eff = float(x_nss.sum() / sAout.sum()) if sAout.sum() != 0 else 0.0
    return x_eff, beta


# ---------------------------- Iteration step -----------------------------

def iteration_real_R(
    step: int,
    output_one: Optional[np.ndarray],
    A: np.ndarray,
    paper_R: int = 0,
) -> np.ndarray:
    n = A.shape[0]
    if n == 0:
        row = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        x0 = np.full(n, 2.0, dtype=float)
        y1 = integrate_to_tf(A, x0)

        if paper_R == 1:
            zeroy1 = np.where(y1 < 1.1)[0]
            if zeroy1.size < n:
                keep = np.setdiff1d(np.arange(n), zeroy1)
                A2 = A[np.ix_(keep, keep)]
                x0_2 = np.full(A2.shape[0], 2.0, dtype=float)
                y1 = integrate_to_tf(A2, x0_2)
                A = A2

        mean_y = float(np.mean(y1))
        xnn, beta = betaspace(A, y1)
        row = np.array([mean_y, xnn, beta, 0.0], dtype=float)

    if output_one is None or output_one.size == 0:
        output = row.reshape(1, -1)
    else:
        output = np.vstack([output_one, row])
    return output


# ------------------------------ Perturbations ----------------------------

def node_removal_R(nnlros: int, A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random node removal:
      - random permutation of nodes
      - progressively remove them in blocks of size nnlros
    """
    n = A.shape[0]
    steps_order = list(range(n - 1, 0, -nnlros)) + [0]
    perm = rng.permutation(n)
    A_perm = A[np.ix_(perm, perm)]
    out = None
    step_idx = 0
    for i in steps_order:
        step_idx += 1
        if i > 0:
            Acur = A_perm[i:, i:]
            cluster = find_components(Acur)
            Acur = Acur[np.ix_(cluster, cluster)] if cluster.size else np.zeros((0, 0))
        else:
            Acur = A_perm.copy()
        out = iteration_real_R(step_idx, out, Acur)
        f = 1.0 - i / n
        out[-1, 3] = f
    return out


def link_removal_R(nnlros: int, A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random link removal.
    """
    Awork = delete_no_output(A)
    Awork = delete_no_input(Awork)
    cluster = find_components(Awork)
    Awork = Awork[np.ix_(cluster, cluster)] if cluster.size else np.zeros((0, 0))

    if Awork.size == 0:
        out = iteration_real_R(1, None, Awork)
        out[-1, 3] = 0.0
        return out

    Aw = coo_matrix(Awork)
    rows, cols, data = Aw.row, Aw.col, Aw.data
    L = len(rows)
    the_links = rng.permutation(L)

    out = None
    step_idx = 0
    for remove in list(range(L - 1, nnlros - 1, -nnlros)) + [0]:
        step_idx += 1
        s1 = data.copy()
        if remove != 0:
            removed = the_links[:remove]
            s1[removed] = 0.0

        max_i_j = int(max(rows.max() if rows.size else 0, cols.max() if cols.size else 0))
        Anew = coo_matrix((s1, (rows, cols)), shape=(max_i_j + 1, max_i_j + 1)).tocsr()

        d = np.asarray(Anew.sum(axis=1)).ravel()
        keep = np.where(d != 0)[0]
        if keep.size:
            Anew = Anew[keep][:, keep]
        else:
            Anew = csr_matrix((0, 0))

        if Anew.shape[0] > 1 and Anew.nnz > 0:
            Gbin = (Anew != 0).astype(int)
            n_comp, labels = connected_components(Gbin, directed=True, connection="strong")
            counts = np.bincount(labels)
            keep_labels = np.where(counts >= 2)[0]
            if keep_labels.size:
                keep_idx = np.concatenate([np.where(labels == lab)[0] for lab in keep_labels])
                keep_idx.sort()
                Anew = Anew[keep_idx][:, keep_idx]
            else:
                Anew = csr_matrix((0, 0))

        Acur = Anew.toarray() if Anew.nnz > 0 else np.zeros((Anew.shape[0], Anew.shape[1]))
        out = iteration_real_R(step_idx, out, Acur)
        f = 1.0 - remove / L
        out[-1, 3] = f

    return out


def targeted_node_removal_R(nnlros: int, A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Targeted node attack (degree-biased, stochastic, fixed step schedule).

    We first build a full removal order of nodes (a permutation of 0..n-1)
    by repeatedly:
      - computing degrees on the current graph (deg_out + deg_in),
      - sampling one node with probability proportional to degree,
      - removing it and continuing.

    This gives a seed-dependent order that prefers high-degree nodes early.
    Then we use the same step schedule as node_removal_R:
      - for i in [n-1, n-1-nnlros, ..., 0], we remove the first i nodes
        in that degree-biased order and keep SCCs with size >= 2.
    """
    n = A.shape[0]
    if n == 0:
        out = iteration_real_R(1, None, A)
        out[-1, 3] = 0.0
        return out

    # Build degree-biased removal order
    remaining = np.arange(n)
    Acur = A.copy()
    removal_order = []

    while remaining.size > 0:
        # degrees in current graph (in+out)
        deg_out = np.asarray(Acur.sum(axis=1)).ravel()
        deg_in = np.asarray(Acur.sum(axis=0)).ravel()
        deg = deg_out + deg_in

        if np.all(deg <= 0):
            probs = np.full(remaining.size, 1.0 / remaining.size)
        else:
            probs = deg / deg.sum()

        # choose one node to remove (local index)
        idx_local = rng.choice(remaining.size, p=probs)
        node_global = remaining[idx_local]
        removal_order.append(node_global)

        # remove from remaining and from Acur
        keep_local = np.setdiff1d(np.arange(remaining.size), np.array([idx_local]))
        remaining = remaining[keep_local]
        Acur = Acur[np.ix_(keep_local, keep_local)]

    removal_order = np.array(removal_order, dtype=int)

    # Build a permutation where nodes are arranged in the order they will be removed
    perm = removal_order
    A_perm = A[np.ix_(perm, perm)]

    steps_order = list(range(n - 1, 0, -nnlros)) + [0]
    out = None
    step_idx = 0
    for i in steps_order:
        step_idx += 1
        if i > 0:
            Astep = A_perm[i:, i:]
            cluster = find_components(Astep)
            Astep = Astep[np.ix_(cluster, cluster)] if cluster.size else np.zeros((0, 0))
        else:
            Astep = A_perm.copy()
        out = iteration_real_R(step_idx, out, Astep)
        f = 1.0 - i / n
        out[-1, 3] = f

    return out


# -------------------------------- Plotting -------------------------------

def plot_panel(outputs_all: np.ndarray, simbal: int, title: str):
    if outputs_all.ndim == 2:
        outputs_all = outputs_all[None, :, :]
    Z, M, N = outputs_all.shape
    color = {1: 'red', 2: 'blue', 3: 'green'}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for z in range(Z):
        out = outputs_all[z]
        f = out[:, 3]
        x = 1.0 - f
        y = out[:, 1]  # x_eff
        ax.plot(x, y, '-', linewidth=0.6, marker='>', color=color[simbal])
    ax.plot(x, y, '-', marker='>', linewidth=2.0, markersize=6, color='black')
    ax.set_xlim(0, 1)
    if simbal == 1:
        xlabel = 'f_n'   # random node
    elif simbal == 2:
        xlabel = 'f_l'   # random link
    else:
        xlabel = 'f_t'   # targeted nodes
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel('<x>', fontsize=30)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    plt.show()


# --------------------------------- MAIN ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV file containing adjacency matrix'
    )
    ap.add_argument(
        '--perturbation',
        type=str,
        choices=['node', 'link', 'target', 'all'],
        default='all',
        help='Which perturbation to run (currently all are computed; this flag is for future use)'
    )
    ap.add_argument('--nnlros', type=int, default=1, help='Nodes/links removed per step')
    ap.add_argument('--num_reali', type=int, default=20, help='Number of realizations')
    ap.add_argument('--outdir', type=str, default='.', help='Output directory')
    ap.add_argument('--seed', type=int, default=1234, help='Random seed')
    ap.add_argument('--plot', action='store_true', help='Plot x_eff vs 1-f panels')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    A_raw = load_A_only(args.data)
    A0 = preprocess_main(A_raw, scale=2.0)
    if A0.size == 0:
        raise RuntimeError("Empty graph after preprocessing (no-input pruning + SCC≥2).")

    base = os.path.splitext(os.path.basename(args.data))[0]
    os.makedirs(args.outdir, exist_ok=True)

    res_node = os.path.join(args.outdir, 'Results_node')
    res_link = os.path.join(args.outdir, 'Results_link')
    res_target = os.path.join(args.outdir, 'Results_target')
    os.makedirs(res_node, exist_ok=True)
    os.makedirs(res_link, exist_ok=True)
    os.makedirs(res_target, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.data))[0]
    file_node   = f"{base_name}_P.mat"
    file_link   = f"{base_name}_link_P.mat"
    file_target = f"{base_name}_target_P.mat"

    # Random node removal
    outputs_node = []
    for _ in range(args.num_reali):
        out = node_removal_R(args.nnlros, A0, rng)
        outputs_node.append(out)
    outputs_node = np.stack(outputs_node, axis=0)
    savemat(os.path.join(res_node, file_node), {'outputs': outputs_node})

    # Random link removal
    outputs_link = []
    for _ in range(args.num_reali):
        out = link_removal_R(args.nnlros, A0, rng)
        outputs_link.append(out)
    outputs_link = np.stack(outputs_link, axis=0)
    savemat(os.path.join(res_link, file_link), {'outputs': outputs_link})

    # Targeted node attack (degree-biased, seed-dependent)
    outputs_target = []
    for _ in range(args.num_reali):
        out = targeted_node_removal_R(args.nnlros, A0, rng)
        outputs_target.append(out)
    outputs_target = np.stack(outputs_target, axis=0)
    savemat(os.path.join(res_target, file_target), {'outputs': outputs_target})

    if args.plot:
        for outputs0, simbal, title in [
            (outputs_node,   1, 'Random Node Removal'),
            (outputs_link,   2, 'Random Link Removal'),
            (outputs_target, 3, 'Targeted Node Attack'),
        ]:
            plot_panel(outputs0, simbal=simbal, title=title)

    print("Done.")


if __name__ == '__main__':
    main()
