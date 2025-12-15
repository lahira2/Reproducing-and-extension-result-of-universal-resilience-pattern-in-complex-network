#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NuRsE-style Fig.2 pipeline in pure Python
-----------------------------------------
- Loads a real mutualistic dataset (MAT file) with variables A, B, M
- Runs node, link, and weight perturbations on the plant network
- Integrates mutualistic ODEs (low/high initial states) to steady state
- Computes x_eff and beta_eff
- Saves outputs to MAT files (compatible structure)
- Plots Fig.2-style panels (node/link/weight)

Usage (example):
    python nurse_fig2_pipeline.py --data M_real_data/ANEMONE_FISH_WEBS_Coral_reefs2007.mat --type_net 2 --num_reali 20
"""

import argparse
import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# -------------------------- Utilities --------------------------

def bfs(A: np.ndarray, c: int) -> np.ndarray:
    """Breadth-first search (undirected) returning visited set from node c (0-based)."""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    queue = [c]
    visited[c] = True
    while queue:
        u = queue.pop(0)
        neigh = np.nonzero(A[u, :] != 0)[0]
        for v in neigh:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    return np.where(visited)[0]


def find_giant_component(A: np.ndarray) -> np.ndarray:
    """Return indices of the largest connected component (undirected)."""
    n = A.shape[0]
    unvisited = set(range(n))
    components = []
    while unvisited:
        start = next(iter(unvisited))
        vis = bfs(A, start)
        components.append(vis)
        for v in vis:
            if v in unvisited:
                unvisited.remove(v)
    sizes = [len(c) for c in components]
    idx = int(np.argmax(sizes))
    return components[idx]


def remove_one_effect_the_other(M: np.ndarray, type_net: int, keep_idx: np.ndarray) -> np.ndarray:
    """
    Build plant-plant A after keeping only a subset on one side of bipartite M.
    type_net == 1: keep subset of plants (rows)
    type_net == 2: keep subset of pollinators (cols)
    A = M_reduced @ M_reduced.T ; diagonal zeroed
    """
    if keep_idx is None or len(keep_idx) == 0:
        return np.zeros((0, 0), dtype=float)
    if type_net == 1:
        Mred = M[keep_idx, :]
        A = Mred @ Mred.T
    else:
        Mred = M[:, keep_idx]
        A = Mred.T @ Mred
    np.fill_diagonal(A, 0.0)
    return A


def betaspace(A: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    Compute x_eff and beta_eff from plant-plant A and steady state x.
    Undirected strengths s_i = sum_j A_ij.
    beta_eff = sum_i s_i^2 / sum_i s_i ; x_eff = (sum_i s_i x_i) / sum_i s_i
    """
    if A.size == 0:
        return float(np.mean(x) if x.size else 0.0), 0.0
    s = A.sum(axis=1)  # strengths
    S = s.sum()
    if S <= 0:
        return float(np.mean(x)), 0.0
    x_eff = float((s @ x) / S)
    beta_eff = float((s @ s) / S)
    return x_eff, beta_eff


# ----------------------- Mutualistic Dynamics -----------------------

def M_system_rhs(t: float, x: np.ndarray, A: np.ndarray,
                 K=5.0, AA=1.0, D=5.0, E=0.9, H=0.1) -> np.ndarray:
    """
    Mutualistic ODE RHS:
      dx_i/dt = 0.1 - x_i*(x_i/K - 1)*(x_i/AA - 1) + sum_j A_ij * x_i x_j / (D + E x_i + H x_j)
    """
    # self dynamics (Alle effect)
    Fv =  0.1- x * (x / K - 1.0) * (x / AA - 1.0)

    if A.size == 0:
        return Fv

    # interaction term; vectorized
    xi = x.reshape(-1, 1)
    xj = x.reshape(1, -1)
    denom = D + E * xi + H * xj
    denom = np.where(denom == 0, 1e-12, denom)
    contrib = A * (xi * xj / denom)
    Fv = Fv + contrib.sum(axis=1)
    return Fv


def integrate_to_ss(A: np.ndarray, x0: np.ndarray,
                    t0=0.0, tf=200.0, rtol=1e-6, atol=1e-8) -> np.ndarray:
    """Integrate ODE to time tf and return final state."""
    fun = lambda t, x: M_system_rhs(t, x, A)
    sol = solve_ivp(fun, (t0, tf), x0, method='RK45', rtol=rtol, atol=atol, vectorized=False)
    if not sol.success:
        sol = solve_ivp(fun, (t0, tf), x0, method='BDF', rtol=rtol, atol=atol)
    return sol.y[:, -1]


# --------------------- One-step iteration wrapper ---------------------

def iteration_real_M(step: int, A: np.ndarray, outputs_so_far: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """
    Do one iteration: integrate from low/high ICs, compute means, x_eff, beta_eff.
    Returns: outputs (updated).
    Columns of one row:
      [mean(x_low_ss), mean(x_high_ss), x_eff_low, x_eff_high, beta_eff, f]
      (f is filled by the caller)
    """
    n = A.shape[0]
    x_low0, x_high0 = 0.0, 5.0
    xl_ss = integrate_to_ss(A, np.full(n, x_low0))
    xh_ss = integrate_to_ss(A, np.full(n, x_high0))
    mean_low = float(np.mean(xl_ss))
    mean_high = float(np.mean(xh_ss))
    xl_nn, beta_eff = betaspace(A, xl_ss)
    xh_nn, _ = betaspace(A, xh_ss)

    row = np.zeros(6, dtype=float)
    row[0] = mean_low
    row[1] = mean_high
    row[2] = xl_nn
    row[3] = xh_nn
    row[4] = beta_eff
    # row[5] set by caller

    if outputs_so_far is None or outputs_so_far.size == 0:
        outputs = row.reshape(1, -1)
    else:
        outputs = np.vstack([outputs_so_far, row])
    return outputs, xl_nn, beta_eff


# --------------------------- Perturbations ---------------------------

def node_removal_M(nnlros: int, A0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Node removal on A0 with steps of nnlros. Random order, remove first i nodes, re-GCC.
    Returns output_one matrix (M x 6).
    """
    n = A0.shape[0]
    perm = rng.permutation(n)
    A0p = A0[perm][:, perm]
    outputs = None
    steps_list = list(range(n-1, 0, -nnlros)) + [0]
    for idx, i in enumerate(steps_list):
        A = A0p.copy()
        if i > 0:
            A = A[i:, i:]  # remove first i nodes
        # giant component
        if A.size == 0:
            A = np.zeros((0, 0))
        else:
            cluster = find_giant_component(A)
            A = A[np.ix_(cluster, cluster)]
        outputs, _, _ = iteration_real_M(idx, A, outputs)
        outputs[-1, 5] = 1.0 - i / n  # fraction removed
    return outputs


def link_removal_M(nnlros: int, type_net: int, M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    
    if type_net == 1:
        total = M.shape[0]  # plants (rows)
    else:
        total = M.shape[1]  # pollinators (cols)
    
    outputs = None
    order = rng.permutation(total)
    
    # Create steps: from total down to 0 in steps of nnlros
    steps = list(range(total, -1, -nnlros))
    if steps[-1] != 0:
        steps.append(0)
    
    for idx, num_to_keep in enumerate(steps):
        if num_to_keep > 0:
            keep_indices = order[:num_to_keep]
            A = remove_one_effect_the_other(M, type_net, keep_indices)
        else:
            A = np.zeros((0, 0), dtype=float)
        
        # Find giant component if network is non-empty
        if A.size > 0 and A.shape[0] > 1:
            cluster = find_giant_component(A)
            A = A[np.ix_(cluster, cluster)]
        
        outputs, _, _ = iteration_real_M(idx, A, outputs)
        outputs[-1, 5] = num_to_keep / total
    
    return outputs


def weight_changes_M(A0: np.ndarray) -> np.ndarray:
    """
    Weight perturbations: construct A0_rand = (rand + 0.5) .* A0, normalize total strength,
    then scale by i in 0..1 with step 0.02 and simulate each.
    """
    n = A0.shape[0]
    outputs = None

    # randomize weights but preserve support
    A0_rand = (np.random.rand(n, n) + 0.5) * A0
    ss = A0.sum()
    ss0 = A0_rand.sum()
    if ss0 > 0:
        A0_rand = A0_rand * (ss / ss0)
    else:
        A0_rand = A0.copy()

    for step, i in enumerate(np.round(np.arange(0.0, 1.0 + 1e-12, 0.02), 2)):
        A = i * A0_rand
        outputs, _, _ = iteration_real_M(step, A, outputs)
        outputs[-1, 5] = float(i)  # fraction of weight retained
    return outputs


def Perturpation_real_M(nnlros: int, A: np.ndarray, perturbation_type: int,
                        type_net: int, M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if perturbation_type == 1:
        return node_removal_M(nnlros, A, rng)
    elif perturbation_type == 2:
        return link_removal_M(nnlros, type_net, M, rng)
    else:
        return weight_changes_M(A)


# ------------------------------ Plotting ------------------------------

def _find_last_nonzero(x: np.ndarray) -> int:
    nz = np.nonzero(x != 0)[0]
    return int(nz[-1]) if nz.size else 0

def figure2_panel(outputs0: np.ndarray, simsize: int, simbal: int, ax=None):
    """
    Plot Fig.2-style panel for one perturbation type.
    simbal: 1=node, 2=link, 3=weight
    """
    if outputs0.ndim == 2:
        outputs0 = outputs0[None, :, :]
    m, n, z = outputs0.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))

    # color mapping per perturbation type
    color = {1: 'red', 2: 'blue', 3: 'green'}
    for i in range(m):
        outputs = outputs0[i, :, :]
        x = 1.0 - outputs[:, 5]  # 1 - f
        y1 = outputs[:, 2]       # x_eff (low)
        y2 = outputs[:, 3]       # x_eff (high)
        j = _find_last_nonzero(y1)
        j2 = min(j+1, len(x))
        # plot thin colored realization lines (colored by perturbation type)
        ax.plot(x[:j2], y1[:j2], '-', linewidth=0.5, color=color[simbal])
        ax.plot(x[:j2], y2[:j2], '-', linewidth=0.5, color=color[simbal])
        # small triangle/symbol markers to match style
        ax.plot(x[:j2], y1[:j2], linestyle='-', marker='>', markersize=4, alpha=0.6, color=color[simbal])
        ax.plot(x[:j2], y2[:j2], linestyle='-', marker='>', markersize=4, alpha=0.6, color=color[simbal])

    # emphasize one trajectory
    ax.plot(x[:j2], y1[:j2], '-', marker='>', color='k',
            linewidth=2, markersize=simsize)
    ax.plot(x[:j2], y2[:j2], '-', marker='>', color='k',
            linewidth=2, markersize=simsize)

    ax.set_xlabel('f_n' if simbal==1 else 'f_l' if simbal==2 else 'f_w', fontsize=16)
    ax.set_ylabel('<x>', fontsize=16)
    ax.grid(True, alpha=0.3)
    return ax


# ------------------------------ MAIN ------------------------------

def load_real_data(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MAT file and extract A,B,M (try common keys)."""
    D = loadmat(mat_path)
    A = D.get('A', None)
    B = D.get('B', None)
    M = D.get('M', None)

    def _maybe_unwrap(X):
        return X if (X is None or isinstance(X, np.ndarray)) else np.array(X)

    A = _maybe_unwrap(A)
    B = _maybe_unwrap(B)
    M = _maybe_unwrap(M)

    if A is None and 'plants' in D: A = D['plants']
    if B is None and 'pollinators' in D: B = D['pollinators']
    if M is None and 'bipartite' in D: M = D['bipartite']

    if A is None or B is None or M is None:
        keys = ', '.join(D.keys())
        raise RuntimeError(f"Could not find A/B/M in {mat_path}. Keys present: {keys}")

    # Ensure float arrays
    A = A.astype(float); B = B.astype(float); M = M.astype(float)
    return A, B, M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to real data .mat (with A,B,M)')
    parser.add_argument('--type_net', type=int, default=2, help='1=plants, 2=pollinators (link removal side)')
    parser.add_argument('--nnlros', type=int, default=1, help='number of nodes/links removed in one step')
    parser.add_argument('--num_reali', type=int, default=20, help='number of realizations (100 in paper)')
    parser.add_argument('--outdir', type=str, default='.', help='base output directory')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--plot', action='store_true', help='draw panels after running')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load data
    A_raw, B_raw, M = load_real_data(args.data)

    # Choose plant network A0 based on type_net (kept consistent with MATLAB)
    A0 = A_raw if args.type_net == 1 else B_raw
    # Giant component
    if A0.size == 0:
        raise RuntimeError("A0 is empty.")
    idx = find_giant_component(A0)
    A0 = A0[np.ix_(idx, idx)]

    # Prepare output folders
    res_node = os.path.join(args.outdir, 'Results_node')
    res_link = os.path.join(args.outdir, 'Results_link')
    res_weight = os.path.join(args.outdir, 'Results_weight')
    os.makedirs(res_node, exist_ok=True)
    os.makedirs(res_link, exist_ok=True)
    os.makedirs(res_weight, exist_ok=True)
    
    # ----- Dynamic filenames based on the input .mat name -----
    base_name = os.path.splitext(os.path.basename(args.data))[0]
    file_node   = f"{base_name}_P.mat"
    file_link   = f"{base_name}_link_P.mat"
    file_weight = f"{base_name}_weight_P.mat"
    # ----------------------------------------------------------

    # Run node removal
    outputs_node = []
    for r in range(args.num_reali):
        out = Perturpation_real_M(args.nnlros, A0, 1, args.type_net, M, rng)
        outputs_node.append(out)
    outputs_node = np.stack(outputs_node, axis=0)
    savemat(os.path.join(res_node, file_node), {'outputs': outputs_node})

    # Run link removal
    outputs_link = []
    for r in range(args.num_reali):
        out = Perturpation_real_M(args.nnlros, A0, 2, args.type_net, M, rng)
        outputs_link.append(out)
    outputs_link = np.stack(outputs_link, axis=0)
    savemat(os.path.join(res_link, file_link), {'outputs': outputs_link})

    # Run weight changes
    outputs_weight = []
    for r in range(args.num_reali):
        out = Perturpation_real_M(args.nnlros, A0, 3, args.type_net, M, rng)
        outputs_weight.append(out)
    outputs_weight = np.stack(outputs_weight, axis=0)
    savemat(os.path.join(res_weight, file_weight), {'outputs': outputs_weight})

    print("Saved results to:")
    print("  ", os.path.join(res_node,   file_node))
    print("  ", os.path.join(res_link,   file_link))
    print("  ", os.path.join(res_weight, file_weight))

    if args.plot:
        # Plot each perturbation separately
        for outputs0, simbal, title in [
            (outputs_node, 1, 'Node Removal'),
            (outputs_link, 2, 'Link Removal'),
            (outputs_weight, 3, 'Weight Change'),
        ]:
            #if outputs0.ndim == 2:
             #   outputs0 = outputs0[None, :, :]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            figure2_panel(outputs0, simsize=8, simbal=simbal, ax=ax)
            ax.set_xlim(0, 1)
            ax.set_title(f'{title}')
            fig.tight_layout()
            plt.show()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NuRsE-style Fig.2 pipeline in pure Python
-----------------------------------------
- Loads a real mutualistic dataset (MAT file) with variables A, B, M
- Runs node, link, and weight perturbations on the plant network
- Integrates mutualistic ODEs (low/high initial states) to steady state
- Computes x_eff and beta_eff
- Saves outputs to MAT files (compatible structure)
- Plots Fig.2-style panels (node/link/weight)

Usage (example):
    python nurse_fig2_pipeline.py --data M_real_data/ANEMONE_FISH_WEBS_Coral_reefs2007.mat --type_net 2 --num_reali 20
"""
import argparse
import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# -------------------------- Utilities --------------------------
def bfs(A: np.ndarray, c: int) -> np.ndarray:
    """Breadth-first search (undirected) returning visited set from node c (0-based)."""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    queue = [c]
    visited[c] = True
    while queue:
        u = queue.pop(0)
        neigh = np.nonzero(A[u, :] != 0)[0]
        for v in neigh:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    return np.where(visited)[0]

def find_giant_component(A: np.ndarray) -> np.ndarray:
    """Return indices of the largest connected component (undirected)."""
    n = A.shape[0]
    unvisited = set(range(n))
    components = []
    while unvisited:
        start = next(iter(unvisited))
        vis = bfs(A, start)
        components.append(vis)
        for v in vis:
            if v in unvisited:
                unvisited.remove(v)
    sizes = [len(c) for c in components]
    idx = int(np.argmax(sizes))
    return components[idx]

def remove_one_effect_the_other(M: np.ndarray, type_net: int, keep_idx: np.ndarray) -> np.ndarray:
    """
    Build plant-plant A after keeping only a subset on one side of bipartite M.
    type_net == 1: keep subset of plants (rows)
    type_net == 2: keep subset of pollinators (cols)
    A = M_reduced @ M_reduced.T ; diagonal zeroed
    """
    if keep_idx is None or len(keep_idx) == 0:
        return np.zeros((0, 0), dtype=float)
    if type_net == 1:
        Mred = M[keep_idx, :]
        A = Mred @ Mred.T
    else:
        Mred = M[:, keep_idx]
        A = Mred.T @ Mred
    np.fill_diagonal(A, 0.0)
    return A

def betaspace(A: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    Compute x_eff and beta_eff from plant-plant A and steady state x.
    Undirected strengths s_i = sum_j A_ij.
    beta_eff = sum_i s_i^2 / sum_i s_i ; x_eff = (sum_i s_i x_i) / sum_i s_i
    """
    if A.size == 0:
        return float(np.mean(x) if x.size else 0.0), 0.0
    s = A.sum(axis=1)  # strengths
    S = s.sum()
    if S <= 0:
        return float(np.mean(x)), 0.0
    x_eff = float((s @ x) / S)
    beta_eff = float((s @ s) / S)
    return x_eff, beta_eff

# ----------------------- Mutualistic Dynamics -----------------------
def M_system_rhs(t: float, x: np.ndarray, A: np.ndarray,
                 K=5.0, AA=1.0, D=5.0, E=0.9, H=0.1) -> np.ndarray:
    """
    Mutualistic ODE RHS:
      dx_i/dt = 0.1 - x_i*(x_i/K - 1)*(x_i/AA - 1) + sum_j A_ij * x_i x_j / (D + E x_i + H x_j)
    """
    # self dynamics (Alle effect)
    Fv =  - x * (x / K - 1.0) * (x / AA - 1.0)

    if A.size == 0:
        return Fv

    # interaction term; vectorized
    xi = x.reshape(-1, 1)
    xj = x.reshape(1, -1)
    denom = D + E * xi + H * xj
    denom = np.where(denom == 0, 1e-12, denom)
    contrib = A * (xi * xj / denom)
    Fv = Fv + contrib.sum(axis=1)
    return Fv

def integrate_to_ss(A: np.ndarray, x0: np.ndarray,
                    t0=0.0, tf=200.0, rtol=1e-6, atol=1e-8) -> np.ndarray:
    """Integrate ODE to time tf and return final state."""
    # If the initial state is empty (n=0) return it immediately to avoid
    # calling solve_ivp with an empty state (which leads to empty outputs
    # and numpy "mean of empty slice" warnings elsewhere).
    if x0.size == 0:
        return x0

    fun = lambda t, x: M_system_rhs(t, x, A)
    sol = solve_ivp(fun, (t0, tf), x0, method='RK45', rtol=rtol, atol=atol, vectorized=False)
    if not sol.success:
        sol = solve_ivp(fun, (t0, tf), x0, method='BDF', rtol=rtol, atol=atol)
    return sol.y[:, -1]

# --------------------- One-step iteration wrapper ---------------------
def iteration_real_M(step: int, A: np.ndarray, outputs_so_far: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """
    Do one iteration: integrate from low/high ICs, compute means, x_eff, beta_eff.
    Returns: outputs (updated).
    Columns of one row:
      [mean(x_low_ss), mean(x_high_ss), x_eff_low, x_eff_high, beta_eff, f]
      (f is filled by the caller)
    """
    n = A.shape[0]
    x_low0, x_high0 = 0.0, 5.0
    xl_ss = integrate_to_ss(A, np.full(n, x_low0))
    xh_ss = integrate_to_ss(A, np.full(n, x_high0))
    # avoid calling np.mean on empty arrays (causes a RuntimeWarning)
    mean_low = float(np.mean(xl_ss)) if xl_ss.size else 0.0
    mean_high = float(np.mean(xh_ss)) if xh_ss.size else 0.0
    xl_nn, beta_eff = betaspace(A, xl_ss)
    xh_nn, _ = betaspace(A, xh_ss)

    row = np.zeros(6, dtype=float)
    row[0] = mean_low
    row[1] = mean_high
    row[2] = xl_nn
    row[3] = xh_nn
    row[4] = beta_eff
    # row[5] set by caller

    if outputs_so_far is None or outputs_so_far.size == 0:
        outputs = row.reshape(1, -1)
    else:
        outputs = np.vstack([outputs_so_far, row])
    return outputs, xl_nn, beta_eff

# --------------------------- Perturbations ---------------------------
def node_removal_M(nnlros: int, A0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Node removal on A0 with steps of nnlros. Random order, remove first i nodes, re-GCC.
    Returns output_one matrix (M x 6).
    """
    n = A0.shape[0]
    perm = rng.permutation(n)
    A0p = A0[perm][:, perm]
    outputs = None
    steps_list = list(range(n-1, 0, -nnlros)) + [0]
    for idx, i in enumerate(steps_list):
        A = A0p.copy()
        if i > 0:
            A = A[i:, i:]  # remove first i nodes
        # giant component
        if A.size == 0:
            A = np.zeros((0, 0))
        else:
            cluster = find_giant_component(A)
            A = A[np.ix_(cluster, cluster)]
        outputs, _, _ = iteration_real_M(idx, A, outputs)
        outputs[-1, 5] = 1.0 - i / n  # fraction removed
    return outputs

def link_removal_M(nnlros: int, type_net: int, M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if type_net == 1:
        total = M.shape[0]  # plants (rows)
    else:
        total = M.shape[1]  # pollinators (cols)
    outputs = None
    order = rng.permutation(total)
    # Create steps: from total down to 0 in steps of nnlros
    steps = list(range(total, -1, -nnlros))
    if steps[-1] != 0:
        steps.append(0)
    for idx, num_to_keep in enumerate(steps):
        if num_to_keep > 0:
            keep_indices = order[:num_to_keep]
            A = remove_one_effect_the_other(M, type_net, keep_indices)
        else:
            A = np.zeros((0, 0), dtype=float)
        # Find giant component if network is non-empty
        if A.size > 0 and A.shape[0] > 1:
            cluster = find_giant_component(A)
            A = A[np.ix_(cluster, cluster)]
        outputs, _, _ = iteration_real_M(idx, A, outputs)
        outputs[-1, 5] = num_to_keep / total
    return outputs

def weight_changes_M(A0: np.ndarray) -> np.ndarray:
    """
    Weight perturbations: construct A0_rand = (rand + 0.5) .* A0, normalize total strength,
    then scale by i in 0..1 with step 0.02 and simulate each.
    """
    n = A0.shape[0]
    outputs = None
    # randomize weights but preserve support
    A0_rand = (np.random.rand(n, n) + 0.5) * A0
    ss = A0.sum()
    ss0 = A0_rand.sum()
    if ss0 > 0:
        A0_rand = A0_rand * (ss / ss0)
    else:
        A0_rand = A0.copy()
    for step, i in enumerate(np.round(np.arange(0.0, 1.0 + 1e-12, 0.02), 2)):
        A = i * A0_rand
        outputs, _, _ = iteration_real_M(step, A, outputs)
        outputs[-1, 5] = float(i)  # fraction of weight retained
    return outputs

def Perturpation_real_M(nnlros: int, A: np.ndarray, perturbation_type: int,
                        type_net: int, M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if perturbation_type == 1:
        return node_removal_M(nnlros, A, rng)
    elif perturbation_type == 2:
        return link_removal_M(nnlros, type_net, M, rng)
    else:
        return weight_changes_M(A)

# ------------------------------ Plotting ------------------------------
def _find_last_nonzero(x: np.ndarray) -> int:
    nz = np.nonzero(x != 0)[0]
    return int(nz[-1]) if nz.size else 0

def figure2_panel(outputs0: np.ndarray, simsize: int, simbal: int, ax=None):
    """
    Plot Fig.2-style panel for one perturbation type.
    simbal: 1=node, 2=link, 3=weight
    """
    if outputs0.ndim == 2:
        outputs0 = outputs0[None, :, :]
    m, n, z = outputs0.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    color = {1: 'red', 2: 'blue', 3: 'green'}
    for i in range(m):
        outputs = outputs0[i, :, :]
        x = 1.0 - outputs[:, 5]  # 1 - f
        y1 = outputs[:, 2]       # x_eff (low)
        y2 = outputs[:, 3]       # x_eff (high)
        j = _find_last_nonzero(y1)
        j2 = min(j+1, len(x))
        ax.plot(x[:j2], y1[:j2], '-', linewidth=0.5, marker='>', color=color[simbal])
        ax.plot(x[:j2], y2[:j2], '-', linewidth=0.5, marker='>', color=color[simbal])
    # emphasize one trajectory
    ax.plot(x[:j2], y1[:j2], '-', marker='>', color='black',
            linewidth=2, markersize=simsize)
    ax.plot(x[:j2], y2[:j2], '-', marker='>', color='black',
            linewidth=2, markersize=simsize)
    ax.set_xlabel('f_n' if simbal==1 else 'f_l' if simbal==2 else 'f_w', fontsize=16)
    ax.set_ylabel('<x>', fontsize=16)
    #ax.grid(True, alpha=0.3)
    return ax

# ------------------------------ MAIN ------------------------------
def load_real_data(mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MAT file and extract A,B,M (try common keys)."""
    D = loadmat(mat_path)
    A = D.get('A', None)
    B = D.get('B', None)
    M = D.get('M', None)
    def _maybe_unwrap(X):
        return X if (X is None or isinstance(X, np.ndarray)) else np.array(X)
    A = _maybe_unwrap(A)
    B = _maybe_unwrap(B)
    M = _maybe_unwrap(M)
    if A is None and 'plants' in D: A = D['plants']
    if B is None and 'pollinators' in D: B = D['pollinators']
    if M is None and 'bipartite' in D: M = D['bipartite']
    if A is None or B is None or M is None:
        keys = ', '.join(D.keys())
        raise RuntimeError(f"Could not find A/B/M in {mat_path}. Keys present: {keys}")
    # Ensure float arrays
    A = A.astype(float); B = B.astype(float); M = M.astype(float)
    return A, B, M

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to real data .mat (with A,B,M)')
    parser.add_argument('--type_net', type=int, default=2, help='1=plants, 2=pollinators (link removal side)')
    parser.add_argument('--nnlros', type=int, default=1, help='number of nodes/links removed in one step')
    parser.add_argument('--num_reali', type=int, default=20, help='number of realizations (100 in paper)')
    parser.add_argument('--outdir', type=str, default='.', help='base output directory')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--plot', action='store_true', help='draw panels after running')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    # Load data
    A_raw, B_raw, M = load_real_data(args.data)
    # Choose plant network A0 based on type_net (kept consistent with MATLAB)
    A0 = A_raw if args.type_net == 1 else B_raw
    # Giant component
    if A0.size == 0:
        raise RuntimeError("A0 is empty.")
    idx = find_giant_component(A0)
    A0 = A0[np.ix_(idx, idx)]
    # Prepare output folders
    res_node = os.path.join(args.outdir, 'Results_node')
    res_link = os.path.join(args.outdir, 'Results_link')
    res_weight = os.path.join(args.outdir, 'Results_weight')
    os.makedirs(res_node, exist_ok=True)
    os.makedirs(res_link, exist_ok=True)
    os.makedirs(res_weight, exist_ok=True)
    # ----- Dynamic filenames based on the input .mat name -----
    base_name = os.path.splitext(os.path.basename(args.data))[0]
    file_node   = f"{base_name}_P.mat"
    file_link   = f"{base_name}_link_P.mat"
    file_weight = f"{base_name}_weight_P.mat"
    # ----------------------------------------------------------
    # Run node removal
    outputs_node = []
    for r in range(args.num_reali):
        out = Perturpation_real_M(args.nnlros, A0, 1, args.type_net, M, rng)
        outputs_node.append(out)
    outputs_node = np.stack(outputs_node, axis=0)
    savemat(os.path.join(res_node, file_node), {'outputs': outputs_node})
    # Run link removal
    outputs_link = []
    for r in range(args.num_reali):
        out = Perturpation_real_M(args.nnlros, A0, 2, args.type_net, M, rng)
        outputs_link.append(out)
    outputs_link = np.stack(outputs_link, axis=0)
    savemat(os.path.join(res_link, file_link), {'outputs': outputs_link})
    # Run weight changes
    outputs_weight = []
    for r in range(args.num_reali):
        out = Perturpation_real_M(args.nnlros, A0, 3, args.type_net, M, rng)
        outputs_weight.append(out)
    outputs_weight = np.stack(outputs_weight, axis=0)
    savemat(os.path.join(res_weight, file_weight), {'outputs': outputs_weight})
    print("Saved results to:")
    print("  ", os.path.join(res_node,   file_node))
    print("  ", os.path.join(res_link,   file_link))
    print("  ", os.path.join(res_weight, file_weight))
    if args.plot:
        # Plot each perturbation separately
        for outputs0, simbal, title in [
            (outputs_node, 1, 'Node Removal'),
            (outputs_link, 2, 'Link Removal'),
            (outputs_weight, 3, 'Weight Change'),
        ]:
            #if outputs0.ndim == 2:
             #   outputs0 = outputs0[None, :, :]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            figure2_panel(outputs0, simsize=8, simbal=simbal, ax=ax)
            ax.set_xlim(0, 1)
            ax.set_title(f'{title}')
            fig.tight_layout()
            plt.show()
    print("Done.")
if __name__ == '__main__':
    main()
