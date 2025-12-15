#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regulatory NuRsE pipeline (MATLAB -> Python faithful port)

Implements the MATLAB functions you shared:
- Main_real_R preprocessing: a=2; delete_no_input; find_components (SCC size >= 2)
- R_system:  dx/dt = -x^f + A * (x^h / (x^h + 1)),  f=1, h=2
- betaspace: x_eff, beta_eff exactly as in MATLAB
- node_removal_R, link_removal_R, weight_changes_R
- iteration_real_R: integrates from x0=2 to t=400, outputs [mean, x_eff, beta_eff, f]

Usage examples:
  python figure_3.py --data TECB.mat --perturbation node --nnlros 1 --num_reali 20 --plot
  python figure_3.py --data TECB.mat --perturbation link --nnlros 5 --num_reali 5
  python figure_3.py --data TECB.mat --perturbation weight --num_reali 10 --plot
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------- I/O + Preprocess ----------------------------

def load_A_only(mat_path: str) -> np.ndarray:
    D = loadmat(mat_path)
    if "A" not in D or not isinstance(D["A"], np.ndarray):
        raise RuntimeError(f"Could not find 'A' in {mat_path}. Keys: {list(D.keys())}")
    return D["A"].astype(float)


def delete_no_input(A: np.ndarray) -> np.ndarray:
    """
    MATLAB logic:
      A = A';
      while true:
        d = sum(A); notin=find(d==0);
        A(notin,:)=[]; A(:,notin)=[];
        if no change: break
      end
      A = A';
    This removes nodes with zero in-degree iteratively until none remain.
    """
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
    """
    MATLAB logic:
      while true:
        d = sum(A); notin=find(d==0);
        A(notin,:)=[]; A(:,notin)=[];
        if no change: break
      end
    This removes nodes with zero out-degree iteratively until none remain.
    """
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
    """
    MATLAB find_components returns all nodes that belong to strongly-connected
    components with size >= 2 (it concatenates them).
    We'll reproduce using scipy's strong components.
    """
    if A.size == 0:
        return np.array([], dtype=int)
    G = csr_matrix(A != 0)
    n_comp, labels = connected_components(G, directed=True, connection="strong")
    # collect nodes in components of size >= 2
    counts = np.bincount(labels)
    keep_labels = np.where(counts >= 2)[0]
    keep_idx = np.concatenate([np.where(labels == lab)[0] for lab in keep_labels]) if keep_labels.size else np.array([], dtype=int)
    return np.sort(keep_idx)


def preprocess_main(A: np.ndarray, scale: float = 2.0) -> np.ndarray:
    # a = 2; A = sparse(A*a); delete_no_input; cluster=find_components; A = A(cluster,cluster)
    A2 = A * scale
    A2 = delete_no_input(A2)
    cluster = find_components(A2)
    if cluster.size == 0:
        return np.zeros((0, 0))
    return A2[np.ix_(cluster, cluster)]


# ------------------------------- Dynamics --------------------------------

def R_system_rhs(t: float, x: np.ndarray, A: np.ndarray, f: float = 1.0, h: float = 2.0) -> np.ndarray:
    # Fv = -x.^f + A*(x.^h./(x.^h+1))
    xx = np.maximum(x, 0.0)
    xh = xx ** h
    hill = xh / (xh + 1.0)
    return -(xx ** f) + A.dot(hill)


def integrate_to_tf(A: np.ndarray, x0: np.ndarray,
                    t0: float = 0.0, tf: float = 400.0,
                    rtol: float = 1e-6, atol: float = 1e-8) -> np.ndarray:
    if x0.size == 0:
        return x0
    fun = lambda t, x: R_system_rhs(t, x, A)
    sol = solve_ivp(fun, (t0, tf), x0, method="RK45", rtol=rtol, atol=atol, vectorized=False)
    if not sol.success:
        sol = solve_ivp(fun, (t0, tf), x0, method="BDF", rtol=rtol, atol=atol)
    return sol.y[:, -1]


def betaspace(A: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    MATLAB version:
      sAout = sum(A);
      x_nss = A*x;
      if sum(sum(A)) == 0: beta=0; x_eff=0;
      else:
        beta = sum(sum(A*A))/sum(sum(A));
        x_eff = sum(x_nss)/sum(sAout)
    """
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

def iteration_real_R(step: int, output_one: Optional[np.ndarray], A: np.ndarray,
                     paper_R: int = 0) -> np.ndarray:
    """
    MATLAB iteration_real_R:
      x0 = 2*ones(n,1); integrate to tf=400; y1 = x(end,:)';
      if paper_R==1: remove nodes y1<1.1 and re-run on subgraph; (default is 0)
      output_one(step,1) = mean(y1)
      [xnn,beta] = betaspace(A,y1)
      output_one(step,2) = xnn; output_one(step,3) = beta
      output_one(step,4) is set by the caller to fraction f
    """
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
        row = np.array([mean_y, xnn, beta, 0.0], dtype=float)  # col4 set by caller

    if output_one is None or output_one.size == 0:
        output = row.reshape(1, -1)
    else:
        output = np.vstack([output_one, row])
    return output


# ------------------------------ Perturbations ----------------------------

def node_removal_R(nnlros: int, A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    MATLAB node_removal_R:
      random permutation of nodes; for i in [n-1:-nnros:nnros, 0]:
        remove first i nodes; keep SCC>=2; iterate; set f = 1 - i/n
    """
    n = A.shape[0]
    steps_order = list(range(n-1, 0, -nnlros)) + [0]
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
    MATLAB link_removal_R:
      A = delete_no_output(A); A = delete_no_input(A); A = A(cluster,cluster);
      A0 = A; [i,j,s] = find(A0); L = length(i); the_links = randperm(L);
      for remove in [L-1:-nnros:nnlros, 0]:
        s1 = s; s1(removed_links)=0; A = sparse(i,j,s1); A(max_i_j,max_i_j)=0;
        d=sum(A); isoluted=find(d==0); A(isoluted,:)=[]; A(:,isoluted)=[];
        cluster = find_components(A); A = A(cluster,cluster);
        iteration_real_R; f = 1 - remove/L
    """
    Awork = delete_no_output(A)
    Awork = delete_no_input(Awork)
    cluster = find_components(Awork)
    Awork = Awork[np.ix_(cluster, cluster)] if cluster.size else np.zeros((0, 0))

    if Awork.size == 0:
        out = None
        # emulate at least a single step with f=0
        out = iteration_real_R(1, out, Awork)
        out[-1, 3] = 0.0
        return out

    # sparse edge list
    Aw = coo_matrix(Awork)
    rows, cols, data = Aw.row, Aw.col, Aw.data
    L = len(rows)
    the_links = rng.permutation(L)

    out = None
    step_idx = 0
    for remove in list(range(L-1, nnlros-1, -nnlros)) + [0]:
        step_idx += 1
        s1 = data.copy()
        if remove != 0:
            removed = the_links[:remove]
            s1[removed] = 0.0
        # rebuild sparse, pad to max index like MATLAB 'A(max_i_j,max_i_j)=0;'
        max_i_j = int(max(rows.max() if rows.size else 0, cols.max() if cols.size else 0))
        Anew = coo_matrix((s1, (rows, cols)), shape=(max_i_j+1, max_i_j+1)).tocsr()

        # prune isolated nodes (zero out-degree in MATLAB's 'd = sum(A)')
        d = np.asarray(Anew.sum(axis=1)).ravel()
        keep = np.where(d != 0)[0]
        if keep.size:
            Anew = Anew[keep][:, keep]
        else:
            Anew = csr_matrix((0, 0))

        # keep SCC>=2
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


def weight_changes_R(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    MATLAB weight_changes_R:
      A0 = rand(n)*2+1; A0(A==0)=0; A0 = A0 * (sum(A)/sum(A0));
      for i=0:0.02:1:  A = i*A0; iteration_real_R; f = i
    """
    n = A.shape[0]
    out = None
    if n == 0:
        out = iteration_real_R(1, out, A)
        out[-1, 3] = 0.0
        return out

    A0 = rng.random((n, n)) * 2.0 + 1.0
    A0[A == 0.0] = 0.0
    sumA = A.sum()
    sumA0 = A0.sum()
    if sumA0 > 0:
        A0 *= (sumA / sumA0)

    step_idx = 0
    for i in np.round(np.arange(0.0, 1.0 + 1e-12, 0.02), 2):
        step_idx += 1
        Acur = i * A0
        out = iteration_real_R(step_idx, out, Acur)
        out[-1, 3] = float(i)

    return out


# -------------------------------- Plotting -------------------------------

def plot_panel(outputs_all: np.ndarray, simbal: int, title: str):
    """
    Outputs have shape (Z, M, 4) where cols are:
      [mean(x), x_eff, beta_eff, f]
    We plot x_eff vs (1 - f) (like Fig.2-style).
    """
    if outputs_all.ndim == 2:
        outputs_all = outputs_all[None, :, :]
    Z, M, N = outputs_all.shape
    color={1: 'red', 2: 'blue', 3: 'green'}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for z in range(Z):
        out = outputs_all[z]
        f = out[:, 3]
        x = 1.0 - f
        y = out[:, 1]  # x_eff
        ax.plot(x, y, '-', linewidth=0.6, marker='>', color=color[simbal])
    # emphasize the last one
    ax.plot(x, y, '-', marker='>', linewidth=2.0, markersize=6, color='black')
    ax.set_xlim(0, 1)
    ax.set_xlabel('f_n' if simbal==1 else 'f_l' if simbal==2 else 'f_w', fontsize=30)
    ax.set_ylabel('<x>', fontsize=30)
    ax.set_title(title, fontsize=15)
    #ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# --------------------------------- MAIN ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Path to .mat containing A')
    ap.add_argument('--perturbation', type=str, choices=['node', 'link', 'weight', 'all'],
                    default='all', help='Which perturbation to run')
    ap.add_argument('--nnlros', type=int, default=1, help='Nodes/links removed per step')
    ap.add_argument('--num_reali', type=int, default=20, help='Number of realizations')
    ap.add_argument('--outdir', type=str, default='.', help='Output directory')
    ap.add_argument('--seed', type=int, default=1234, help='Random seed')
    ap.add_argument('--plot', action='store_true', help='Plot Fig.2-style x_eff panels')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load and preprocess as in MATLAB Main_real_R
    A_raw = load_A_only(args.data)
    A0 = preprocess_main(A_raw, scale=2.0)
    if A0.size == 0:
        raise RuntimeError("Empty graph after preprocessing (no-input pruning + SCC≥2).")

    base = os.path.splitext(os.path.basename(args.data))[0]
    os.makedirs(args.outdir, exist_ok=True)

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
        out = node_removal_R(args.nnlros, A0, rng)
        outputs_node.append(out)
    outputs_node = np.stack(outputs_node, axis=0)
    savemat(os.path.join(res_node, file_node), {'outputs': outputs_node})

    # Run link removal
    outputs_link = []
    for r in range(args.num_reali):
        out = link_removal_R(args.nnlros, A0, rng)
        outputs_link.append(out)
    outputs_link = np.stack(outputs_link, axis=0)
    savemat(os.path.join(res_link, file_link), {'outputs': outputs_link})

    # Run weight changes
    outputs_weight = []
    for r in range(args.num_reali):
        out = weight_changes_R(A0, rng)
        outputs_weight.append(out)
    outputs_weight = np.stack(outputs_weight, axis=0)
    savemat(os.path.join(res_weight, file_weight), {'outputs': outputs_weight})

    if args.plot:
        # Plot each perturbation separately
        for outputs0, simbal, title in [
            (outputs_node, 1, 'Node Removal'),
            (outputs_link, 2, 'Link Removal'),
            (outputs_weight, 3, 'Weight Change'),
        ]:
            
            fig, ax = plt.subplots(figsize=(6, 4.5))
            plot_panel(outputs0, simbal=simbal, title=title)
            ax.set_xlim(0, 1)
            ax.set_title(f'{title}')
            fig.tight_layout()
            plt.show()
    print("Done.")

if __name__ == '__main__':
    main()
