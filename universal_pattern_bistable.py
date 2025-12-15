#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# =====================================================================
# 1. Load Laplacian outputs
# =====================================================================

def load_outputs_mat(path: str) -> np.ndarray:
    """
    Load 'outputs' from a .mat file and normalize to shape (R, T, 4).

    Each row: [ mean_x, x_eff, beta_eff, f ]
    """
    D = loadmat(path, squeeze_me=True)
    if "outputs" not in D:
        raise RuntimeError(f"'outputs' not found in {path}. Keys: {list(D.keys())}")
    arr = np.asarray(D["outputs"])
    if arr.ndim == 2 and arr.shape[1] == 4:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise RuntimeError(f"Expected outputs shape (R,T,4) or (T,4). Got {arr.shape} in {path}")
    return arr.astype(float, copy=False)


def find_saved_paths_laplacian(base_dir: str, stem: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Laplacian naming scheme:

      Node:   Results_node/<stem>_P.mat
      Link:   Results_link/<stem>_link_P.mat
      Target: Results_target/<stem>_target_P.mat

    Returns (node_path, link_path, target_path)
    """
    node_path   = os.path.join(base_dir, "Results_node",   f"{stem}_P.mat")
    link_path   = os.path.join(base_dir, "Results_link",   f"{stem}_link_P.mat")
    target_path = os.path.join(base_dir, "Results_target", f"{stem}_target_P.mat")

    node_path   = node_path   if os.path.isfile(node_path)   else None
    link_path   = link_path   if os.path.isfile(link_path)   else None
    target_path = target_path if os.path.isfile(target_path) else None

    return node_path, link_path, target_path


def infer_single_stem(base_dir: str) -> str:
    """
    Infer a dataset stem from Results_node/*_P.mat, e.g.:

      Results_node/celegans131matrix_P.mat  -> stem: 'celegans131matrix'
    """
    cands = glob.glob(os.path.join(base_dir, "Results_node", "*_P.mat"))
    if not cands:
        raise FileNotFoundError("Could not infer a stem — no Laplacian node output files found in Results_node.")
    latest = max(cands, key=os.path.getmtime)
    base = os.path.basename(latest)      # e.g. "celegans131matrix_P.mat"
    stem, _ = os.path.splitext(base)     # "celegans131matrix_P"
    if stem.endswith("_P"):
        stem = stem[:-2]
    return stem


def load_triplet_for_stem(base_dir: str, stem: str):
    """
    Load (node, link, target) outputs for one dataset stem.
    """
    node_path, link_path, target_path = find_saved_paths_laplacian(base_dir, stem)

    def bname(p):
        return os.path.basename(p) if p else None

    print(f"[INFO] Stem: {stem}")
    print(f"[INFO]   Node   -> {node_path}   (file: {bname(node_path)})")
    print(f"[INFO]   Link   -> {link_path}   (file: {bname(link_path)})")
    print(f"[INFO]   Target -> {target_path} (file: {bname(target_path)})")

    outputs_node   = load_outputs_mat(node_path)   if node_path   else None
    outputs_link   = load_outputs_mat(link_path)   if link_path   else None
    outputs_target = load_outputs_mat(target_path) if target_path else None
    return outputs_node, outputs_link, outputs_target


# =====================================================================
# 2. Panel plots (Fig.2-style) for one dataset
# =====================================================================

def figure_panel(outputs0: np.ndarray, simsize: int, simbal: int, ax=None, title=None):
    """
    Fig.2-style panel for one perturbation.

    outputs0: (R,T,4) or (T,4)
    x-axis: 1 - f
    y-axis: x_eff

    simbal: 1=node, 2=link, 3=target (label only)
    """
    if outputs0.ndim == 2:
        outputs0 = outputs0[None, :, :]
    R, T, _ = outputs0.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))

    color = {1: 'red', 2: 'blue', 3: 'green'}

    # plot all realizations as faint lines
    for r in range(R):
        out = outputs0[r]
        f = out[:, 3]
        x = 1.0 - f
        y = out[:, 1]  # x_eff
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size:
            ax.plot(x, y, '-', linewidth=0.7, alpha=0.35, color=color[simbal])

    # emphasize the first realization
    out = outputs0[0]
    f = out[:, 3]
    x = 1.0 - f
    y = out[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size:
        marker = 'v' if simbal == 1 else '<' if simbal == 2 else 's'
        ax.plot(x, y, '-', marker=marker, linewidth=2, markersize=simsize)

    if simbal == 1:
        xlabel = 'f_n'   # random node
    elif simbal == 2:
        xlabel = 'f_l'   # random link
    else:
        xlabel = 'f_t'   # targeted node

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r'$x_{\mathrm{eff}}$', fontsize=13)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    return ax


# =====================================================================
# 3. Universal x_eff vs beta_eff plot (multi-stem) + theory
# =====================================================================

def plot_universal_multi(outputs_map: Dict[str, List[Optional[np.ndarray]]],
                         title="Universal resilience patterns (Laplacian dynamics)"):
    """
    Create a universal x_eff vs beta_eff scatter plot combining multiple datasets.
    X-axis uses 1/beta_eff (as in your current code), Y-axis is x_eff.
    Then fit a mean-field relation x_eff^2 ≈ A - B * beta_eff and overlay
    the corresponding theoretical curve (converted to the same x-axis variable 1/beta).
    """
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab10')

    stems = list(outputs_map.keys())
    colors = [cmap(i % 10) for i in range(len(stems))]

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    handles = []
    any_points = False

    # collect all points for theory fit
    all_x2 = []
    all_beta = []

    for ci, stem in enumerate(stems):
        cols = outputs_map[stem]   # [node, link, target]
        xs, betas = [], []

        for outputs in cols:
            if outputs is None:
                continue
            if outputs.ndim == 2:
                out = outputs
                xeff = out[:, 1]
                beta = out[:, 2]
                mask = np.isfinite(xeff) & np.isfinite(beta) & (beta > 0)
                if np.any(mask):
                    xs.append(xeff[mask])
                    betas.append(beta[mask])
            else:
                for r in range(outputs.shape[0]):
                    out = outputs[r]
                    xeff = out[:, 1]
                    beta = out[:, 2]
                    mask = np.isfinite(xeff) & np.isfinite(beta) & (beta > 0)
                    if np.any(mask):
                        xs.append(xeff[mask])
                        betas.append(beta[mask])

        if not betas:
            continue

        any_points = True
        x_all = np.concatenate(xs) if xs else np.array([])
        b_all = np.concatenate(betas)
        b_inv = 1.0 / b_all

        # store for theory fit
        all_x2.append(x_all**2)
        all_beta.append(b_all)

        # scatter data: x_eff vs 1/beta_eff
        ax.scatter(
            b_inv, x_all,
            s=12, alpha=0.45, edgecolors='none',
            c=[colors[ci]]
        )

        handles.append(
            plt.Line2D(
                [0], [0],
                marker='o', linestyle='None',
                color=colors[ci],
                label=stem
            )
        )

    if not any_points:
        print("[WARN] No valid points across datasets. Nothing to plot.")
        return

    # ----------------- mean-field theory fit: x_eff^2 ≈ A - B * beta -----------------
    all_x2 = np.concatenate(all_x2)
    all_beta = np.concatenate(all_beta)

    mask_fit = np.isfinite(all_x2) & np.isfinite(all_beta) & (all_beta > 0)
    beta_fit = all_beta[mask_fit]
    x2_fit = all_x2[mask_fit]

    if beta_fit.size >= 2:
        # Linear least squares: x2 ≈ A - B * beta
        # => x2 = [1, -beta] @ [A, B]^T
        M = np.column_stack([np.ones_like(beta_fit), -beta_fit])
        theta, _, _, _ = np.linalg.lstsq(M, x2_fit, rcond=None)
        A, B = theta[0], theta[1]

        # critical beta where x_eff -> 0 in this approximation
        beta_c = A / B if B != 0 else np.inf

        print(f"[THEORY] Fitted A (≈ a/b) = {A:.4f},  B (≈ σC/b) = {B:.4f},  beta_c = A/B ≈ {beta_c:.4f}")

        # build theoretical curve in terms of beta
        beta_th = np.linspace(beta_fit.min(), beta_fit.max(), 400)
        x2_th = A - B * beta_th
        x2_th[x2_th <= 0] = np.nan
        x_th = np.sqrt(x2_th)

        # convert to the same x-axis variable used for data: 1/beta
        s_th = 1.0 / beta_th
        mask_th = np.isfinite(s_th) & np.isfinite(x_th)

        ax.plot(
            s_th[mask_th],
            x_th[mask_th],
            'k-',
            linewidth=2.5,
            label='mean-field theory'
        )
    else:
        print("[THEORY] Not enough points to fit theory.")

    # -------------------------------- axis formatting --------------------------------
    # ax.xscale('log')  # keep commented if you want linear x
    ax.set_yscale('log')
    ax.set_xlabel(r'$1 / \beta_{\mathrm{eff}}$' , fontsize=13)
    ax.set_ylabel(r'$x_{\mathrm{eff}}$', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if handles:
        handles.append(
            plt.Line2D([0], [0], color='k', linewidth=2.5, label='mean-field theory')
        )
        fig.legend(handles=handles, title="Datasets", loc='lower center',
                   ncol=max(1, len(stems) + 1), frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.show()


# =====================================================================
# 4. Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir", type=str, default=".",
        help="Directory containing Results_node, Results_link, Results_target"
    )
    ap.add_argument(
        "--stems", nargs='+', default=None,
        help='One or more dataset stems (e.g., celegans131matrix). '
             'If omitted, will try to infer from Results_node/*_P.mat'
    )
    ap.add_argument(
        "--show-panels", action='store_true',
        help="Also draw Fig.2-style panels for the FIRST stem."
    )
    args = ap.parse_args()

    stems = args.stems if args.stems else [infer_single_stem(args.base_dir)]
    print("[INFO] Stems:", stems)

    outputs_by_stem: Dict[str, List[Optional[np.ndarray]]] = {}
    for stem in stems:
        try:
            node, link, target = load_triplet_for_stem(args.base_dir, stem)
            outputs_by_stem[stem] = [node, link, target]
        except Exception as e:
            print(f"[ERROR] Failed to load {stem}: {e}")
            outputs_by_stem[stem] = [None, None, None]

    # Optional: Fig.2-style panels for first stem
    if args.show_panels and stems:
        first = stems[0]
        node, link, target = outputs_by_stem[first]

        fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))

        if node is not None:
            figure_panel(node, simsize=6, simbal=1, ax=axs[0], title=f"{first}: Random node removal")
        else:
            axs[0].set_title(f"{first}: Random node removal (missing)")
            axs[0].axis('off')

        if link is not None:
            figure_panel(link, simsize=6, simbal=2, ax=axs[1], title=f"{first}: Random link removal")
        else:
            axs[1].set_title(f"{first}: Random link removal (missing)")
            axs[1].axis('off')

        if target is not None:
            figure_panel(target, simsize=6, simbal=3, ax=axs[2], title=f"{first}: Targeted node attack")
        else:
            axs[2].set_title(f"{first}: Targeted node attack (missing)")
            axs[2].axis('off')

        fig.suptitle(f"{first}", fontsize=15)
        fig.tight_layout()
        plt.show()

    # Universal x_eff vs beta_eff plot (combined) + theory overlay
    plot_universal_multi(
        outputs_by_stem,
        title="Universal resilience patterns (Laplacian dynamics)"
    )


if __name__ == "__main__":
    main()
