#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Optional, Tuple, Dict, List

# ---------------------------------------------------------------------
# Loader for regulatory-format outputs
# Each row: [ mean(x), x_eff, beta_eff, f ]
# Accept (T,4) or (R,T,4) -> normalize to (R,T,4)
# ---------------------------------------------------------------------

def load_outputs_mat_R(path: str) -> np.ndarray:
    D = loadmat(path, squeeze_me=True)
    if "outputs" not in D:
        raise RuntimeError(f"'outputs' not found in {path}. Keys: {list(D.keys())}")
    arr = np.asarray(D["outputs"])
    if arr.ndim == 2 and arr.shape[1] == 4:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 4:
        raise RuntimeError(f"Expected outputs shape (R,T,4) or (T,4). Got {arr.shape} in {path}")
    return arr.astype(float, copy=False)

# ---------------------------------------------------------------------
# Robust path discovery (root + Results_*; plain & [bracketed] stems)
# ---------------------------------------------------------------------

def _pick_latest(paths: List[str]) -> Optional[str]:
    paths = [p for p in paths if p and os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else None

def find_saved_paths_R(base_dir: str, stem: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Try:
      root: <stem>_{node|link|weight}_R_outputs.mat
      root: [<stem>]_{node|link|weight}_R_outputs.mat
      Results_* fallbacks: *{stem}*_R*.mat and bracketed variants
    Returns (node_path, link_path, weight_path)
    """

    def latest(*patterns: str) -> Optional[str]:
        cands = []
        for pat in patterns:
            cands.extend(glob.glob(pat))
        return _pick_latest(cands)

    # root plain
    plain_node   = os.path.join(base_dir, f"{stem}_node_R_outputs.mat")
    plain_link   = os.path.join(base_dir, f"{stem}_link_R_outputs.mat")
    plain_weight = os.path.join(base_dir, f"{stem}_weight_R_outputs.mat")
    # root bracketed
    br_node   = os.path.join(base_dir, f"[{stem}]_node_R_outputs.mat")
    br_link   = os.path.join(base_dir, f"[{stem}]_link_R_outputs.mat")
    br_weight = os.path.join(base_dir, f"[{stem}]_weight_R_outputs.mat")

    node_path = latest(plain_node, br_node) or latest(
        os.path.join(base_dir, "Results_node",   f"{stem}*_R*.mat"),
        os.path.join(base_dir, "Results_node",   f"[{stem}]*_R*.mat"),
        os.path.join(base_dir, "Results_node",   f"{stem}*.mat"),
    )
    link_path = latest(plain_link, br_link) or latest(
        os.path.join(base_dir, "Results_link",   f"{stem}*_R*.mat"),
        os.path.join(base_dir, "Results_link",   f"[{stem}]*_R*.mat"),
        os.path.join(base_dir, "Results_link",   f"{stem}*link*.mat"),
        os.path.join(base_dir, "Results_link",   f"{stem}*.mat"),
    )
    weight_path = latest(plain_weight, br_weight) or latest(
        os.path.join(base_dir, "Results_weight", f"{stem}*_R*.mat"),
        os.path.join(base_dir, "Results_weight", f"[{stem}]*_R*.mat"),
        os.path.join(base_dir, "Results_weight", f"{stem}*weight*.mat"),
        os.path.join(base_dir, "Results_weight", f"{stem}*.mat"),
    )

    return node_path, link_path, weight_path

def infer_single_stem(base_dir: str) -> str:
    """
    Infer a stem by looking for the latest *_node_R_outputs.mat in root
    (supports bracketed names), fallback to Results_node.
    """
    roots = glob.glob(os.path.join(base_dir, "*_node_R_outputs.mat"))
    roots += glob.glob(os.path.join(base_dir, "[[]*]_node_R_outputs.mat"))  # match literal [
    cands = sorted(roots, key=os.path.getmtime, reverse=True)

    if not cands:
        rn = glob.glob(os.path.join(base_dir, "Results_node", "*_R*.mat"))
        rn += glob.glob(os.path.join(base_dir, "Results_node", "[[]*]_R*.mat"))
        cands = sorted(rn, key=os.path.getmtime, reverse=True)

    if not cands:
        raise FileNotFoundError("Could not infer a stem — no regulatory node output files found.")

    fname = os.path.basename(cands[0])
    base, _ = os.path.splitext(fname)

    # strip known suffixes
    for suf in ("_node_R_outputs", "_R_node"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break

    # un-bracket [stem]
    if base.startswith("[") and base.endswith("]") and len(base) >= 2:
        base = base[1:-1]

    return base

# ---------------------------------------------------------------------
# Plotting helpers (Regulatory)
# ---------------------------------------------------------------------

def figure_panel_R(outputs0: np.ndarray, simsize: int, simbal: int, ax=None, title=None):
    """
    Regulatory Fig.2-style panel for one perturbation kind.
    outputs0: (R,T,4) or (T,4)
    x-axis: 1 - f
    y-axis: x_eff
    simbal: 1=node, 2=link, 3=weight (label only)
    """
    if outputs0.ndim == 2:
        outputs0 = outputs0[None, :, :]
    R, T, _ = outputs0.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
    color = {1: 'red', 2: 'blue', 3: 'green'}
    for r in range(R):
        out = outputs0[r]
        f = out[:, 3]; x = 1.0 - f
        y = out[:, 1]  # x_eff
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size:
            ax.plot(x, y, '-', linewidth=0.7, alpha=0.35, color=color[simbal])

    # emphasize first trajectory
    out = outputs0[0]
    f = out[:, 3]; x = 1.0 - f
    y = out[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size:
        ax.plot(
            x, y, '-',
            marker=('v' if simbal == 1 else '<' if simbal == 2 else 's'),
            linewidth=2, markersize=simsize
        )

    ax.set_xlabel('f_n' if simbal == 1 else 'f_l' if simbal == 2 else 'f_w', fontsize=13)
    ax.set_ylabel(r'$x_{\mathrm{eff}}$', fontsize=13)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    return ax

# ---------------------------------------------------------------------
# Mean-field theory (1-D) for overlay
#   dx_eff/dt = -B x_eff^f + beta_eff * x_eff^h/(x_eff^h + 1)
#   => beta_th(x) = B * (x^f + x^(f-h))
# ---------------------------------------------------------------------

def theory_parametric(B: float, f: float, h: float,
                      x_min: float, x_max: float, n: int = 2000):
    """
    1-D mean-field theory curve:
      beta(x) = B * (x^f + x^(f-h))
    Return (beta, x) sampled on x in [x_min, x_max].
    """
    x = np.linspace(x_min, x_max, n)
    x = x[(x > 0) & np.isfinite(x)]
    beta = B * (np.power(x, f) + np.power(x, f - h))
    return beta, x

def beta_c_and_xc(B: float, f: float, h: float) -> Tuple[float, float]:
    """
    Critical point where d beta / d x = 0:
      d/dx [B (x^f + x^(f-h))] = B [ f x^(f-1) + (f-h) x^(f-h-1) ] = 0
      => f x^h + (f-h) = 0  ->  x_c = ((h-f)/f)^(1/h)  (exists only if h>f)
      beta_c = B (x_c^f + x_c^(f-h))
    """
    if h <= f:
        return 0.0, 0.0
    xc = ((h - f) / f) ** (1.0 / h)
    betac = B * (xc ** f + xc ** (f - h))
    return float(betac), float(xc)

def overlay_theory(ax, beta_all: np.ndarray, B: float, f: float, h: float):
    """
    Draw the 1-D mean-field curve beta_th(x)=B*(x^f + x^(f-h)).
    We adapt x-range to the scatter so the line is guaranteed to land in view.
    """
    if beta_all.size == 0:
        return

    # current visible x-range (beta) from axis OR from data if axes not set yet
    # use data-driven bounds, then expand slightly
    bmin_data = float(np.nanmin(beta_all[beta_all > 0])) if np.any(beta_all > 0) else 1e-3
    bmax_data = float(np.nanmax(beta_all)) if beta_all.size else 10.0
    bmin = max(1e-6, bmin_data / 1.2)
    bmax = max(bmin * 1.5, bmax_data * 1.2)

    # choose x-range so that beta_th(x) spans [bmin, bmax]
    # crude but robust: scan x over a wide grid, then keep only points inside [bmin,bmax]
    x_grid = np.logspace(-3, 3, 4000)   # 1e-3 .. 1e3 covers most cases
    beta_th = B * (np.power(x_grid, f) + np.power(x_grid, f - h))

    # keep only the portion that falls inside the current beta window (and finite)
    mask = np.isfinite(beta_th) & (beta_th >= bmin) & (beta_th <= bmax) & np.isfinite(x_grid) & (x_grid > 0)
    if not np.any(mask):
        # fallback: just plot everything (maybe your cloud is empty or super tight)
        mask = np.isfinite(beta_th) & (beta_th > 0) & np.isfinite(x_grid) & (x_grid > 0)

    # draw parametric curve
    ax.plot(beta_th[mask], x_grid[mask], color='k', linestyle='--', linewidth=2.5, zorder=10, label='theory')

    # special closed form for f=1, h=2: add upper/lower branches and β_c
    if abs(f - 1.0) < 1e-12 and abs(h - 2.0) < 1e-12:
        bgrid = np.linspace(max(2.0 * B, bmin), bmax, 800)
        ratio = bgrid / max(B, 1e-12)
        disc = np.sqrt(np.maximum(0.0, ratio**2 - 4.0))
        xH = 0.5 * (ratio + disc)
        xL = 0.5 * (ratio - disc)
        ax.plot(bgrid, xH, color='k', linestyle='--', linewidth=2.5, zorder=11)                        # solid upper
        ax.plot(bgrid, xL, color='k', linestyle='--', linewidth=2.0, alpha=0.85, zorder=11)  # dashed lower
        # --- draw short critical segment instead of full vertical line ---
        # --- draw short critical segment (just above x-axis) ---
        betac = 2.0 * B
        xcrit = 0.5 * (betac / B - np.sqrt((betac / B)**2 - 4.0))
        y_bottom = 0.02 * xcrit        # small offset above axis
        y_top = 0.9 * xcrit            # stop a little before xcrit
        ax.plot([betac, betac],
                [y_bottom, y_top],
                color='k', linestyle='-', linewidth=1.8, zorder=12, clip_on=False)
        # optional base dot
        ax.scatter([betac], [y_bottom], linestyle='-',  s=25, zorder=13)

    # helpful console diagnostics
    print(f"[THEORY] B={B}, f={f}, h={h} | beta_data~[{bmin_data:.3g}, {bmax_data:.3g}] | "
          f"beta_th~[{np.nanmin(beta_th):.3g}, {np.nanmax(beta_th):.3g}] "
          f"| plotted in [{bmin:.3g}, {bmax:.3g}]")

# ---------------------------------------------------------------------
# Combined universal plot with (optional) theory overlay
# ---------------------------------------------------------------------

def plot_universal_R_multi(outputs_map: Dict[str, List[Optional[np.ndarray]]],
                           add_theory: bool, fth: float, hth: float, Bth: float,
                           title="Universal resilience (regulatory, combined)"):
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab10')
    stems = list(outputs_map.keys())
    colors = [cmap(i % 10) for i in range(len(stems))]

    plt.figure(figsize=(7.6, 6.0))
    handles = []
    any_points = False
    beta_all_concat = []

    for ci, stem in enumerate(stems):
        cols = outputs_map[stem]
        xs, betas = [], []
        for outputs in cols:
            if outputs is None:
                continue
            if outputs.ndim == 2:
                out = outputs
                xeff = out[:, 1]; beta = out[:, 2]
                mask = np.isfinite(xeff) & np.isfinite(beta) & (beta > 0)
                if np.any(mask):
                    xs.append(xeff[mask]); betas.append(beta[mask])
            else:
                for r in range(outputs.shape[0]):
                    out = outputs[r]
                    xeff = out[:, 1]; beta = out[:, 2]
                    mask = np.isfinite(xeff) & np.isfinite(beta) & (beta > 0)
                    if np.any(mask):
                        xs.append(xeff[mask]); betas.append(beta[mask])

        if not betas:
            continue

        any_points = True
        x_all = np.concatenate(xs) if xs else np.array([])
        b_all = np.concatenate(betas)
        beta_all_concat.append(b_all)
        plt.scatter(b_all, x_all, s=12, alpha=0.45, edgecolors='none', c=[colors[ci]])
        handles.append(plt.Line2D([0],[0], marker='o', linestyle='None', color=colors[ci], label=stem))

    if not any_points:
        print("[WARN] No valid points across datasets.")
        return

    # Overlay theory after plotting points
    if add_theory:
        beta_all = np.concatenate(beta_all_concat) if beta_all_concat else np.array([])
        overlay_theory(plt.gca(), beta_all, B=Bth, f=fth, h=hth)

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$\beta_{\mathrm{eff}}$', fontsize=13)
    plt.ylabel(r'$x_{\mathrm{eff}}$', fontsize=13)
    plt.title(title, fontsize=14)
    #plt.grid(False, which='both', alpha=0.3)
    if handles:
        plt.legend(handles=handles, title="Datasets", loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Loader wrapper with clearer logging
# ---------------------------------------------------------------------

def load_triplet_for_stem_R(base_dir: str, stem: str):
    node_path, link_path, weight_path = find_saved_paths_R(base_dir, stem)

    def bname(p): return os.path.basename(p) if p else None

    print(f"[INFO] Stem request: {stem}")
    print(f"[INFO]   Node   -> {node_path}   (file: {bname(node_path)})")
    print(f"[INFO]   Link   -> {link_path}   (file: {bname(link_path)})")
    print(f"[INFO]   Weight -> {weight_path} (file: {bname(weight_path)})")

    outputs_node   = load_outputs_mat_R(node_path)   if node_path   else None
    outputs_link   = load_outputs_mat_R(link_path)   if link_path   else None
    outputs_weight = load_outputs_mat_R(weight_path) if weight_path else None
    return outputs_node, outputs_link, outputs_weight

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=".",
                    help="Directory containing saved .mat outputs")
    ap.add_argument("--stems", nargs='+', default=None,
                    help='One or more dataset stems (e.g., TECB, "[file_name]").')
    ap.add_argument("--show-panels", action='store_true',
                    help="Also draw Fig.2-style panels for the FIRST stem.")
    # theory options
    ap.add_argument("--theory", action='store_true',
                    help="Overlay 1-D mean-field theoretical curve.")
    ap.add_argument("--f", type=float, default=1.0, help="Regulatory exponent f (default 1).")
    ap.add_argument("--h", type=float, default=2.0, help="Regulatory exponent h (default 2).")
    ap.add_argument("--B", type=float, default=1.0, help="Decay coefficient B (default 1).")
    args = ap.parse_args()

    stems = args.stems if args.stems else [infer_single_stem(args.base_dir)]
    print("[INFO] Stems:", stems)

    outputs_by_stem: Dict[str, List[Optional[np.ndarray]]] = {}
    for stem in stems:
        try:
            outputs_by_stem[stem] = list(load_triplet_for_stem_R(args.base_dir, stem))
        except Exception as e:
            print(f"[ERROR] Failed to load {stem}: {e}")
            outputs_by_stem[stem] = [None, None, None]

    # Panels for first stem (optional)
    if args.show_panels and stems:
        first = stems[0]
        node, link, weight = outputs_by_stem[first]
        fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
        if node is not None:   figure_panel_R(node,   simsize=6, simbal=1, ax=axs[0], title=f"{first}: Node removal")
        else:                  axs[0].set_title(f"{first}: Node removal (missing)"); axs[0].axis('off')
        if link is not None:   figure_panel_R(link,   simsize=6, simbal=2, ax=axs[1], title=f"{first}: Link removal")
        else:                  axs[1].set_title(f"{first}: Link removal (missing)"); axs[1].axis('off')
        if weight is not None: figure_panel_R(weight, simsize=6, simbal=3, ax=axs[2], title=f"{first}: Weight weakening")
        else:                  axs[2].set_title(f"{first}: Weight weakening (missing)"); axs[2].axis('off')
        fig.suptitle(f"Fig.2-style panels (regulatory): {first}", fontsize=15)
        fig.tight_layout()
        plt.show()

    # Universal with optional theory overlay
    plot_universal_R_multi(outputs_by_stem,
                           add_theory=args.theory, fth=args.f, hth=args.h, Bth=args.B,
                           title=("Universal resilience Patterns for the transcription regulatory networks of S. cerevisiae and E. coli"))

if __name__ == "__main__":
    main()
