#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Optional, Tuple, Dict, List

# ========================= IO: 6-column outputs =========================
# row = [mean_low, mean_high, x_eff_low, x_eff_high, beta_eff, f]
# Accept (T,6) or (R,T,6); normalize to (R,T,6)

def load_outputs_mat(path: str) -> np.ndarray:
    D = loadmat(path, squeeze_me=True)
    if 'outputs' not in D:
        raise RuntimeError(f"'outputs' not found in {path}. Keys: {list(D.keys())}")
    arr = np.asarray(D['outputs'])
    if arr.ndim == 2 and arr.shape[1] == 6:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise RuntimeError(f"outputs in {path} must be (R,T,6) or (T,6). Got {arr.shape}")
    return arr.astype(float, copy=False)

def _find_last_nonzero(x: np.ndarray) -> int:
    nz = np.nonzero(x != 0)[0]
    return int(nz[-1]) if nz.size else 0

# ========================= Fig.2-style panels =========================

def figure2_panel(outputs0: np.ndarray, simsize: int, simbal: int, ax=None, title=None):
    if outputs0.ndim == 2:
        outputs0 = outputs0[None, :, :]
    R, T, _ = outputs0.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
    color = {1: 'red', 2: 'blue', 3: 'green'}
    for r in range(R):
        out = outputs0[r]
        x = 1.0 - out[:, 5]          # 1 - f
        y1 = out[:, 2]               # x_eff low
        y2 = out[:, 3]               # x_eff high
        m = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
        x, y1, y2 = x[m], y1[m], y2[m]
        if x.size == 0: continue
        j2 = min(_find_last_nonzero(y1) + 1, len(x))
        ax.plot(x[:j2], y1[:j2], '-', linewidth=0.5, marker='>', color=color[simbal])
        ax.plot(x[:j2], y2[:j2], '-', linewidth=0.5, marker='^', color=color[simbal])

    out = outputs0[0]
    x = 1.0 - out[:, 5]
    y1 = out[:, 2]
    y2 = out[:, 3]
    m = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    x, y1, y2 = x[m], y1[m], y2[m]
    if x.size:
        j2 = min(_find_last_nonzero(y1) + 1, len(x))
        ax.plot(x[:j2], y1[:j2], '-', lw=2, ms=simsize, marker='>', color='black')
        ax.plot(x[:j2], y2[:j2], '-', lw=2, ms=simsize, marker='>', color='black')

    ax.set_xlabel('f_n' if simbal==1 else 'f_l' if simbal==2 else 'f_w', fontsize=30)
    ax.set_ylabel('<x>', fontsize=30)
    if title: ax.set_title(title, fontsize=15)
    #ax.grid(True, alpha=0.3)
    return ax

# ========================= Universal scatter (6-col) =========================

def plot_universal_resilience(outputs_list, title="Universal resilience function (Fig.2n-like)"):
    xs_low, xs_high, betas = [], [], []
    for outputs in outputs_list:
        if outputs is None: continue
        if outputs.ndim == 2:
            out = outputs
            xL = out[:, 2]; xH = out[:, 3]; beta = out[:, 4]
            m = np.isfinite(xL) & np.isfinite(xH) & np.isfinite(beta) & (beta > 0)
            xs_low.append(xL[m]); xs_high.append(xH[m]); betas.append(beta[m])
        else:
            for r in range(outputs.shape[0]):
                out = outputs[r]
                xL = out[:, 2]; xH = out[:, 3]; beta = out[:, 4]
                m = np.isfinite(xL) & np.isfinite(xH) & np.isfinite(beta) & (beta > 0)
                xs_low.append(xL[m]); xs_high.append(xH[m]); betas.append(beta[m])

    if not betas:
        print("[WARN] No valid points for universal resilience plot.")
        return

    xL_all = np.concatenate(xs_low) if xs_low else np.array([])
    xH_all = np.concatenate(xs_high) if xs_high else np.array([])
    beta_all = np.concatenate(betas)

    plt.figure(figsize=(6.2, 5.0))
    if xL_all.size:
        plt.scatter(beta_all, xL_all, s=7, c='#d62728', alpha=0.35, label=r'$x_\mathrm{eff}^{L}$')
    if xH_all.size:
        plt.scatter(beta_all, xH_all, s=7, c='#1f77b4', alpha=0.35, label=r'$x_\mathrm{eff}^{H}$')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'$\beta_{\mathrm{eff}}$', fontsize=13)
    plt.ylabel(r'$x_{\mathrm{eff}}$', fontsize=13)
    plt.title(title, fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

# ========================= File discovery =========================

def _pick_latest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else None

def _expected_paths(base_dir: str, stem: str) -> Tuple[str, str, str]:
    node = os.path.join(base_dir, "Results_node",   f"{stem}_P.mat")
    link = os.path.join(base_dir, "Results_link",   f"{stem}_link_P.mat")
    weight = os.path.join(base_dir, "Results_weight", f"{stem}_weight_P.mat")
    return node, link, weight

def find_saved_paths(base_dir: str, stem_hint: str):
    node_exact, link_exact, weight_exact = _expected_paths(base_dir, stem_hint)
    node_path   = node_exact   if os.path.isfile(node_exact)   else None
    link_path   = link_exact   if os.path.isfile(link_exact)   else None
    weight_path = weight_exact if os.path.isfile(weight_exact) else None

    if node_path is None:
        node_globs = [
            os.path.join(base_dir, "Results_node", f"{stem_hint}*_P.mat"),
            os.path.join(base_dir, "Results_node", f"{stem_hint}*.mat"),
        ]
        node_path = _pick_latest([p for g in node_globs for p in glob.glob(g)])

    if link_path is None:
        link_globs = [
            os.path.join(base_dir, "Results_link", f"{stem_hint}*link*_P.mat"),
            os.path.join(base_dir, "Results_link", f"{stem_hint}*link*.mat"),
            os.path.join(base_dir, "Results_link", f"{stem_hint}*_P.mat"),
        ]
        link_path = _pick_latest([p for g in link_globs for p in glob.glob(g)])

    if weight_path is None:
        weight_globs = [
            os.path.join(base_dir, "Results_weight", f"{stem_hint}*weight*_P.mat"),
            os.path.join(base_dir, "Results_weight", f"{stem_hint}*weight*.mat"),
            os.path.join(base_dir, "Results_weight", f"{stem_hint}*_P.mat"),
        ]
        weight_path = _pick_latest([p for g in weight_globs for p in glob.glob(g)])

    return node_path, link_path, weight_path

def infer_stem_if_missing(base_dir: str, stem: Optional[str]) -> str:
    if stem:
        return stem
    node_dir = os.path.join(base_dir, "Results_node")
    cands = sorted(glob.glob(os.path.join(node_dir, "*_P.mat")), key=os.path.getmtime, reverse=True)
    if not cands:
        raise FileNotFoundError("Could not infer --stem: no files like *_P.mat in Results_node.")
    fname = os.path.basename(cands[0])
    return fname[:-len("_P.mat")] if fname.endswith("_P.mat") else os.path.splitext(fname)[0]

# ========================= Theory overlay (your equation) =========================

def beta_theory(x, B, K, C, D, E, H):
    """β_th(x) = -[B + x(1-x/K)(x/C-1)] * (D + (E+H)x) / x^2"""
    x = np.asarray(x, dtype=float)
    num = B + x * (1.0 - x / K) * (x / C - 1.0)
    denom = D + (E + H) * x
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = - num * denom / (x**2)
    return beta

def overlay_theory(ax, beta_data: np.ndarray,
                   B: float, K: float, C: float, D: float, E: float, H: float):
    """
    Draw the full theoretical curve but break it into contiguous segments so
    Matplotlib doesn't connect across gaps (which creates a vertical line).
    """
    if beta_data.size == 0:
        return

    # data-driven beta window
    bpos = beta_data[beta_data > 0]
    if bpos.size == 0:
        return
    bmin = float(np.nanmin(bpos)) * 0.8
    bmax = float(np.nanmax(bpos)) * 1.2

    # dense x-grid and theory
    x_grid = np.logspace(-4, 4, 8000)
    beta_th = beta_theory(x_grid, B, K, C, D, E, H)

    # valid region within the plot window
    m = np.isfinite(beta_th) & (beta_th > 0) & (beta_th >= bmin) & (beta_th <= bmax)

    if not np.any(m):
        return

    # --- KEY PART: split into contiguous True segments and plot each separately ---
    idx = np.where(m)[0]
    # break points where the index jumps by more than 1
    breaks = np.where(np.diff(idx) > 1)[0]
    # segments are between [start,end] inclusive over idx
    starts = np.r_[0, breaks + 1]
    ends   = np.r_[breaks, len(idx) - 1]

    for s, e in zip(starts, ends):
        seg = idx[s:e+1]
        xv = x_grid[seg]
        bv = beta_th[seg]
        # keep your original solid style; if you prefer dashed for the lower branch, change to 'k--'
        ax.plot(bv, xv, color='k', linestyle='--', lw=1, zorder=10)

    print(f"[THEORY] B={B}, K={K}, C={C}, D={D}, E={E}, H={H} | "
          f"beta window [{bmin:.3g}, {bmax:.3g}] | "
          f"segments plotted: {len(starts)}")



# ========================= Universal (multi) + theory =========================

def plot_universal_resilience_multi(outputs_map: Dict[str, List[Optional[np.ndarray]]],
                                    title="Universal resilience (combined)",
                                    add_theory: bool=False,
                                    B: float=0.1, K: float=5.0, C: float=1.0, D: float=5.0, E: float=0.9, H: float=0.1):
    from matplotlib import colormaps
    cmap = colormaps.get_cmap('tab10')
    stems = list(outputs_map.keys())
    color_cycle = [cmap(i % 10) for i in range(len(stems))]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    any_points = False
    beta_all_concat = []
    legend_handles = []

    for ci, stem in enumerate(stems):
        color = color_cycle[ci]
        xs_low, xs_high, betas = [], [], []
        for outputs in outputs_map[stem]:  # [node, link, weight]
            if outputs is None: continue
            if outputs.ndim == 2:
                out = outputs
                xL = out[:, 2]; xH = out[:, 3]; beta = out[:, 4]
                m = np.isfinite(xL) & np.isfinite(xH) & np.isfinite(beta) & (beta > 0)
                if np.any(m): xs_low.append(xL[m]); xs_high.append(xH[m]); betas.append(beta[m])
            else:
                for r in range(outputs.shape[0]):
                    out = outputs[r]
                    xL = out[:, 2]; xH = out[:, 3]; beta = out[:, 4]
                    m = np.isfinite(xL) & np.isfinite(xH) & np.isfinite(beta) & (beta > 0)
                    if np.any(m): xs_low.append(xL[m]); xs_high.append(xH[m]); betas.append(beta[m])

        if not betas: continue
        any_points = True
        xL_all = np.concatenate(xs_low) if xs_low else np.array([])
        xH_all = np.concatenate(xs_high) if xs_high else np.array([])
        beta_all = np.concatenate(betas)
        beta_all_concat.append(beta_all)

        if xL_all.size:
            ax.scatter(beta_all, xL_all, s=12, alpha=0.45, marker='o', edgecolors='none', c=[color], zorder=1)
        if xH_all.size:
            ax.scatter(beta_all, xH_all, s=12, alpha=0.45, marker='^', edgecolors='none', c=[color], zorder=1)

        proxy = plt.Line2D([0],[0], marker='o', linestyle='None', color=color, label=stem)
        legend_handles.append(proxy)

    if not any_points:
        print("[WARN] No valid points across datasets.")
        return

    if add_theory:
        beta_all = np.concatenate(beta_all_concat) if beta_all_concat else np.array([])
        overlay_theory(ax, beta_all, B=B, K=K, C=C, D=D, E=E, H=H)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\beta_{\mathrm{eff}}$', fontsize=13)
    ax.set_ylabel(r'$x_{\mathrm{eff}}$', fontsize=13)
    ax.set_title(title, fontsize=14)
    #ax.grid(False, which='both', alpha=0.3)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Datasets", loc='lower right', frameon=False)
    fig.tight_layout()
    plt.show()

# ========================= Load trio for a stem =========================

def load_triplet_for_stem(base_dir: str, stem: str):
    node_path, link_path, weight_path = find_saved_paths(base_dir, stem)
    print(f"[INFO] [{stem}] Node:   {node_path}")
    print(f"[INFO] [{stem}] Link:   {link_path}")
    print(f"[INFO] [{stem}] Weight: {weight_path}")
    outputs_node   = load_outputs_mat(node_path)    if node_path    else None
    outputs_link   = load_outputs_mat(link_path)    if link_path    else None
    outputs_weight = load_outputs_mat(weight_path)  if weight_path  else None
    return outputs_node, outputs_link, outputs_weight

# ========================= Main =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=".",
                    help="Directory that contains Results_node/Results_link/Results_weight")
    ap.add_argument("--stems", nargs='+', default=None,
                    help="One or more filename stems (e.g. Coral_reefs2007 AnotherDataset ...).")
    ap.add_argument("--stem", type=str, default=None,
                    help="Single filename stem (deprecated; use --stems). If omitted, inferred from latest *_P.mat in Results_node.")

    # theory flags / params
    ap.add_argument("--theory", action="store_true", help="Overlay theoretical curve.")
    ap.add_argument("--B", type=float, default=0.1)
    ap.add_argument("--K", type=float, default=5.0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=5.0)
    ap.add_argument("--E", type=float, default=0.9)
    ap.add_argument("--H", type=float, default=0.1)

    args = ap.parse_args()

    # Resolve list of stems
    if args.stems:
        stems = args.stems
    elif args.stem:
        stems = [args.stem]
    else:
        stems = [infer_stem_if_missing(args.base_dir, None)]
    print("[INFO] Stems:", stems)

    # Load datasets
    outputs_by_stem: Dict[str, List[Optional[np.ndarray]]] = {}
    for stem in stems:
        try:
            outputs_by_stem[stem] = list(load_triplet_for_stem(args.base_dir, stem))
        except Exception as e:
            print(f"[ERROR] Failed to load {stem}: {e}")
            outputs_by_stem[stem] = [None, None, None]

    # Panels for the first stem
    first = stems[0]
    first_node, first_link, first_weight = outputs_by_stem[first]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    if first_node is not None:   figure2_panel(first_node, simsize=6, simbal=1, ax=axs[0], title="Node removal")
    else:                        axs[0].set_title(f"{first}: Node removal (file not found)"); axs[0].axis('off')
    if first_link is not None:   figure2_panel(first_link, simsize=6, simbal=2, ax=axs[1], title="Link removal")
    else:                        axs[1].set_title(f"{first}: Link removal (file not found)"); axs[1].axis('off')
    if first_weight is not None: figure2_panel(first_weight, simsize=6, simbal=3, ax=axs[2], title="Weight weakening")
    else:                        axs[2].set_title(f"{first}: Weight weakening (file not found)"); axs[2].axis('off')
    fig.suptitle(f'{first}', fontsize=15)
    fig.tight_layout()
    plt.show()

    # Combined universal
    plot_universal_resilience_multi(
        outputs_by_stem,
        title="Universal resilience Patterns for Mutualistic Networks",
        add_theory=args.theory,
        B=args.B, K=args.K, C=args.C, D=args.D, E=args.E, H=args.H
    )

if __name__ == "__main__":
    main()
