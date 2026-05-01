"""
visualize_auxetics.py
=====================

Render auxetic lace ground patterns as bar-and-pin networks, with their
Poisson profiles and deformed configurations under uniaxial stretch.

For each ground:
  - Panel 1: bar network (3x3 or 4x4 tiling for symmetry visualization)
  - Panel 2: Poisson ratio polar plot, ν(θ)
  - Panel 3: rest configuration overlaid with deformed configuration under
             a unit strain in the direction of nu_min (showing the
             auxetic motion)

Usage:
    python3 visualize_auxetics.py --catalog tesselace_catalog \\
        --results mechanics_results.csv --classification homogeneously_auxetic \\
        --output figures/

    # Or pick specific grounds:
    python3 visualize_auxetics.py --grounds 3_6/2x4_86 3_6/R3M3_6x6_1

    # Or from a list file:
    python3 visualize_auxetics.py --ground-list top_auxetics.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .parse_to_graph import LaceGraph, parse_file
from .mechanics import (
    analyze, homogenize, assemble_stiffness, fix_translation,
    poisson_profile, default_lattice_vectors, vertex_position,
    edge_geometry, cell_vectors,
)


# -----------------------------------------------------------------------
# Bar network rendering
# -----------------------------------------------------------------------

def draw_lace_pattern(ax, graph: LaceGraph,
                      n_tiles_x: int = 3, n_tiles_y: int = 3,
                      L: np.ndarray = None,
                      vertex_color: str = '#222222',
                      edge_color: str = '#1f4e79',
                      cell_outline: str = '#888888',
                      vertex_size: float = 30,
                      edge_lw: float = 1.2):
    """Draw the bar-and-pin lace pattern, tiled n_tiles_x x n_tiles_y.

    Note on coordinates: TesseLace uses y-down (row increases downward).
    For visualization we flip y so the pattern reads natural with y-up.
    Axis limits are computed from actual rendered positions, so this
    works correctly under any (possibly rotated) lattice basis L.
    """
    if L is None:
        L = default_lattice_vectors()

    # Track all rendered points to compute the bounding box later
    all_x = []
    all_y = []

    # Draw vertices
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            for v in graph.vertices:
                p = L @ np.array([v[0] + tx * graph.n_cols,
                                  v[1] + ty * graph.n_rows])
                ax.scatter([p[0]], [-p[1]], s=vertex_size,
                           c=vertex_color, zorder=5, edgecolor='white',
                           linewidth=0.5)
                all_x.append(p[0]); all_y.append(-p[1])

    # Draw edges
    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                v_src = graph.vertices[e.src_idx]
                p_src = L @ np.array([v_src[0] + tx * graph.n_cols,
                                      v_src[1] + ty * graph.n_rows])
                dst_tx = tx + e.wrap[0]
                dst_ty = ty + e.wrap[1]
                v_dst = graph.vertices[e.dst_idx]
                p_dst = L @ np.array([v_dst[0] + dst_tx * graph.n_cols,
                                      v_dst[1] + dst_ty * graph.n_rows])
                ax.plot([p_src[0], p_dst[0]], [-p_src[1], -p_dst[1]],
                        '-', color=edge_color, lw=edge_lw, zorder=2,
                        solid_capstyle='round')
                all_x.extend([p_src[0], p_dst[0]])
                all_y.extend([-p_src[1], -p_dst[1]])

    # Cell outline at corner (0,0): draw the parallelogram defined by
    # the lattice vectors (which may be rotated).
    a1 = L @ np.array([graph.n_cols, 0.0])
    a2 = L @ np.array([0.0, graph.n_rows])
    corners = np.array([
        [0, 0],
        [a1[0], -a1[1]],
        [a1[0] + a2[0], -(a1[1] + a2[1])],
        [a2[0], -a2[1]],
        [0, 0],
    ])
    ax.plot(corners[:, 0], corners[:, 1], '--', color=cell_outline,
            lw=1.5, zorder=4)

    ax.set_aspect('equal')

    # Compute bbox from actual positions, not from naive grid extent
    if all_x and all_y:
        margin_x = 0.5 * (max(all_x) - min(all_x)) * 0.05 + 0.5
        margin_y = 0.5 * (max(all_y) - min(all_y)) * 0.05 + 0.5
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


# -----------------------------------------------------------------------
# Deformed configuration
# -----------------------------------------------------------------------

def deformed_positions(graph: LaceGraph, eps_voigt: np.ndarray,
                       k_per_unit_length: float = 1.0,
                       k_angular: float = 0.01,
                       L: np.ndarray = None
                       ) -> np.ndarray:
    """Compute deformed vertex positions under macroscopic strain eps_voigt
    (3-vector, [exx, eyy, 2*exy]). Returns an (N, 2) array of cartesian
    positions for the unit cell (no tiling). The macroscopic strain is
    applied as an affine deformation, plus the equilibrium periodic
    perturbation u^* solved from the homogenization.
    """
    if L is None:
        L = default_lattice_vectors()
    K, G, H, A = assemble_stiffness(graph, L, k_per_unit_length, k_angular)
    K_red, G_red, kept = fix_translation(K, G)

    # Solve for periodic perturbation: K u = -G eps
    rhs = -G_red @ eps_voigt
    try:
        u_red = np.linalg.solve(K_red, rhs)
    except np.linalg.LinAlgError:
        u_red = np.linalg.pinv(K_red) @ rhs

    # Reconstruct full u with vertex 0 pinned at zero
    N = len(graph.vertices)
    u_full = np.zeros(2 * N)
    u_full[2:] = u_red
    u_full = u_full.reshape((N, 2))

    # Affine displacement at each vertex: epsilon . p
    # epsilon_2x2 = [[exx, exy], [exy, eyy]]
    exx, eyy, gxy = eps_voigt
    exy = gxy / 2.0
    epsilon_mat = np.array([[exx, exy], [exy, eyy]])

    deformed = []
    for i, v in enumerate(graph.vertices):
        p_rest = L @ np.array([v[0], v[1]], dtype=float)
        affine = epsilon_mat @ p_rest
        new_p = p_rest + affine + u_full[i]
        deformed.append(new_p)
    return np.array(deformed)


def draw_deformed_pattern(ax, graph: LaceGraph, eps_voigt: np.ndarray,
                          k_angular: float = 0.01,
                          n_tiles_x: int = 3, n_tiles_y: int = 3,
                          L: np.ndarray = None):
    """Overlay the rest configuration (light) with the deformed
    configuration (dark) for the given macro strain. Used to visualize
    the auxetic mechanism.
    """
    if L is None:
        L = default_lattice_vectors()

    # Get deformed in-cell positions, then tile with the deformed
    # lattice vectors
    deformed_uc = deformed_positions(graph, eps_voigt, k_angular=k_angular, L=L)
    exx, eyy, gxy = eps_voigt
    exy = gxy / 2.0
    eps_mat = np.array([[exx, exy], [exy, eyy]])

    # Deformed lattice vectors: (I + eps) @ original
    F = np.eye(2) + eps_mat
    a1 = L @ np.array([graph.n_cols, 0.0])
    a2 = L @ np.array([0.0, graph.n_rows])
    a1_d = F @ a1
    a2_d = F @ a2

    all_x = []
    all_y = []

    # ----- Rest configuration (light gray) -----
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            for v in graph.vertices:
                p = L @ np.array([v[0] + tx * graph.n_cols,
                                  v[1] + ty * graph.n_rows])
                ax.scatter([p[0]], [-p[1]], s=20, c='#bbbbbb', zorder=3)
                all_x.append(p[0]); all_y.append(-p[1])
    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                v_src = graph.vertices[e.src_idx]
                v_dst = graph.vertices[e.dst_idx]
                p_src = L @ np.array([v_src[0] + tx * graph.n_cols,
                                      v_src[1] + ty * graph.n_rows])
                p_dst = L @ np.array([v_dst[0] + (tx + e.wrap[0]) * graph.n_cols,
                                      v_dst[1] + (ty + e.wrap[1]) * graph.n_rows])
                ax.plot([p_src[0], p_dst[0]], [-p_src[1], -p_dst[1]],
                        '-', color='#dddddd', lw=1.0, zorder=1)

    # ----- Deformed configuration (dark red) -----
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_offset = tx * a1_d + ty * a2_d
            for i, v in enumerate(graph.vertices):
                p = deformed_uc[i] + tile_offset
                ax.scatter([p[0]], [-p[1]], s=22, c='#c63300', zorder=5,
                           edgecolor='white', linewidth=0.4)
                all_x.append(p[0]); all_y.append(-p[1])
    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_offset_src = tx * a1_d + ty * a2_d
                tile_offset_dst = (tx + e.wrap[0]) * a1_d + (ty + e.wrap[1]) * a2_d
                p_src = deformed_uc[e.src_idx] + tile_offset_src
                p_dst = deformed_uc[e.dst_idx] + tile_offset_dst
                ax.plot([p_src[0], p_dst[0]], [-p_src[1], -p_dst[1]],
                        '-', color='#c63300', lw=1.2, zorder=4,
                        solid_capstyle='round')

    ax.set_aspect('equal')
    if all_x and all_y:
        margin_x = 0.5 * (max(all_x) - min(all_x)) * 0.05 + 0.5
        margin_y = 0.5 * (max(all_y) - min(all_y)) * 0.05 + 0.5
        ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
        ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


# -----------------------------------------------------------------------
# Poisson polar plot
# -----------------------------------------------------------------------

def draw_poisson_polar(ax, graph: LaceGraph,
                       k_angular: float = 0.01,
                       max_clip: float = 3.0):
    """Polar plot of Poisson ratio over angles, with auxetic
    (negative) regions filled in red and non-auxetic in blue."""
    h = homogenize(graph, k_angular=k_angular)
    C = h["C"]
    thetas, nus = poisson_profile(C, n_samples=360)

    # Replicate over [0, 2*pi) for polar plot (period is pi)
    thetas_full = np.concatenate([thetas, thetas + np.pi])
    nus_full = np.concatenate([nus, nus])

    # Use |nu| for radial axis, color by sign
    nus_clipped = np.clip(nus_full, -max_clip, max_clip)
    radii = np.abs(nus_clipped)

    # Plot: split into auxetic (negative) and non-auxetic
    mask_neg = nus_full < 0
    mask_pos = ~mask_neg

    # Fill regions
    ax.fill_between(thetas_full, 0, radii, where=mask_neg,
                     color='#cc1100', alpha=0.5, label='ν < 0 (auxetic)')
    ax.fill_between(thetas_full, 0, radii, where=mask_pos,
                     color='#2266aa', alpha=0.4, label='ν > 0')
    ax.plot(thetas_full, radii, '-', color='#222222', lw=0.5)

    ax.set_ylim(0, max_clip)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1', '2', '3'], fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, alpha=0.3)
    return float(np.nanmin(nus)), float(np.nanmax(nus))


# -----------------------------------------------------------------------
# Top-level: render one row (3 panels) per ground
# -----------------------------------------------------------------------

def render_ground_row(graph: LaceGraph, fig, gs_row,
                      k_angular: float = 0.01,
                      n_tiles: int = 3,
                      strain_magnitude: float = 0.03) -> Dict:
    """Render one ground as a row of three panels.

    Panel 1: rest pattern, tiled
    Panel 2: Poisson polar plot
    Panel 3: deformed configuration under uniaxial strain in the direction
             of maximum auxetic response. The frame is rotated so that
             the strain axis is horizontal, making the auxetic *vertical
             expansion* visible at a glance.

    Returns analysis info for the title.
    """
    # Run analysis to find the strain direction of nu_min
    h = homogenize(graph, k_angular=k_angular)
    C = h["C"]
    thetas, nus = poisson_profile(C, n_samples=360)
    if np.all(np.isnan(nus)):
        theta_aux = 0.0
    else:
        theta_aux = float(thetas[np.nanargmin(nus)])

    # We will rotate the entire visualization so the auxetic strain axis
    # is horizontal. Conceptually: instead of applying strain at angle
    # theta_aux in the original frame, we apply strain along x in a
    # rotated frame (multiply lattice vectors by R^T).
    c, s = np.cos(theta_aux), np.sin(theta_aux)
    R = np.array([[c, s], [-s, c]])  # rotates by -theta_aux (so axis lines up with x)
    L_rot = R @ default_lattice_vectors()

    # Three panels in this row
    ax1 = fig.add_subplot(gs_row[0])
    ax2 = fig.add_subplot(gs_row[1], projection='polar')
    ax3 = fig.add_subplot(gs_row[2])

    # Panel 1: rest pattern in NATURAL (unrotated) frame, so the lace
    # always reads in the orientation a lace-maker would recognize
    draw_lace_pattern(ax1, graph, n_tiles_x=n_tiles, n_tiles_y=n_tiles,
                      L=default_lattice_vectors())
    deg = np.degrees(theta_aux) % 180
    ax1.set_title(f'{graph.family}/{graph.name}\n'
                  f'{len(graph.vertices)}v, {len(graph.edges)}e, '
                  f'{graph.n_rows}×{graph.n_cols} cell',
                  fontsize=9)

    # Panel 2: Poisson polar (always in original frame; this panel is
    # invariant to the rendering rotation)
    nu_min, nu_max = draw_poisson_polar(ax2, graph, k_angular=k_angular)
    ax2.set_title(f'ν profile (original frame)\n'
                  f'ν$_{{min}}$={nu_min:+.3f} at θ={deg:.0f}°, '
                  f'ν$_{{max}}$={nu_max:+.3f}',
                  fontsize=9, pad=12)

    # Panel 3: deformed in ROTATED frame, so strain axis is horizontal
    # and the auxetic vertical-expansion is visible at a glance
    eps_voigt_rot = np.array([strain_magnitude, 0.0, 0.0])
    draw_deformed_pattern(ax3, graph, eps_voigt_rot,
                          k_angular=k_angular,
                          n_tiles_x=n_tiles, n_tiles_y=n_tiles,
                          L=L_rot)
    direction_label = "→" if strain_magnitude > 0 else "←"
    ax3.set_title(f'stretch {direction_label} {strain_magnitude:.0%} along most-auxetic axis\n'
                  f'(rest=gray, deformed=red); '
                  f'expected ν={nu_min:+.2f}',
                  fontsize=9)

    return {"nu_min": nu_min, "nu_max": nu_max, "theta_aux_deg": deg}


def render_grounds(grounds: List[LaceGraph], output_path: str,
                   k_angular: float = 0.01, n_tiles: int = 2,
                   strain_magnitude: float = 0.03,
                   title: str = "Top auxetic lace grounds") -> None:
    """Render a single multi-row figure for a list of grounds."""
    n = len(grounds)
    fig = plt.figure(figsize=(11.5, 3.0 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[1.2, 0.9, 1.2],
                           hspace=0.55, wspace=0.25,
                           left=0.04, right=0.98, top=0.96, bottom=0.02)
    summary_rows = []
    for i, g in enumerate(grounds):
        gs_row = [gs[i, 0], gs[i, 1], gs[i, 2]]
        try:
            info = render_ground_row(g, fig, gs_row, k_angular=k_angular,
                              n_tiles=n_tiles, strain_magnitude=strain_magnitude)
            summary_rows.append({
                "family": g.family, "name": g.name,
                "n_vertices": len(g.vertices), "n_edges": len(g.edges),
                "n_rows": g.n_rows, "n_cols": g.n_cols,
                "nu_min": info["nu_min"], "nu_max": info["nu_max"],
                "theta_aux_deg": info["theta_aux_deg"],
            })
        except Exception as exc:
            print(f"  WARNING: failed to render {g.family}/{g.name}: {exc}",
                  file=sys.stderr)
    fig.suptitle(title, fontsize=13, y=1.0)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")

    # Also dump a small CSV summary alongside the figure
    if summary_rows:
        csv_path = os.path.splitext(output_path)[0] + "_summary.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"Saved summary: {csv_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def select_grounds(catalog_dir: str, results_csv: str,
                   classification: Optional[str] = None,
                   explicit_paths: Optional[List[str]] = None,
                   max_n: Optional[int] = None) -> List[LaceGraph]:
    """Pick grounds by classification (from results CSV) or explicit paths.
    Returns a list of LaceGraph objects."""
    selected_paths: List[Tuple[str, str, str]] = []  # (family, name, path)

    if explicit_paths:
        for p in explicit_paths:
            # Accept "family/name" or full path
            if p.endswith(".txt"):
                full = p
            else:
                full = os.path.join(catalog_dir, "tl", p + ".txt")
            family = os.path.basename(os.path.dirname(full))
            name = os.path.splitext(os.path.basename(full))[0]
            selected_paths.append((family, name, full))
    elif classification:
        with open(results_csv) as f:
            for row in csv.DictReader(f):
                if row["classification"] == classification:
                    family = row["family"]
                    name = row["name"]
                    path = os.path.join(catalog_dir, "tl",
                                         family, name + ".txt")
                    selected_paths.append((family, name, path))
    else:
        raise ValueError("must specify --classification or --grounds")

    if max_n is not None:
        selected_paths = selected_paths[:max_n]

    grounds = []
    for family, name, path in selected_paths:
        if not os.path.exists(path):
            print(f"  warning: not found: {path}", file=sys.stderr)
            continue
        g = parse_file(path, name=name, family=family)
        if g is not None:
            grounds.append(g)
    return grounds


def main():
    ap = argparse.ArgumentParser(
        description="Visualize auxetic lace ground patterns with bar models, "
                    "Poisson polar plots, and deformed configurations.")
    ap.add_argument("--catalog", default="tesselace_catalog",
                    help="catalog root directory (containing tl/ subdir)")
    ap.add_argument("--results", default="mechanics_results.csv",
                    help="mechanics CSV with classification labels")
    ap.add_argument("--classification", default="homogeneously_auxetic",
                    help="select all grounds with this classification "
                         "(default: homogeneously_auxetic)")
    ap.add_argument("--grounds", nargs="+", default=None,
                    help="explicit list, e.g. 3_6/2x4_86 3_5/2x3_10")
    ap.add_argument("--output", "-o", default="auxetic_visualization.pdf",
                    help="output file (.pdf or .png)")
    ap.add_argument("--k-angular", type=float, default=0.01)
    ap.add_argument("--tiles", type=int, default=2,
                    help="number of unit cells per side in the visualization")
    ap.add_argument("--strain", type=float, default=0.03,
                    help="strain magnitude for deformed view (default 3%)")
    ap.add_argument("--max", type=int, default=None,
                    help="cap on number of grounds rendered")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    grounds = select_grounds(args.catalog, args.results,
                              classification=(args.classification
                                              if not args.grounds else None),
                              explicit_paths=args.grounds,
                              max_n=args.max)
    print(f"Selected {len(grounds)} grounds.")
    if not grounds:
        print("Nothing to render.")
        return 1

    title = args.title or (
        f"Top auxetic lace grounds ({args.classification or 'custom'}): "
        f"bar networks, Poisson profiles, and deformed configurations"
    )
    render_grounds(grounds, args.output,
                    k_angular=args.k_angular,
                    n_tiles=args.tiles,
                    strain_magnitude=args.strain,
                    title=title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
