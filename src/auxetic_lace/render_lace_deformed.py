"""
render_lace_deformed.py
=======================

Render a TesseLace ground as a stretched lace swatch, showing the auxetic
deformation. Designed to be visually consistent with render_lace_view.py
(same pair-diagram aesthetic) but adds a deformed configuration overlay.

The panel shows:
  - Rest swatch (light gray, faded) at original size
  - Deformed swatch (colored, on top) under uniaxial stretch along the
    most-auxetic direction
  - Cell parallelogram outline in both states (red rest, blue deformed)
  - Arrows indicating the applied stretch direction

For homogeneously auxetic grounds, the deformed swatch should clearly
expand BOTH along the stretch direction AND perpendicular to it (the
defining auxetic property). Visible strain magnitudes (default 12%)
are used so the deformation reads at a glance — the mechanics is
linear elasticity, so this is a visualization choice not a physics
claim.

Usage:
    python3 render_lace_deformed.py --ground 3_6/2x4_86 \\
        --output figures/2x4_86_deformed.png

    python3 render_lace_deformed.py --catalog tesselace_catalog \\
        --classification homogeneously_auxetic \\
        --output-dir figures/deformed_lace_views/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parse_to_graph import LaceGraph, parse_file
from .mechanics import (
    homogenize, default_lattice_vectors, poisson_profile,
)
from .visualize_auxetics import deformed_positions


# -----------------------------------------------------------------------
# Compute deformed swatch positions and edges
# -----------------------------------------------------------------------

def _compute_swatch(graph: LaceGraph,
                    n_tiles_x: int, n_tiles_y: int,
                    L: np.ndarray,
                    deformed_uc: Optional[np.ndarray] = None,
                    F: Optional[np.ndarray] = None):
    """Compute (vertex_positions, edge_segments, cell_corners) for a
    swatch.

    If deformed_uc and F are None, returns the rest swatch in lattice
    L. If both are provided, returns the deformed swatch:
      - in-cell vertices placed at deformed_uc[i] (the equilibrium
        positions under the macro strain),
      - tile offsets in the deformed lattice F @ a1, F @ a2.

    All returned coordinates have y already flipped to screen-up.
    """
    deformed = deformed_uc is not None
    if deformed:
        a1 = L @ np.array([graph.n_cols, 0.0])
        a2 = L @ np.array([0.0, graph.n_rows])
        a1_t = F @ a1
        a2_t = F @ a2
    else:
        a1_t = L @ np.array([graph.n_cols, 0.0])
        a2_t = L @ np.array([0.0, graph.n_rows])

    positions = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_offset = tx * a1_t + ty * a2_t
            for i, v in enumerate(graph.vertices):
                if deformed:
                    p = deformed_uc[i] + tile_offset
                else:
                    p = L @ np.array([v[0], v[1]], dtype=float) + tile_offset
                positions.append((p[0], -p[1]))

    edge_segments = []
    V = len(graph.vertices)
    for e in graph.edges:
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                src_offset = tx * a1_t + ty * a2_t
                dst_offset = (tx + e.wrap[0]) * a1_t + (ty + e.wrap[1]) * a2_t
                if deformed:
                    p_src = deformed_uc[e.src_idx] + src_offset
                    p_dst = deformed_uc[e.dst_idx] + dst_offset
                else:
                    v_src = graph.vertices[e.src_idx]
                    v_dst = graph.vertices[e.dst_idx]
                    p_src = (L @ np.array([v_src[0], v_src[1]], dtype=float)
                             + src_offset)
                    p_dst = (L @ np.array([v_dst[0], v_dst[1]], dtype=float)
                             + dst_offset)
                edge_segments.append(((p_src[0], -p_src[1]),
                                       (p_dst[0], -p_dst[1])))

    # Cell corners (the (0,0) tile)
    cell_corners = np.array([
        [0, 0],
        [a1_t[0], -a1_t[1]],
        [a1_t[0] + a2_t[0], -(a1_t[1] + a2_t[1])],
        [a2_t[0], -a2_t[1]],
        [0, 0],
    ])
    return positions, edge_segments, cell_corners


# -----------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------

def render_deformed_lace_swatch(ax, graph: LaceGraph,
                                 strain_magnitude: float = 0.12,
                                 n_tiles_x: int = 3, n_tiles_y: int = 3,
                                 k_angular: float = 0.01,
                                 rest_color: str = '#7a7a7a',
                                 rest_pin_color: str = '#bbbbbb',
                                 deformed_color: str = '#7a2418',
                                 deformed_pin_color: str = 'white',
                                 bg_color: str = '#f5f0e3',
                                 rest_lw: float = 3.5,
                                 deformed_lw: float = 4.0,
                                 pin_size: float = 50,
                                 show_arrow: bool = True,
                                 ) -> dict:
    """Render a lace swatch deformed under uniaxial stretch along the
    most-auxetic axis. Rest configuration is drawn faded, deformed
    configuration is drawn on top.

    Returns analysis info: nu_min, nu_max, theta_aux_deg, applied_strain.
    """
    ax.set_facecolor(bg_color)

    # Identify the most-auxetic direction
    h = homogenize(graph, k_angular=k_angular)
    C = h["C"]
    thetas, nus = poisson_profile(C, n_samples=360)
    valid = ~np.isnan(nus)
    if not np.any(valid):
        theta_aux = 0.0
        nu_min = nu_max = float('nan')
    else:
        idx = np.nanargmin(nus)
        theta_aux = float(thetas[idx])
        nu_min = float(np.nanmin(nus))
        nu_max = float(np.nanmax(nus))

    # Construct the strain tensor that produces uniaxial stress along the
    # most-auxetic axis. Working in the ORIGINAL frame (we will rotate
    # the visualization at draw time, but mechanics is in the original
    # frame, otherwise we'd be re-defining the framework's geometry).
    #
    # In a frame aligned with the most-auxetic axis (call it x'), under
    # uniaxial stress along x', the strain is
    #   eps' = [eps_x', eps_y', gamma_xy'] = S_rot @ [sigma, 0, 0]
    # where S_rot is the compliance rotated by theta_aux.
    # Then the strain tensor in the rotated frame is
    #   eps_mat' = [[eps_x', eps_xy'], [eps_xy', eps_y']]
    # and we rotate it back to the lab frame:
    #   eps_mat = R eps_mat' R^T
    S = np.linalg.inv(C)

    # Rotate compliance by theta_aux to get response under stress along
    # the auxetic axis. Use the existing rotate_voigt helper.
    from .mechanics import rotate_voigt
    S_rot = rotate_voigt(S, theta_aux)
    # Pick a stress magnitude that produces strain_magnitude along x' axis:
    # eps_x' = S_rot[0,0] * sigma  =>  sigma = strain_magnitude / S_rot[0,0]
    if abs(S_rot[0, 0]) < 1e-12:
        sigma_mag = 0.0
    else:
        sigma_mag = strain_magnitude / S_rot[0, 0]
    eps_prime = S_rot @ np.array([sigma_mag, 0.0, 0.0])
    eps_x_p, eps_y_p, gxy_p = eps_prime  # in rotated (auxetic-aligned) frame

    # Convert rotated-frame Voigt strain to a 2x2 strain tensor
    eps_mat_prime = np.array([[eps_x_p,    gxy_p / 2.0],
                              [gxy_p / 2.0, eps_y_p]])
    # Rotate back to lab frame: eps_lab = R @ eps_prime @ R^T
    c, s = np.cos(theta_aux), np.sin(theta_aux)
    R_lab = np.array([[c, -s], [s, c]])  # rotates by +theta_aux
    eps_mat = R_lab @ eps_mat_prime @ R_lab.T
    eps_voigt = np.array([eps_mat[0, 0], eps_mat[1, 1], 2 * eps_mat[0, 1]])

    # Use lattice in lab frame for mechanics
    L_lab = default_lattice_vectors()
    deformed_uc = deformed_positions(graph, eps_voigt,
                                      k_angular=k_angular, L=L_lab)
    F = np.eye(2) + eps_mat

    # For VISUALIZATION ONLY, rotate everything by -theta_aux so the
    # auxetic axis appears horizontal in the figure.
    R_view = np.array([[c, s], [-s, c]])  # rotates by -theta_aux
    L_rot = R_view @ L_lab
    F_view = R_view @ F @ R_view.T
    # Apply the same rotation to the deformed in-cell positions
    deformed_uc_view = deformed_uc @ R_view.T  # row-vectors

    # Compute rest and deformed swatches (both in rotated view so the
    # auxetic axis is horizontal in the figure)
    rest_pos, rest_edges, rest_cell = _compute_swatch(
        graph, n_tiles_x, n_tiles_y, L_rot)
    def_pos, def_edges, def_cell = _compute_swatch(
        graph, n_tiles_x, n_tiles_y, L_rot,
        deformed_uc=deformed_uc_view, F=F_view)

    # ---- Draw rest swatch (faded, underneath) ----
    for (p1, p2) in rest_edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                '-', color=rest_color, lw=rest_lw, zorder=2,
                solid_capstyle='round', alpha=0.85)
    rest_xs = [p[0] for p in rest_pos]
    rest_ys = [p[1] for p in rest_pos]
    ax.scatter(rest_xs, rest_ys, s=pin_size * 0.7, c=rest_pin_color,
               edgecolor='#555', linewidth=0.6, zorder=3, alpha=0.9)

    # Compute the rest swatch's centroid for re-anchoring the deformed
    # swatch: this aligns them centroid-to-centroid so the auxetic
    # *expansion* is the visual story, not an overall translation.
    rest_cx0 = (min(rest_xs) + max(rest_xs)) / 2
    rest_cy0 = (min(rest_ys) + max(rest_ys)) / 2
    def_cx0 = sum(p[0] for p in def_pos) / len(def_pos)
    def_cy0 = sum(p[1] for p in def_pos) / len(def_pos)
    # Use the bounding-box centroid for both, for symmetry
    def_cx0 = (min(p[0] for p in def_pos) + max(p[0] for p in def_pos)) / 2
    def_cy0 = (min(p[1] for p in def_pos) + max(p[1] for p in def_pos)) / 2
    shift_x = rest_cx0 - def_cx0
    shift_y = rest_cy0 - def_cy0

    # ---- Draw deformed swatch (on top, re-centered) ----
    for (p1, p2) in def_edges:
        ax.plot([p1[0] + shift_x, p2[0] + shift_x],
                [p1[1] + shift_y, p2[1] + shift_y],
                '-', color=deformed_color, lw=deformed_lw, zorder=5,
                solid_capstyle='round')
    def_xs = [p[0] + shift_x for p in def_pos]
    def_ys = [p[1] + shift_y for p in def_pos]
    ax.scatter(def_xs, def_ys, s=pin_size, c=deformed_pin_color,
               edgecolor=deformed_color, linewidth=1.2, zorder=7)

    # ---- Cell parallelograms and envelope bounding boxes are not drawn
    # to keep the figure clean. We still compute envelope strains for
    # the return dict, since they're useful in titles. ----
    rest_xmin, rest_xmax = min(rest_xs), max(rest_xs)
    rest_ymin, rest_ymax = min(rest_ys), max(rest_ys)
    def_xmin, def_xmax = min(def_xs), max(def_xs)
    def_ymin, def_ymax = min(def_ys), max(def_ys)

    rest_w = rest_xmax - rest_xmin
    rest_h = rest_ymax - rest_ymin
    def_w = def_xmax - def_xmin
    def_h = def_ymax - def_ymin

    # Envelope expansion ratios → apparent Poisson from swatch boundary
    swatch_strain_x = (def_w - rest_w) / rest_w if rest_w > 0 else 0.0
    swatch_strain_y = (def_h - rest_h) / rest_h if rest_h > 0 else 0.0
    if abs(swatch_strain_x) > 1e-9:
        nu_apparent = -swatch_strain_y / swatch_strain_x
    else:
        nu_apparent = float('nan')

    # ---- Strain direction arrows ----
    all_x = rest_xs + def_xs
    all_y = rest_ys + def_ys
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    width = xmax - xmin
    height = ymax - ymin

    if show_arrow:
        arrow_color = rest_color
        arrow_y = ymax + 0.08 * height
        arrow_len = 0.22 * width
        # Arrows point OUTWARD from the specimen (engineering tension
        # convention: external pulls applied to the rest body). Drawn in
        # the rest color so they read as "applied to the unstretched
        # specimen" rather than as part of the deformed result.
        ax.annotate('', xy=(xmin - arrow_len, arrow_y),
                    xytext=(xmin, arrow_y),
                    arrowprops=dict(arrowstyle='-|>', color=arrow_color,
                                     lw=2.5, mutation_scale=22),
                    zorder=10)
        ax.annotate('', xy=(xmax + arrow_len, arrow_y),
                    xytext=(xmax, arrow_y),
                    arrowprops=dict(arrowstyle='-|>', color=arrow_color,
                                     lw=2.5, mutation_scale=22),
                    zorder=10)
        ax.text((xmin + xmax) / 2, arrow_y + 0.04 * height,
                f'tension ε = {strain_magnitude:.0%}',
                ha='center', va='bottom', fontsize=9,
                color=arrow_color, fontweight='bold')

    ax.set_aspect('equal')
    margin = 0.08 * max(width, height) + 0.5
    if show_arrow:
        # Extra horizontal margin to fit the outward-pointing arrows
        ax.set_xlim(xmin - 0.30 * width - margin,
                    xmax + 0.30 * width + margin)
        ax.set_ylim(ymin - margin, ymax + margin + 0.18 * height)
    else:
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    return {
        "nu_min": nu_min,
        "nu_max": nu_max,
        "theta_aux_deg": np.degrees(theta_aux) % 180,
        "strain": strain_magnitude,
        "swatch_strain_x": swatch_strain_x,
        "swatch_strain_y": swatch_strain_y,
        "nu_apparent_swatch": nu_apparent,
    }


# -----------------------------------------------------------------------
# Public entry point: render a single ground
# -----------------------------------------------------------------------

def render_ground_deformed(graph: LaceGraph,
                            output_path: str,
                            strain_magnitude: float = 0.12,
                            n_tiles: int = 3,
                            k_angular: float = 0.01) -> dict:
    """Render a single-panel figure showing the deformed lace swatch."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    info = render_deformed_lace_swatch(
        ax, graph,
        strain_magnitude=strain_magnitude,
        n_tiles_x=n_tiles, n_tiles_y=n_tiles,
        k_angular=k_angular)

    fig.suptitle(f'{graph.family}/{graph.name} stretched along most-auxetic axis\n'
                 f'(rest = faded gray, deformed = dark red);  '
                 f'ν$_{{min}}$ = {info["nu_min"]:+.3f} at θ = '
                 f'{info["theta_aux_deg"]:.0f}°, '
                 f'ν$_{{max}}$ = {info["nu_max"]:+.3f}\n'
                 f'swatch envelope: width Δ = {info["swatch_strain_x"]:+.1%}, '
                 f'height Δ = {info["swatch_strain_y"]:+.1%}, '
                 f'apparent ν = {info["nu_apparent_swatch"]:+.2f}',
                 fontsize=10, y=0.98)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#f5f0e3')
    plt.close(fig)
    print(f"Saved {output_path}")
    return info


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_ground_arg(arg: str, catalog_dir: str) -> Tuple[str, str, str]:
    """Parse a ground specifier of the form 'family/name' or full file
    path into (family, name, full_path)."""
    if arg.endswith(".txt"):
        full = arg
        family = os.path.basename(os.path.dirname(full))
        name = os.path.splitext(os.path.basename(full))[0]
    else:
        family, name = arg.split("/", 1)
        full = os.path.join(catalog_dir, "tl", family, name + ".txt")
    return family, name, full


def main():
    ap = argparse.ArgumentParser(
        description="Render a TesseLace ground as a deformed lace swatch "
                    "under tension along the most-auxetic axis. "
                    "Visualizes the auxetic effect at a glance.")
    ap.add_argument("--catalog", default="tesselace_catalog",
                    help="catalog root (containing tl/ subdir)")
    ap.add_argument("--ground", default=None,
                    help="single ground to render, e.g. 3_6/2x4_86")
    ap.add_argument("--results", default="mechanics_results.csv",
                    help="results CSV (used with --classification)")
    ap.add_argument("--classification", default=None,
                    help="render every ground with this classification")
    ap.add_argument("--output", "-o", default="lace_deformed.png",
                    help="output file path (single-ground mode)")
    ap.add_argument("--output-dir", default=None,
                    help="output directory (multi-ground mode)")
    ap.add_argument("--tiles", type=int, default=3,
                    help="number of unit cells per side (default 3)")
    ap.add_argument("--strain", type=float, default=0.12,
                    help="applied strain magnitude (default 12%)")
    ap.add_argument("--k-angular", type=float, default=0.01)
    args = ap.parse_args()

    if args.ground:
        family, name, path = _parse_ground_arg(args.ground, args.catalog)
        if not os.path.exists(path):
            print(f"ground not found: {path}", file=sys.stderr)
            return 1
        g = parse_file(path, name=name, family=family)
        if g is None:
            print(f"failed to parse: {path}", file=sys.stderr)
            return 1
        render_ground_deformed(g, args.output,
                                strain_magnitude=args.strain,
                                n_tiles=args.tiles,
                                k_angular=args.k_angular)
        return 0

    if args.classification:
        if not args.output_dir:
            print("--output-dir required with --classification",
                  file=sys.stderr)
            return 1
        n = 0
        with open(args.results) as f:
            for row in csv.DictReader(f):
                if row["classification"] != args.classification:
                    continue
                family = row["family"]
                name = row["name"]
                path = os.path.join(args.catalog, "tl", family,
                                     name + ".txt")
                if not os.path.exists(path):
                    print(f"  skip (not found): {path}", file=sys.stderr)
                    continue
                g = parse_file(path, name=name, family=family)
                if g is None:
                    continue
                out_path = os.path.join(args.output_dir,
                                         f"{family}_{name}_deformed.png")
                render_ground_deformed(g, out_path,
                                        strain_magnitude=args.strain,
                                        n_tiles=args.tiles,
                                        k_angular=args.k_angular)
                n += 1
        print(f"Rendered {n} grounds to {args.output_dir}")
        return 0

    print("specify either --ground or --classification", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
