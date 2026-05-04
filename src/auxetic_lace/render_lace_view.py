"""
render_lace_view.py
===================

Render a TesseLace ground as a "pair diagram" - the standard lace-design
intermediate representation between the abstract bar-and-pin graph and a
finished lace fabric.

In a pair diagram:
  - Each edge of the lace graph = a single thick band representing one
    pair of threads
  - Each vertex of the lace graph = a small open circle representing a
    pin (the point where one pair crosses with another)
  - Tiled at modest tile count (default 5x5) to give a "swatch" look

The pair diagram is what you'd hand a lacemaker as a working diagram. It
captures the lace's structural skeleton honestly without fudging the
per-vertex action sequence ζ(v) that determines actual stitch texture.

This view complements visualize_auxetics.py:
  - visualize_auxetics.py = mechanics-focused (bar lattice, Poisson polar,
    deformed configuration). Useful for the paper's mechanics figures.
  - render_lace_view.py   = lace-focused (pair diagram). Useful for
    showing readers what the structure actually looks like as a textile.

Usage:
    python3 render_lace_view.py --ground 3_6/2x4_86 --output lace_view.png
    python3 render_lace_view.py --ground 3_6/R3M3_6x6_1 --tiles 6 \\
        --output figures/R3M3_6x6_1_lace.png
    python3 render_lace_view.py --catalog tesselace_catalog \\
        --classification homogeneously_auxetic \\
        --output-dir figures/lace_views/
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
from .mechanics import default_lattice_vectors


def _tile_positions_and_edges(graph: LaceGraph,
                              n_tiles_x: int, n_tiles_y: int,
                              L: np.ndarray):
    """Compute all rendered positions and edges for an n_tiles_x x n_tiles_y
    tiled patch. Returns (positions, edge_segments) where positions is a
    list of (x, y) and edge_segments is a list of ((x1,y1), (x2,y2)).
    Note: y is already flipped to screen-up convention.
    """
    positions = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            for v in graph.vertices:
                p = L @ np.array([v[0] + tx * graph.n_cols,
                                  v[1] + ty * graph.n_rows])
                positions.append((p[0], -p[1]))

    edge_segments = []
    for e in graph.edges:
        v_src = graph.vertices[e.src_idx]
        v_dst = graph.vertices[e.dst_idx]
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                p_src = L @ np.array([v_src[0] + tx * graph.n_cols,
                                      v_src[1] + ty * graph.n_rows])
                p_dst = L @ np.array([
                    v_dst[0] + (tx + e.wrap[0]) * graph.n_cols,
                    v_dst[1] + (ty + e.wrap[1]) * graph.n_rows])
                edge_segments.append(((p_src[0], -p_src[1]),
                                       (p_dst[0], -p_dst[1])))
    return positions, edge_segments


def render_pair_diagram(ax, graph: LaceGraph,
                        n_tiles_x: int = 5, n_tiles_y: int = 5,
                        L: np.ndarray = None,
                        pair_color: str = '#3a3a3a',
                        pair_lw: float = 4.5,
                        pin_color: str = 'white',
                        pin_edge_color: str = '#222',
                        pin_size: float = 70,
                        bg_color: str = '#f5f0e3',
                        show_cell_outline: bool = True,
                        cell_outline_color: str = '#aa3333',
                        cell_outline_lw: float = 1.5):
    """Render a pair diagram: each lace-graph edge as a thick band, each
    vertex as a small open circle (representing a pin). Background is
    given a paper-like cream color by default, evoking real lace
    documentation."""
    if L is None:
        L = default_lattice_vectors()

    ax.set_facecolor(bg_color)

    positions, edges = _tile_positions_and_edges(graph, n_tiles_x, n_tiles_y, L)

    # Draw pairs (edges) underneath
    for (p1, p2) in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                '-', color=pair_color, lw=pair_lw, zorder=2,
                solid_capstyle='round')

    # Draw pins (vertices) on top
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.scatter(xs, ys, s=pin_size, c=pin_color,
               edgecolor=pin_edge_color, linewidth=1.5, zorder=5)

    # Optional cell outline
    if show_cell_outline:
        a1 = L @ np.array([graph.n_cols, 0.0])
        a2 = L @ np.array([0.0, graph.n_rows])
        corners = np.array([
            [0, 0],
            [a1[0], -a1[1]],
            [a1[0] + a2[0], -(a1[1] + a2[1])],
            [a2[0], -a2[1]],
            [0, 0],
        ])
        ax.plot(corners[:, 0], corners[:, 1], '--',
                color=cell_outline_color, lw=cell_outline_lw, zorder=6)

    ax.set_aspect('equal')

    # Compute bbox from rendered positions
    if positions:
        xmin = min(xs); xmax = max(xs)
        ymin = min(ys); ymax = max(ys)
        margin_x = (xmax - xmin) * 0.04 + 0.5
        margin_y = (ymax - ymin) * 0.04 + 0.5
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def render_thread_diagram(ax, graph: LaceGraph,
                          n_tiles_x: int = 5, n_tiles_y: int = 5,
                          L: np.ndarray = None,
                          thread_color: str = '#1a1a1a',
                          thread_lw: float = 1.4,
                          thread_offset: float = 0.10,
                          pin_color: str = '#aa6633',
                          pin_size: float = 18,
                          bg_color: str = '#f5f0e3',
                          show_cell_outline: bool = True):
    """Render a *thread* diagram: each pair becomes two parallel threads
    offset perpendicular to the bar direction. This approximates how lace
    actually looks, though without committing to specific stitch types
    (cross-twist patterns) at each vertex.

    The threads run parallel along each bar; at each pin they appear to
    pass over/under but we draw both as solid for legibility.
    Note this is a simplified "default-stitch" rendering — real lace
    texture depends on the action sequence ζ(v) at each vertex which
    is NOT encoded in the TesseLace catalog format we're consuming.
    """
    if L is None:
        L = default_lattice_vectors()

    ax.set_facecolor(bg_color)

    positions, edges = _tile_positions_and_edges(graph, n_tiles_x, n_tiles_y, L)

    # For each edge, draw two parallel offset threads
    for (p1, p2) in edges:
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        n = np.linalg.norm(v)
        if n < 1e-9:
            continue
        # Perpendicular direction
        perp = np.array([-v[1], v[0]]) / n * thread_offset

        # Thread A
        ax.plot([p1[0] + perp[0], p2[0] + perp[0]],
                [p1[1] + perp[1], p2[1] + perp[1]],
                '-', color=thread_color, lw=thread_lw, zorder=2,
                solid_capstyle='round', alpha=0.85)
        # Thread B
        ax.plot([p1[0] - perp[0], p2[0] - perp[0]],
                [p1[1] - perp[1], p2[1] - perp[1]],
                '-', color=thread_color, lw=thread_lw, zorder=2,
                solid_capstyle='round', alpha=0.85)

    # Pins as small dots (the holes a lacemaker would pin into)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.scatter(xs, ys, s=pin_size, c=pin_color, zorder=5,
               edgecolor='white', linewidth=0.6)

    if show_cell_outline:
        a1 = L @ np.array([graph.n_cols, 0.0])
        a2 = L @ np.array([0.0, graph.n_rows])
        corners = np.array([
            [0, 0],
            [a1[0], -a1[1]],
            [a1[0] + a2[0], -(a1[1] + a2[1])],
            [a2[0], -a2[1]],
            [0, 0],
        ])
        ax.plot(corners[:, 0], corners[:, 1], '--',
                color='#aa3333', lw=1.0, zorder=6, alpha=0.7)

    ax.set_aspect('equal')
    if positions:
        xmin = min(xs); xmax = max(xs)
        ymin = min(ys); ymax = max(ys)
        margin_x = (xmax - xmin) * 0.04 + 0.5
        margin_y = (ymax - ymin) * 0.04 + 0.5
        ax.set_xlim(xmin - margin_x, xmax + margin_x)
        ax.set_ylim(ymin - margin_y, ymax + margin_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


# -----------------------------------------------------------------------
# Public entry point: render single ground as a multi-style figure
# -----------------------------------------------------------------------

def render_ground_lace_views(graph: LaceGraph,
                              output_path: str,
                              n_tiles: int = 5,
                              include_thread_view: bool = True) -> None:
    """Render a single ground as a 2-panel figure: pair diagram on the
    left, thread diagram on the right. The thread view approximates
    lace texture using a simple default stitch model (no ζ(v))."""
    if include_thread_view:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
        render_pair_diagram(axes[0], graph, n_tiles_x=n_tiles, n_tiles_y=n_tiles)
        axes[0].set_title('pair diagram\n'
                          '(each band = one thread pair, '
                          'each pin = stitch interaction)',
                          fontsize=10)
        render_thread_diagram(axes[1], graph, n_tiles_x=n_tiles, n_tiles_y=n_tiles)
        axes[1].set_title('thread sketch\n'
                          '(individual threads, default cross stitch)',
                          fontsize=10)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        render_pair_diagram(ax, graph, n_tiles_x=n_tiles, n_tiles_y=n_tiles)
        ax.set_title('pair diagram',
                     fontsize=10)

    fig.suptitle(f'{graph.family}/{graph.name}    '
                 f'{graph.n_rows}×{graph.n_cols} unit cell, '
                 f'{len(graph.vertices)} stitches, {len(graph.edges)} pair-segments',
                 fontsize=12, y=0.97)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f5f0e3')
    plt.close(fig)
    print(f"Saved {output_path}")




def render_ground_lace_views_split(graph: LaceGraph,
                                    thread_path: str,
                                    n_tiles: int = 5) -> None:
    """Render the chunky-bands view as thread.png.

    The woven over-under "lace" view was dropped from the project; the
    chunky-bands view is the canonical thumbnail per ground."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    render_pair_diagram(ax, graph, n_tiles_x=n_tiles, n_tiles_y=n_tiles)
    ax.set_title("")
    out_dir = os.path.dirname(thread_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(thread_path, dpi=150, bbox_inches='tight', pad_inches=0.1,
                facecolor='#f5f0e3')
    plt.close(fig)
    print(f"Saved {thread_path}")


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
        description="Render a TesseLace ground as a lace-style pair "
                    "diagram (and optional thread sketch).")
    ap.add_argument("--catalog", default="tesselace_catalog",
                    help="catalog root (containing tl/ subdir)")
    ap.add_argument("--ground", default=None,
                    help="single ground to render, e.g. 3_6/2x4_86")
    ap.add_argument("--results", default="mechanics_results.csv",
                    help="results CSV (used with --classification)")
    ap.add_argument("--classification", default=None,
                    help="render every ground with this classification, "
                         "e.g. homogeneously_auxetic")
    ap.add_argument("--output", "-o", default="lace_view.png",
                    help="output file path (single-ground mode)")
    ap.add_argument("--output-dir", default=None,
                    help="output directory (multi-ground mode)")
    ap.add_argument("--tiles", type=int, default=5,
                    help="number of unit cells per side (default 5)")
    ap.add_argument("--no-thread-view", action="store_true",
                    help="skip the thread sketch panel; pair diagram only")
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
        render_ground_lace_views(g, args.output,
                                  n_tiles=args.tiles,
                                  include_thread_view=not args.no_thread_view)
        return 0

    if args.classification:
        if not args.output_dir:
            print("--output-dir required with --classification", file=sys.stderr)
            return 1
        # iterate over CSV
        n = 0
        with open(args.results) as f:
            for row in csv.DictReader(f):
                if row["classification"] != args.classification:
                    continue
                family = row["family"]
                name = row["name"]
                path = os.path.join(args.catalog, "tl", family, name + ".txt")
                if not os.path.exists(path):
                    print(f"  skip (not found): {path}", file=sys.stderr)
                    continue
                g = parse_file(path, name=name, family=family)
                if g is None:
                    continue
                out_path = os.path.join(args.output_dir,
                                         f"{family}_{name}.png")
                render_ground_lace_views(g, out_path,
                                          n_tiles=args.tiles,
                                          include_thread_view=
                                            not args.no_thread_view)
                n += 1
        print(f"Rendered {n} grounds to {args.output_dir}")
        return 0

    print("specify either --ground or --classification", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
