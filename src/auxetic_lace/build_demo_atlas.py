"""
build_demo_atlas.py
===================

Builds a small demo atlas.json using hand-built graphs (no catalog
needed). Useful for:
  - Testing the atlas builder pipeline end-to-end
  - Providing a small demo dataset for the visualizer development
  - Smoke test before running the real catalog

Outputs `demo_atlas.json` with 5-10 representative graphs:
  - square_1x1 (cubic-like, isotropic, nu=0)
  - cloth_2x1 (simplest lace ground, isotropic)
  - kagome (well-known auxetic candidate)
  - hex_honeycomb (well-known, nu=1 in some directions)
  - re-entrant honeycomb (canonical auxetic)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from .parse_to_graph import LaceGraph, Edge
from .build_atlas import (
    build_ground_record,
    SPRING_K_ANG_GRID, BEAM_AR_GRID, THETA_GRID_DEG,
)


def make_square() -> LaceGraph:
    """1×1 square lattice."""
    return LaceGraph(
        name="square_1x1", family="test", keyword="Lattice Path",
        n_rows=1, n_cols=1,
        vertices=[(0, 0)],
        edges=[
            Edge(src_idx=0, dst_idx=0, wrap=(1, 0), polyline=()),
            Edge(src_idx=0, dst_idx=0, wrap=(0, 1), polyline=()),
        ],
    )


def make_cloth_2x1() -> LaceGraph:
    """2×1 cloth ground."""
    return LaceGraph(
        name="cloth_2x1", family="test_cloth", keyword="Lattice Path",
        n_rows=1, n_cols=2,
        vertices=[(0, 0), (1, 0)],
        edges=[
            Edge(src_idx=0, dst_idx=1, wrap=(0, 0), polyline=()),
            Edge(src_idx=0, dst_idx=0, wrap=(0, 1), polyline=()),
            Edge(src_idx=1, dst_idx=0, wrap=(1, 0), polyline=()),
            Edge(src_idx=1, dst_idx=1, wrap=(0, 1), polyline=()),
        ],
    )


def make_kagome() -> LaceGraph:
    """Kagome lattice approximated on a 2x2 square unit cell.

    True kagome is hexagonal; we approximate with a square cell containing
    3 vertices arranged as a triangle with corner-sharing equilateral
    triangles.

    Vertex layout (in unit cell coordinates):
      v0 = (0, 0)        bottom-left
      v1 = (1, 0)        bottom-right
      v2 = (0, 1)        upper-left

    Edges form triangles + connections to wrapped neighbors.
    Each vertex has 4 edges (4-coordinated, 2-regular as digraph).
    """
    return LaceGraph(
        name="kagome_2x2", family="test_kagome", keyword="Lattice Path",
        n_rows=2, n_cols=2,
        vertices=[(0, 0), (1, 0), (0, 1)],
        edges=[
            # v0 -> v1 (within cell)
            Edge(src_idx=0, dst_idx=1, wrap=(0, 0), polyline=()),
            # v0 -> v2 (within cell)
            Edge(src_idx=0, dst_idx=2, wrap=(0, 0), polyline=()),
            # v1 -> v0 (next cell, wrapping col)
            Edge(src_idx=1, dst_idx=0, wrap=(1, 0), polyline=()),
            # v1 -> v2 (with diagonal connection)
            Edge(src_idx=1, dst_idx=2, wrap=(0, 0), polyline=()),
            # v2 -> v0 (next cell up, wrapping row)
            Edge(src_idx=2, dst_idx=0, wrap=(0, 1), polyline=()),
            # v2 -> v1 (next cell, wrapping row but staying in col)
            Edge(src_idx=2, dst_idx=1, wrap=(0, 1), polyline=()),
        ],
    )


def make_reentrant_honeycomb() -> LaceGraph:
    """Re-entrant honeycomb — the canonical auxetic structure.

    A unit cell has 2 vertices at slightly inward-pointing positions,
    connected so that the angle between bars is < 90 degrees.

    Approximation on a 2x2 square cell (we'd need a proper hexagonal
    lattice for the true re-entrant geometry, but this gives the basic
    topology).

    For demo purposes we use a simpler "bowtie" with 2 vertices per cell.
    """
    return LaceGraph(
        name="reentrant_2x2", family="test_reentrant", keyword="Lattice Path",
        n_rows=2, n_cols=2,
        vertices=[(0, 0), (1, 1)],
        edges=[
            # v0 -> v1 within cell (diagonal)
            Edge(src_idx=0, dst_idx=1, wrap=(0, 0), polyline=()),
            # v0 -> v1 wrapping in col (anti-diagonal)
            Edge(src_idx=0, dst_idx=1, wrap=(-1, 0), polyline=()),
            # v1 -> v0 wrapping in row+col (back to origin in next cell)
            Edge(src_idx=1, dst_idx=0, wrap=(1, 1), polyline=()),
            # v1 -> v0 wrapping in row only
            Edge(src_idx=1, dst_idx=0, wrap=(0, 1), polyline=()),
        ],
    )


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/auxetic_lace/demo_atlas.json"
    output_dir = os.path.dirname(output) or "."
    thumb_dir = os.path.join(output_dir, "demo_thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    graphs = [
        make_square(),
        make_cloth_2x1(),
        make_kagome(),
        make_reentrant_honeycomb(),
    ]

    grounds = []
    print(f"Building demo atlas from {len(graphs)} test graphs...")
    t0 = time.time()
    for g in graphs:
        try:
            print(f"  Processing {g.family}/{g.name}...")

            # Render thumbnails
            ground_dir = os.path.join(thumb_dir, g.family, g.name)
            os.makedirs(ground_dir, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                from auxetic_lace.render_lace_view import render_ground_lace_views
                from auxetic_lace.render_lace_deformed import render_ground_deformed
                render_ground_lace_views(
                    g, os.path.join(ground_dir, "lace.png"), n_tiles=3)
                render_ground_deformed(g, os.path.join(ground_dir, "deformed.png"))
            except Exception as exc:
                print(f"    thumbnail render FAILED: {exc}")

            record = build_ground_record(g, name=g.name, family=g.family,
                                           thumbnail_dir="demo_thumbnails")
            grounds.append(record)
            spring_k01 = record["spring"]["nu_min"][SPRING_K_ANG_GRID.index(0.01)]
            beam_AR10 = record["beam"]["nu_min"][BEAM_AR_GRID.index(10.0)]
            spring_str = f"{spring_k01:+.3f}" if spring_k01 is not None else "N/A"
            beam_str = f"{beam_AR10:+.3f}" if beam_AR10 is not None else "N/A"
            print(f"    nu_min: spring(0.01)={spring_str}, beam(AR=10)={beam_str}")
        except Exception as exc:
            print(f"    FAIL: {exc}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0

    # Summary index
    spring_default_idx = (SPRING_K_ANG_GRID.index(0.01)
                           if 0.01 in SPRING_K_ANG_GRID else 1)
    beam_default_idx = (BEAM_AR_GRID.index(10.0)
                         if 10.0 in BEAM_AR_GRID else 1)
    summary = []
    for i, g in enumerate(grounds):
        s = {
            "idx": i,
            "name": g["name"],
            "family": g["family"],
            "n_rows": g["n_rows"],
            "n_cols": g["n_cols"],
            "n_vertices": g["n_vertices"],
            "spring_default_nu_min": g["spring"]["nu_min"][spring_default_idx],
            "spring_default_nu_max": g["spring"]["nu_max"][spring_default_idx],
            "spring_default_classification": g["spring"]["classification"][spring_default_idx],
            "beam_default_nu_min": g["beam"]["nu_min"][beam_default_idx],
            "beam_default_nu_max": g["beam"]["nu_max"][beam_default_idx],
            "beam_default_classification": g["beam"]["classification"][beam_default_idx],
        }
        summary.append(s)

    atlas = {
        "metadata": {
            "n_grounds": len(grounds),
            "spring_k_ang_grid": SPRING_K_ANG_GRID,
            "spring_default_idx": spring_default_idx,
            "beam_AR_grid": BEAM_AR_GRID,
            "beam_default_idx": beam_default_idx,
            "theta_grid_deg": THETA_GRID_DEG,
            "build_date": datetime.now().isoformat(),
            "build_elapsed_seconds": elapsed,
            "attribution": "Demo dataset — hand-built test graphs",
            "demo": True,
        },
        "summary": summary,
        "grounds": grounds,
        "failures": [],
    }

    with open(output, "w") as f:
        json.dump(atlas, f, separators=(",", ":"))
    size_kb = os.path.getsize(output) / 1024
    print(f"\nWrote {output} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
