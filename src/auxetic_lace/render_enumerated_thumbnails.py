"""
render_enumerated_thumbnails.py
================================

Renders lace + pair + thread thumbnails for enumerated grounds in
atlas.json. Skips deformed and dispersion views (those are meaningful
only for non-floppy lattices, and most enumerated grounds at small
sizes are floppy).

Reads ground records directly from atlas.json (no catalog file needed)
since enumerated grounds aren't backed by .txt files.

Idempotent: skips grounds that already have all expected PNGs unless
--force is passed.

USAGE:
    python3 -m auxetic_lace.render_enumerated_thumbnails \\
        --atlas docs/atlas.json \\
        --output-dir docs/thumbnails

    # Limit to a small sample for smoke-testing:
    python3 -m auxetic_lace.render_enumerated_thumbnails --limit 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless

from .parse_to_graph import LaceGraph, Edge, Vertex
from .render_lace_view import render_ground_lace_views_split


def dict_to_lace_graph(record: dict) -> LaceGraph:
    """Convert an atlas ground dict back into a LaceGraph object the
    renderers can consume."""
    vertices: List[Vertex] = [(v[0], v[1]) for v in record["vertices"]]
    edges: List[Edge] = []
    n_cols = record["n_cols"]
    n_rows = record["n_rows"]
    for e in record["edges"]:
        u, v = e["src"], e["dst"]
        wrap = tuple(e["wrap"])
        # Synthesize a 3-point polyline: src -> midpoint -> dst+wrap
        sx, sy = vertices[u]
        dx_in, dy_in = vertices[v]
        dx = dx_in + wrap[0] * n_cols
        dy = dy_in + wrap[1] * n_rows
        # midpoint (rounded to nearest int for polyline schema)
        mx = int(round((sx + dx) / 2.0))
        my = int(round((sy + dy) / 2.0))
        polyline = ((sx, sy), (mx, my), (dx, dy))
        edges.append(Edge(src_idx=u, dst_idx=v, wrap=wrap, polyline=polyline))
    return LaceGraph(
        name=record["name"],
        family=record["family"],
        keyword="enumerated",
        n_rows=n_rows,
        n_cols=n_cols,
        vertices=vertices,
        edges=edges,
    )


def thumbnail_paths(out_dir: str, family: str, name: str) -> dict:
    base = os.path.join(out_dir, family, name)
    return {
        "dir": base,
        "thread": os.path.join(base, "thread.png"),
    }


def render_one(record: dict, output_dir: str, n_tiles: int = 3,
                force: bool = False, verbose: bool = False) -> bool:
    """Render lace + pair + thread thumbnails for one ground.

    Returns True on success, False on any rendering error.
    """
    paths = thumbnail_paths(output_dir, record["family"], record["name"])

    # Skip if thread exists (idempotent)
    if not force and os.path.exists(paths["thread"]):
        return True

    try:
        graph = dict_to_lace_graph(record)
    except Exception as exc:
        if verbose:
            print(f"  build LaceGraph failed for {record['family']}/{record['name']}: {exc}")
        return False

    os.makedirs(paths["dir"], exist_ok=True)
    try:
        # Suppress matplotlib warnings about degenerate geometry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Render thread.png (chunky bands)
            render_ground_lace_views_split(
                graph, paths["thread"], n_tiles=n_tiles
            )
        return True
    except Exception as exc:
        if verbose:
            print(f"  render failed for {record['family']}/{record['name']}: {exc}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Render thumbnails for enumerated grounds in atlas.json"
    )
    ap.add_argument("--atlas", default="docs/atlas.json",
                    help="Atlas JSON to read")
    ap.add_argument("--output-dir", default="docs/thumbnails",
                    help="Where to write thumbnail PNGs")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only render the first N enumerated grounds (smoke test)")
    ap.add_argument("--n-tiles", type=int, default=3,
                    help="Tile count for periodic preview (default 3 -> 3x3 swatch)")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if PNGs already exist")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"Loading atlas from {args.atlas}...")
    import json
    with open(args.atlas) as f:
        atlas = json.load(f)

    # Filter to enumerated grounds
    enumerated = [
        g for g in atlas["grounds"]
        if g.get("provenance", {}).get("source", "").startswith("taylor")
    ]
    print(f"  {len(atlas['grounds'])} total grounds, "
          f"{len(enumerated)} enumerated")

    if args.limit is not None:
        enumerated = enumerated[:args.limit]
        print(f"  --limit {args.limit}: rendering only {len(enumerated)}")

    if not enumerated:
        print("Nothing to render.")
        return

    print()
    print(f"Rendering thumbnails for {len(enumerated)} enumerated grounds...")
    print(f"  output_dir: {args.output_dir}")
    print(f"  n_tiles: {args.n_tiles}")
    print()

    t0 = time.time()
    n_success = 0
    n_fail = 0
    for i, g in enumerate(enumerated):
        ok = render_one(g, args.output_dir, n_tiles=args.n_tiles,
                          force=args.force, verbose=args.verbose)
        if ok:
            n_success += 1
        else:
            n_fail += 1
        if (i + 1) % 50 == 0 or args.verbose:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(enumerated) - i - 1)
            print(f"  [{i+1}/{len(enumerated)}] "
                  f"{g['family']}/{g['name']}  "
                  f"({n_success} ok, {n_fail} fail; "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s.")
    print(f"  Success: {n_success}")
    print(f"  Failed:  {n_fail}")
    if n_fail > 0 and not args.verbose:
        print(f"  (Re-run with --verbose to see error messages.)")


if __name__ == "__main__":
    main()
