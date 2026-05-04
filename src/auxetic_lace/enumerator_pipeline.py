"""
enumerator_pipeline.py
=======================

Generates new candidate grounds at small cell sizes via brute-force
4-regular periodic graph enumeration, computes their manufacturability
properties + physics + canonical forms, and appends them to atlas.json.

Pipeline stages:

  1. Enumerate undirected 4-regular periodic graphs at small vertex counts
  2. Filter out duplicates of Irvine grounds (by graph_canonical)
  3. For each new graph, find a 2-in-2-out orientation
  4. Build a LaceGraph object with grid-placed vertices
  5. Compute manufacturability properties
  6. Compute mechanics (spring + beam) physics
  7. Render thumbnails (lace, pair, thread, deformed)
  8. Compute canonical-form fingerprints
  9. Append to atlas.json

Design choices (as discussed):
  - All enumerated grounds added (not filtered by lace-workability)
  - One atlas entry per undirected graph (one chosen orientation)
  - Cell sizes: 1, 2, 3 vertices

USAGE:
    python3 -m auxetic_lace.enumerator_pipeline \\
        --atlas docs/atlas.json \\
        --output docs/atlas.json \\
        --thumbnail-dir docs/thumbnails \\
        --max-vertices 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .canonicalize import graph_canonical, lace_canonical
from .manufacturability import (
    is_2in2out, manufacturability_block, provenance_block
)
from .lace_workability import is_lace_workable
from .parse_to_graph import LaceGraph, Edge, Vertex
from .build_atlas import build_ground_record


# ---------------------------------------------------------------------------
# Stage 1: undirected 4-regular periodic graph enumeration
# ---------------------------------------------------------------------------

def _all_edge_types(n_vertices: int, max_wrap: int = 1
                     ) -> List[Tuple[int, int, Tuple[int, int]]]:
    """All possible undirected edge types (u, v, wrap) with u <= v.

    Self-loops with zero wrap are excluded as not physically meaningful.
    For self-loops (u == v), wrap is canonicalized to lex-smaller of
    (wrap, -wrap).
    """
    types = set()
    for u in range(n_vertices):
        for v in range(n_vertices):
            for wx in range(-max_wrap, max_wrap + 1):
                for wy in range(-max_wrap, max_wrap + 1):
                    if u == v and wx == 0 and wy == 0:
                        continue
                    if u < v:
                        types.add((u, v, (wx, wy)))
                    elif v < u:
                        types.add((v, u, (-wx, -wy)))
                    else:
                        neg = (-wx, -wy)
                        cw = min((wx, wy), neg)
                        types.add((u, u, cw))
    return sorted(types)


def _degree_contribution(edge: Tuple[int, int, Tuple[int, int]],
                          n_vertices: int) -> List[int]:
    u, v, _ = edge
    deg = [0] * n_vertices
    if u == v:
        deg[u] += 2
    else:
        deg[u] += 1
        deg[v] += 1
    return deg


def _enumerate_4regular_multisets(edge_types, n_vertices: int,
                                    max_multiplicity: int = 4):
    """Yield each multiset of edges that gives every vertex degree 4."""
    target = [4] * n_vertices
    n = len(edge_types)

    def backtrack(idx, current_mults, current_deg):
        if idx == n:
            if current_deg == target:
                multiset = []
                for i, m in enumerate(current_mults):
                    for _ in range(m):
                        multiset.append(edge_types[i])
                yield tuple(sorted(multiset))
            return
        edge = edge_types[idx]
        contrib = _degree_contribution(edge, n_vertices)
        max_m = max_multiplicity
        for v_i, c in enumerate(contrib):
            if c > 0:
                remaining = target[v_i] - current_deg[v_i]
                if remaining < 0:
                    return
                max_m = min(max_m, remaining // c)
        for m in range(0, max_m + 1):
            new_deg = [current_deg[i] + m * contrib[i] for i in range(n_vertices)]
            if any(new_deg[i] > target[i] for i in range(n_vertices)):
                continue
            current_mults.append(m)
            yield from backtrack(idx + 1, current_mults, new_deg)
            current_mults.pop()

    yield from backtrack(0, [], [0] * n_vertices)


def _multiset_to_pseudo_ground(multiset, n_vertices, n_cols, n_rows) -> dict:
    """Build a ground-shaped dict that graph_canonical can hash.
    Vertices placed at integer grid points; edges are turned into
    src/dst/wrap dicts (treating each undirected edge as src=u, dst=v).
    """
    vertices = []
    for i in range(n_vertices):
        col = i % n_cols
        row = i // n_cols
        vertices.append([col, row])
    edges = [{"src": u, "dst": v, "wrap": list(w)} for (u, v, w) in multiset]
    return {
        "family": "_enum_temp",
        "name": "_enum_temp",
        "n_cols": n_cols,
        "n_rows": n_rows,
        "n_vertices": n_vertices,
        "n_edges": len(edges),
        "vertices": vertices,
        "edges": edges,
        "lattice": [[1.0, 0.0], [0.0, 1.0]],
    }


def enumerate_undirected_4regular(n_vertices: int, n_cols: int, n_rows: int,
                                    max_wrap: int = 1
                                    ) -> List[Tuple[List, str]]:
    """Generate all distinct (by graph_canonical) undirected 4-regular
    periodic graphs at the given vertex/cell size.

    Returns a list of (multiset, canonical_hash) pairs.
    """
    edge_types = _all_edge_types(n_vertices, max_wrap=max_wrap)
    seen = set()
    representatives = []
    for multiset in _enumerate_4regular_multisets(edge_types, n_vertices):
        # Filter: skip any multiset with parallel-duplicate edges (multigraphs).
        # An edge that appears more than once with identical (u, v, wrap) means
        # two physical sticks occupying the same periodic copy of space —
        # physically degenerate, excluded from the catalog.
        if len(set(multiset)) != len(multiset):
            continue
        ground = _multiset_to_pseudo_ground(multiset, n_vertices, n_cols, n_rows)
        try:
            cert = graph_canonical(ground)
        except Exception:
            continue
        if cert in seen:
            continue
        seen.add(cert)
        representatives.append((multiset, cert))
    return representatives


# ---------------------------------------------------------------------------
# Stage 3: 2-in-2-out orientation finder
# ---------------------------------------------------------------------------

def find_2in2out_orientation(multiset, n_vertices: int
                              ) -> Optional[List[Tuple[int, int, Tuple[int, int]]]]:
    """Given an undirected 4-regular periodic edge multiset, find a
    2-in-2-out directed orientation.

    Strategy: greedy/backtracking. Process edges in order; for each
    non-self-loop edge (u, v, wrap), try both orientations (u → v) and
    (v → u, with negated wrap) — pick the one that doesn't push any
    vertex over the 2-in-or-2-out cap. Self-loops always orient as
    (u, u, wrap) — a self-loop in either direction is balanced by its
    other end.

    Returns a list of directed edges as (src, dst, wrap) tuples, or None
    if no valid orientation found.
    """
    n = len(multiset)
    # Separate self-loops from non-self-loops
    edges_idx = list(range(n))

    # Pre-orient self-loops (they always contribute 1 in + 1 out at their vertex)
    oriented = [None] * n
    in_deg = [0] * n_vertices
    out_deg = [0] * n_vertices

    for i, (u, v, w) in enumerate(multiset):
        if u == v:
            oriented[i] = (u, v, w)
            out_deg[u] += 1
            in_deg[v] += 1

    # Backtrack on non-self-loop edges
    non_self = [i for i in edges_idx if oriented[i] is None]

    def try_orient(k):
        if k == len(non_self):
            return True
        ei = non_self[k]
        u, v, w = multiset[ei]
        # Try (u → v, w)
        if out_deg[u] < 2 and in_deg[v] < 2:
            oriented[ei] = (u, v, w)
            out_deg[u] += 1; in_deg[v] += 1
            if try_orient(k + 1):
                return True
            out_deg[u] -= 1; in_deg[v] -= 1
            oriented[ei] = None
        # Try (v → u, -w)
        if out_deg[v] < 2 and in_deg[u] < 2:
            oriented[ei] = (v, u, (-w[0], -w[1]))
            out_deg[v] += 1; in_deg[u] += 1
            if try_orient(k + 1):
                return True
            out_deg[v] -= 1; in_deg[u] -= 1
            oriented[ei] = None
        return False

    if not try_orient(0):
        return None
    # Sanity check
    for v in range(n_vertices):
        if in_deg[v] != 2 or out_deg[v] != 2:
            return None
    return list(oriented)


# ---------------------------------------------------------------------------
# Stage 4: build a LaceGraph from oriented edges
# ---------------------------------------------------------------------------

def build_lace_graph(oriented_edges, n_vertices: int, n_cols: int, n_rows: int,
                      family: str, name: str) -> LaceGraph:
    """Build a LaceGraph object from oriented edges + grid-placed vertices."""
    vertices: List[Vertex] = []
    for i in range(n_vertices):
        col = i % n_cols
        row = i // n_cols
        vertices.append((col, row))
    edges: List[Edge] = []
    for (u, v, w) in oriented_edges:
        # polyline: 3-point sequence from src through midpoint to dst+wrap.
        # Use straight-line midpoint for synthesized grounds.
        sx, sy = vertices[u]
        dx_in, dy_in = vertices[v]
        dx = dx_in + w[0] * n_cols
        dy = dy_in + w[1] * n_rows
        mx = (sx + dx) // 2 if (sx + dx) % 2 == 0 else (sx + dx) / 2.0
        my = (sy + dy) // 2 if (sy + dy) % 2 == 0 else (sy + dy) / 2.0
        # Edge expects integer coordinates in polyline; round to nearest
        polyline = ((sx, sy), (int(round(mx)), int(round(my))), (dx, dy))
        edges.append(Edge(src_idx=u, dst_idx=v, wrap=w, polyline=polyline))
    return LaceGraph(
        name=name, family=family, keyword="enumerated",
        n_rows=n_rows, n_cols=n_cols,
        vertices=vertices, edges=edges,
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

CELL_FOR_VERTEX_COUNT = {
    1: (1, 1),
    2: (2, 1),
    3: (3, 1),
}


def generate_new_grounds(existing_atlas: dict, max_vertices: int = 3,
                          enumerator_run: str = None,
                          thumbnail_dir: Optional[str] = None,
                          compute_physics: bool = True,
                          verbose: bool = False
                          ) -> List[dict]:
    """Top-level orchestrator.

    Returns a list of full ground records (matching the schema of Irvine
    grounds in atlas.json) with the additional `manufacturability`,
    `provenance`, `graph_canonical`, `lace_canonical` blocks.

    Parameters
    ----------
    existing_atlas : dict
        Loaded atlas to deduplicate against (uses graph_canonical field).
    max_vertices : int
        Max cell vertex count to enumerate (1, 2, ..., max_vertices).
    enumerator_run : str
        Identifier for this enumerator run (timestamp-based by default).
    thumbnail_dir : str
        URL-prefix for thumbnails in the output. None = omit thumbnails.
    compute_physics : bool
        Whether to run mechanics/phonons/humidity. Set False for fast
        smoke testing.
    verbose : bool
        Print progress.
    """
    if enumerator_run is None:
        enumerator_run = f"enum_{int(time.time())}"

    # Collect existing graph_canonicals to deduplicate
    existing_certs = {g.get("graph_canonical") for g in existing_atlas["grounds"]
                       if "graph_canonical" in g}
    if verbose:
        print(f"Existing atlas has {len(existing_certs)} distinct graph_canonicals")

    new_grounds = []
    for n_vertices in range(1, max_vertices + 1):
        n_cols, n_rows = CELL_FOR_VERTEX_COUNT[n_vertices]
        if verbose:
            print(f"\n--- {n_vertices} vertex/vertices "
                  f"(cell {n_cols}x{n_rows}) ---")
        reps = enumerate_undirected_4regular(n_vertices, n_cols, n_rows)
        if verbose:
            print(f"  {len(reps)} unique undirected 4-regular graphs")

        new_at_size = 0
        not_oriented = 0
        for serial, (multiset, cert) in enumerate(reps, start=1):
            if cert in existing_certs:
                continue
            oriented = find_2in2out_orientation(multiset, n_vertices)
            if oriented is None:
                not_oriented += 1
                continue

            family = "taylor"
            name = f"V{n_vertices}_{n_cols}x{n_rows}_{serial:03d}"
            lg = build_lace_graph(oriented, n_vertices, n_cols, n_rows,
                                   family, name)

            if compute_physics:
                # Use build_ground_record to compute all physics blocks
                try:
                    record = build_ground_record(
                        lg, name=name, family=family,
                        thumbnail_dir=thumbnail_dir,
                    )
                except Exception as exc:
                    if verbose:
                        print(f"  physics FAIL on {name}: {exc}")
                    continue
            else:
                # Build minimal record (vertices/edges only)
                vertices = [[v[0], v[1]] for v in lg.vertices]
                edges = [{"src": e.src_idx, "dst": e.dst_idx,
                          "wrap": [e.wrap[0], e.wrap[1]]} for e in lg.edges]
                record = {
                    "family": family,
                    "name": name,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_vertices": n_vertices,
                    "n_edges": len(edges),
                    "vertices": vertices,
                    "edges": edges,
                    "lattice": [[1.0, 0.0], [0.0, 1.0]],
                    "cell_area": float(n_cols * n_rows),
                }

            # Manufacturability + provenance + canonical
            record["manufacturability"] = manufacturability_block(
                record, source="enumerated")
            record["provenance"] = provenance_block(
                source=f"taylor_{n_cols}x{n_rows}",
                irvine_label=None,
                enumerator_run=enumerator_run,
            )
            record["graph_canonical"] = cert
            record["lace_canonical"] = lace_canonical(record)

            new_grounds.append(record)
            existing_certs.add(cert)
            new_at_size += 1
        if verbose:
            print(f"  {new_at_size} added; {not_oriented} could not be oriented")
    return new_grounds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate new enumerated grounds and append to atlas.json"
    )
    ap.add_argument("--atlas", default="docs/atlas.json",
                    help="Existing atlas.json to read")
    ap.add_argument("--output", default="docs/atlas.json",
                    help="Where to write augmented atlas")
    ap.add_argument("--max-vertices", type=int, default=3,
                    help="Max cell vertex count to enumerate (default 3)")
    ap.add_argument("--no-physics", action="store_true",
                    help="Skip mechanics computation (fast smoke test only)")
    ap.add_argument("--thumbnail-dir", type=str, default="thumbnails",
                    help="Relative URL prefix for thumbnails in output. "
                         "Note: this records URLs in the atlas; thumbnails "
                         "themselves still need to be rendered separately.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute new grounds but don't write atlas")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"Loading atlas from {args.atlas}...")
    with open(args.atlas) as f:
        atlas = json.load(f)
    n_existing = len(atlas["grounds"])
    print(f"  {n_existing} existing grounds")
    print()

    print(f"Generating new enumerated grounds up to {args.max_vertices} vertices...")
    t0 = time.time()
    new_grounds = generate_new_grounds(
        atlas,
        max_vertices=args.max_vertices,
        thumbnail_dir=args.thumbnail_dir if args.thumbnail_dir else None,
        compute_physics=not args.no_physics,
        verbose=args.verbose,
    )
    elapsed = time.time() - t0
    print()
    print(f"Generated {len(new_grounds)} new grounds in {elapsed:.1f}s")
    print()

    # Stats
    n_lace_workable = sum(1 for g in new_grounds
                           if g["manufacturability"]["is_lace_workable"])
    print(f"  is_lace_workable=True: {n_lace_workable} / {len(new_grounds)}")
    print(f"  is_lace_workable=False: "
          f"{len(new_grounds) - n_lace_workable} / {len(new_grounds)}")
    print()

    if args.dry_run:
        print("(--dry-run) atlas.json NOT modified")
        return

    # Append new grounds
    atlas["grounds"].extend(new_grounds)
    atlas["metadata"]["n_grounds"] = len(atlas["grounds"])
    atlas["metadata"]["n_enumerated"] = len(new_grounds)
    atlas["metadata"]["enumerator_run_time"] = elapsed

    # Update summary block (same shape as build_atlas.py builds)
    spring_default_idx = atlas["metadata"].get("spring_default_idx", 2)
    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)
    summary = atlas.get("summary", [])
    next_idx = len(summary)
    for g in new_grounds:
        if "spring" in g and "beam" in g:
            try:
                s = {
                    "idx": next_idx,
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
                next_idx += 1
            except (KeyError, IndexError):
                pass  # Ground has no physics; skip
    atlas["summary"] = summary

    # Write
    print(f"Writing {args.output}...")
    with open(args.output, "w") as f:
        json.dump(atlas, f, separators=(",", ":"))
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  {size_mb:.2f} MB")
    print(f"  Total grounds in atlas: {len(atlas['grounds'])}")
    print()
    print("To render thumbnails for the new grounds, run a separate")
    print("rendering script that takes ATLAS.grounds entries (the grounds")
    print("themselves contain enough info to draw lace and deformed views).")


if __name__ == "__main__":
    main()
