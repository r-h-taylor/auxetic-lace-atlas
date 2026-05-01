"""
parse_to_graph.py
=================

Convert TesseLace .txt template files into a clean toroidal graph data
structure suitable for downstream mechanical / topological analysis.

A TesseLace ground is a 2-regular doubly-periodic directed graph embedded
on a torus. The .txt format encodes this as:

    <Lattice Path|CHECKER>\\t<n_rows>\\t<n_cols>
    [x1,y1,x2,y2,x3,y3]\\t[x1,y1,x2,y2,x3,y3]\\t...

Each [x1,y1,x2,y2,x3,y3] cell specifies one VERTEX of the ground at
position (x1, y1) along with its TWO outgoing arcs:
    (x1, y1) -> (x2, y2)
    (x1, y1) -> (x3, y3)

This interpretation matches Veronika Irvine's reference Inkscape extension
(d-bl/inkscape-bobbinlace, lace_ground.py:draw), which renders each cell
as `self.line(x1,y1,x2,y2); self.line(x1,y1,x3,y3)`.

The (Lattice Path | CHECKER) keyword on the header line is metadata only;
the Inkscape extension reads it but ignores it. Both file types parse
identically.

Coordinate convention follows TesseLace: x is column, y is row, y increases
downward. The period parallelogram is [0, n_cols) x [0, n_rows). Cells
where any of the three positions falls outside this rectangle indicate
arcs wrapping across the torus.

This module produces a `LaceGraph` dataclass with:

    vertices            list of (col, row) tuples in [0, n_cols) x [0, n_rows)
    edges               list of Edge: src_vertex_idx, dst_vertex_idx,
                        polyline geometry, wrap (col, row) telling which
                        periodic image of the destination this edge actually
                        goes to

Plus convenience methods:

    get_2regular_violations()  list any vertex with !=2 in or !=2 out arcs
    cartesian_position(idx)    real-valued lattice position
    expand_to_finite_patch(N)  generate a NxN tile for FEA / visualization

This is the bridge between Irvine's combinatorial enumeration and
mechanical-metamaterial analysis: a faithful, validated graph
representation that any downstream pipeline can consume.

Attribution: ground patterns parsed by this module originate from
Veronika Irvine's TesseLace catalog, https://d-bl.github.io/tesselace-to-gf/,
licensed CC-BY 4.0. The format reading logic mirrors the BSD-licensed
Inkscape extension at https://github.com/d-bl/inkscape-bobbinlace.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Reuse parse_template from the scraper to keep one source of truth.
# This module is in the same directory as scrape_tesselace.py.
from .scrape_tesselace import parse_template


# -----------------------------------------------------------------------
# Coordinate utilities
# -----------------------------------------------------------------------

# A vertex coordinate is (col, row) with col in [0, n_cols), row in [0, n_rows).
# We use tuple[int, int] for vertex keys so they hash cleanly.

Vertex = Tuple[int, int]


def wrap_to_period(col: int, row: int,
                   n_cols: int, n_rows: int) -> Tuple[Vertex, Tuple[int, int]]:
    """Wrap a lattice point (col, row) into the period parallelogram.

    Returns:
        (wrapped_vertex, (wrap_col, wrap_row))
    where wrapped_vertex = (col % n_cols, row % n_rows) and (wrap_col, wrap_row)
    is the integer wrap offset (how many periods we crossed):
        col = wrapped_vertex[0] + wrap_col * n_cols
        row = wrapped_vertex[1] + wrap_row * n_rows
    """
    wc = col // n_cols
    wr = row // n_rows
    return ((col - wc * n_cols, row - wr * n_rows), (wc, wr))


# -----------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    """A single arc in the toroidal graph.

    src_idx and dst_idx index into LaceGraph.vertices (the canonical, in-period
    vertex list). wrap gives the periodic image of the destination this edge
    actually reaches:

        true_dst_position = vertex[dst_idx] + (wrap[0] * n_cols, wrap[1] * n_rows)

    polyline is the original list of (col, row) points specifying the arc's
    geometry, in extended coordinates (i.e. as given in the .txt file, NOT
    wrapped). For a typical arc this is a 3-point sequence
    [(x1,y1), (x2,y2), (x3,y3)].
    """
    src_idx: int
    dst_idx: int
    wrap: Tuple[int, int]                       # (wrap_col, wrap_row)
    polyline: Tuple[Tuple[int, int], ...]       # extended coords


@dataclass
class LaceGraph:
    """Toroidal graph representation of a single TesseLace ground."""
    name: str                                   # e.g. "2x4_111"
    family: str                                 # e.g. "3_5"
    keyword: str                                # "Lattice Path" | "CHECKER"
    n_rows: int
    n_cols: int
    vertices: List[Vertex]                      # canonical list, indices 0..V-1
    edges: List[Edge]                           # arcs

    # ---- derived / convenience methods ----

    def vertex_index(self, v: Vertex) -> int:
        """Index of vertex v in self.vertices (must already exist)."""
        return self._vertex_to_idx[v]

    def __post_init__(self):
        self._vertex_to_idx: Dict[Vertex, int] = {
            v: i for i, v in enumerate(self.vertices)
        }

    def degree(self) -> List[Tuple[int, int]]:
        """For each vertex, return (in_degree, out_degree)."""
        in_d = [0] * len(self.vertices)
        out_d = [0] * len(self.vertices)
        for e in self.edges:
            out_d[e.src_idx] += 1
            in_d[e.dst_idx] += 1
        return list(zip(in_d, out_d))

    def get_2regular_violations(self) -> List[Tuple[int, Vertex, int, int]]:
        """List vertices that violate the 2-regular invariant.
        Returns list of (idx, vertex, in_deg, out_deg) tuples."""
        out = []
        for i, (in_d, out_d) in enumerate(self.degree()):
            if in_d != 2 or out_d != 2:
                out.append((i, self.vertices[i], in_d, out_d))
        return out

    def cartesian_position(self, idx: int,
                           wrap: Tuple[int, int] = (0, 0)) -> Tuple[float, float]:
        """Return real-valued (x, y) position of a vertex in some periodic
        image. Origin is at (0, 0), x = col, y = row (TesseLace convention,
        y increases downward; flip y at render time if you want screen up)."""
        v = self.vertices[idx]
        return (float(v[0] + wrap[0] * self.n_cols),
                float(v[1] + wrap[1] * self.n_rows))

    def expand_to_finite_patch(self, n_tiles_x: int, n_tiles_y: int
                               ) -> Tuple[List[Tuple[float, float]],
                                          List[Tuple[int, int, Tuple[Tuple[float, float], ...]]]]:
        """Build a finite patch n_tiles_x by n_tiles_y copies of the period
        parallelogram. Returns (positions, edges) where:

            positions: list of (x, y) for each (vertex, tile_x, tile_y)
            edges: list of (src_index, dst_index, polyline_points) where
                   indices are into the positions list.

        Vertex indexing in the expanded patch:
            new_idx(vertex_idx, tx, ty) = vertex_idx + (ty * n_tiles_x + tx) * V

        Edges that cross periodic boundaries connect to the appropriate
        neighboring tile. Edges whose destination falls outside the patch
        (i.e. tx + e.wrap[0] not in [0, n_tiles_x)) are dropped to give a
        clean finite specimen — set n_tiles_x, n_tiles_y large enough that
        boundary effects don't dominate any analysis you do.
        """
        V = len(self.vertices)
        positions: List[Tuple[float, float]] = []
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                for v in self.vertices:
                    positions.append((float(v[0] + tx * self.n_cols),
                                      float(v[1] + ty * self.n_rows)))

        def idx_of(vidx: int, tx: int, ty: int) -> int:
            return vidx + (ty * n_tiles_x + tx) * V

        edges_out: List[Tuple[int, int, Tuple[Tuple[float, float], ...]]] = []
        for e in self.edges:
            for ty in range(n_tiles_y):
                for tx in range(n_tiles_x):
                    dst_tx = tx + e.wrap[0]
                    dst_ty = ty + e.wrap[1]
                    if not (0 <= dst_tx < n_tiles_x and 0 <= dst_ty < n_tiles_y):
                        continue
                    si = idx_of(e.src_idx, tx, ty)
                    di = idx_of(e.dst_idx, dst_tx, dst_ty)
                    # Shift polyline by (tx*n_cols, ty*n_rows) so the start
                    # is at the source's tile, and let it extend into the
                    # destination tile naturally.
                    poly_shifted = tuple(
                        (float(p[0] + tx * self.n_cols),
                         float(p[1] + ty * self.n_rows))
                        for p in e.polyline
                    )
                    edges_out.append((si, di, poly_shifted))
        return positions, edges_out


# -----------------------------------------------------------------------
# Parsing logic
# -----------------------------------------------------------------------

def _arcs_to_graph(parsed: Dict, name: str, family: str) -> LaceGraph:
    """Given the dict returned by parse_template, build a LaceGraph.

    Each cell in the template, [x1,y1,x2,y2,x3,y3], specifies a single
    VERTEX at (x1, y1) together with its TWO outgoing arcs:
        (x1, y1) -> (x2, y2)
        (x1, y1) -> (x3, y3)

    This matches the behavior of Veronika Irvine's reference Inkscape
    extension (lace_ground.py, draw() function), which renders each cell as
    `self.line(x1,y1,x2,y2); self.line(x1,y1,x3,y3)`.

    The 2-regular invariant arises naturally:
      - each vertex appears as (x1, y1) in exactly one cell, contributing
        2 outgoing arcs
      - each vertex is the destination (x2, y2) or (x3, y3) of exactly two
        arcs from other cells, contributing 2 incoming arcs
    """
    n_rows = parsed["n_rows"]
    n_cols = parsed["n_cols"]

    # Vertices: any lattice point that appears as (x1, y1), (x2, y2), or
    # (x3, y3) in any cell, wrapped into [0, n_cols) x [0, n_rows). All
    # three positions per cell are vertices in the graph (per Inkscape
    # extension behavior); the middle position is not a "bend point", it's
    # the endpoint of one of the two outgoing arcs.
    vertex_set: Dict[Vertex, None] = {}

    # Each cell yields two arcs.
    raw_arcs: List[Dict] = []

    for cell in parsed["arcs"]:
        x1, y1, x2, y2, x3, y3 = cell

        # Source vertex (origin of both outgoing arcs from this cell)
        src_wrap, src_off = wrap_to_period(x1, y1, n_cols, n_rows)
        vertex_set.setdefault(src_wrap, None)

        # Two destinations
        for (xd, yd) in [(x2, y2), (x3, y3)]:
            dst_wrap, dst_off = wrap_to_period(xd, yd, n_cols, n_rows)
            vertex_set.setdefault(dst_wrap, None)
            # Wrap is the dst tile relative to the src tile. We anchor the
            # arc at the source tile, so src is in tile (0, 0) and dst is
            # in tile dst_off - src_off.
            wrap_relative = (dst_off[0] - src_off[0],
                             dst_off[1] - src_off[1])
            # Polyline: just the two endpoints. We could also store a
            # straight-line polyline here for downstream rendering — keep it
            # in source-anchored coordinates (subtract src_off so that the
            # start point equals src_wrap).
            polyline = (
                (x1 - src_off[0] * n_cols, y1 - src_off[1] * n_rows),
                (xd - src_off[0] * n_cols, yd - src_off[1] * n_rows),
            )
            assert polyline[0] == src_wrap, \
                f"shift bug: got {polyline[0]}, expected {src_wrap}"
            raw_arcs.append({
                "src": src_wrap,
                "dst": dst_wrap,
                "wrap": wrap_relative,
                "polyline": polyline,
            })

    # Deterministic vertex ordering: by (row, col)
    vertices = sorted(vertex_set.keys(), key=lambda v: (v[1], v[0]))
    vidx = {v: i for i, v in enumerate(vertices)}

    edges = []
    for arc in raw_arcs:
        edges.append(Edge(
            src_idx=vidx[arc["src"]],
            dst_idx=vidx[arc["dst"]],
            wrap=arc["wrap"],
            polyline=arc["polyline"],
        ))

    return LaceGraph(
        name=name,
        family=family,
        keyword=parsed["keyword"],
        n_rows=n_rows,
        n_cols=n_cols,
        vertices=vertices,
        edges=edges,
    )


def parse_file(path: str, name: Optional[str] = None,
               family: Optional[str] = None) -> Optional[LaceGraph]:
    """Read a single .txt file and return a LaceGraph. None on failure."""
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
    if family is None:
        # Try to infer family from parent directory
        family = os.path.basename(os.path.dirname(path))
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"  could not read {path}: {e}", file=sys.stderr)
        return None
    parsed = parse_template(text)
    if parsed is None:
        return None
    return _arcs_to_graph(parsed, name=name, family=family)


def parse_manifest(manifest_csv: str, base_dir: str = ""
                   ) -> List[LaceGraph]:
    """Parse every ground listed in a manifest CSV. base_dir is prepended
    to local_path if local_path is relative."""
    out: List[LaceGraph] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local = row["local_path"]
            if not os.path.isabs(local) and base_dir:
                local = os.path.join(base_dir, local)
            g = parse_file(local, name=row["name"], family=row["family"])
            if g is not None:
                out.append(g)
    return out


# -----------------------------------------------------------------------
# Validation report
# -----------------------------------------------------------------------

def validation_report(graphs: List[LaceGraph]) -> Dict:
    """Compute statistics + invariant violations across a set of graphs."""
    stats = {
        "total": len(graphs),
        "with_violations": 0,
        "violations": [],
        "size_distribution": defaultdict(int),
        "family_distribution": defaultdict(int),
        "keyword_distribution": defaultdict(int),
        "vertex_count_distribution": defaultdict(int),
        "edge_count_distribution": defaultdict(int),
    }
    for g in graphs:
        viols = g.get_2regular_violations()
        if viols:
            stats["with_violations"] += 1
            stats["violations"].append({
                "name": g.name, "family": g.family,
                "size": (g.n_rows, g.n_cols),
                "violations": viols,
            })
        stats["size_distribution"][(g.n_rows, g.n_cols)] += 1
        stats["family_distribution"][g.family] += 1
        stats["keyword_distribution"][g.keyword] += 1
        stats["vertex_count_distribution"][len(g.vertices)] += 1
        stats["edge_count_distribution"][len(g.edges)] += 1
    return stats


def print_report(stats: Dict) -> None:
    print(f"Total graphs:          {stats['total']}")
    print(f"With 2-regular issues: {stats['with_violations']}")
    print()
    print("Family distribution:")
    for fam in sorted(stats["family_distribution"]):
        print(f"  {fam:<12s} {stats['family_distribution'][fam]}")
    print()
    print("Keyword distribution:")
    for kw in sorted(stats["keyword_distribution"]):
        print(f"  {kw:<14s} {stats['keyword_distribution'][kw]}")
    print()
    print("Period parallelogram size distribution (top 10):")
    sized = sorted(stats["size_distribution"].items(),
                   key=lambda kv: -kv[1])[:10]
    for size, count in sized:
        print(f"  {size[0]}x{size[1]:<5d} {count}")
    print()
    print("Vertices-per-ground distribution (top 10):")
    vd = sorted(stats["vertex_count_distribution"].items(),
                key=lambda kv: kv[0])[:10]
    for nv, count in vd:
        print(f"  {nv:>3d} verts: {count}")
    print()
    if stats["violations"]:
        print(f"First few violations (out of {len(stats['violations'])}):")
        for v in stats["violations"][:5]:
            print(f"  {v['family']}/{v['name']} [{v['size'][0]}x{v['size'][1]}]")
            for idx, vert, in_d, out_d in v["violations"][:3]:
                print(f"    vertex {idx} {vert}: in={in_d}, out={out_d}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Parse TesseLace catalog into LaceGraph data structures, "
                    "report statistics, and validate invariants.")
    ap.add_argument("--catalog", "-c", default="tesselace_catalog",
                    help="catalog directory containing manifest.csv "
                         "(default: tesselace_catalog)")
    ap.add_argument("--single", "-s", default=None,
                    help="parse only this single file path (debug)")
    ap.add_argument("--show", action="store_true",
                    help="print full info for each parsed graph")
    args = ap.parse_args()

    if args.single:
        g = parse_file(args.single)
        if g is None:
            print(f"Failed to parse {args.single}")
            return 1
        print(f"Parsed: {g.family}/{g.name}")
        print(f"  keyword:  {g.keyword}")
        print(f"  size:     {g.n_rows} x {g.n_cols}")
        print(f"  vertices: {len(g.vertices)}")
        print(f"  edges:    {len(g.edges)}")
        viols = g.get_2regular_violations()
        if viols:
            print(f"  ! 2-regular violations: {len(viols)}")
            for idx, v, in_d, out_d in viols:
                print(f"    vertex {idx} at {v}: in={in_d}, out={out_d}")
        else:
            print("  2-regular: OK")
        if args.show:
            print("  vertex list:")
            for i, v in enumerate(g.vertices):
                print(f"    {i:3d}: {v}")
            print("  edge list:")
            for e in g.edges:
                print(f"    {e.src_idx} -> {e.dst_idx} wrap={e.wrap} "
                      f"poly={e.polyline}")
        return 0

    manifest = os.path.join(args.catalog, "manifest.csv")
    if not os.path.exists(manifest):
        print(f"manifest not found at {manifest}")
        print("Run scrape_tesselace.py first.")
        return 1

    print(f"Parsing all grounds from {manifest}...")
    graphs = parse_manifest(manifest, base_dir="")
    print(f"  parsed {len(graphs)} grounds")
    print()
    stats = validation_report(graphs)
    print_report(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
