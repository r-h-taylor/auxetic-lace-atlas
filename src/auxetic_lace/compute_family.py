"""
compute_family.py — face-set "family" labels for periodic 4-regular
planar graphs (lace grounds).

The canonical family identifier is the sorted set of face sizes in the
planar embedding, joined with '_'. Examples:
    {3, 6}        -> "3_6"     (kagome / trihexagonal)
    {4}           -> "4"       (square tiling)
    {3, 4, 6, 8}  -> "3_4_6_8"

This matches Irvine's TesseLace folder structure for non-traditionally-
named grounds. Some grounds also have a traditional name (cloth, kat,
diamond, rose, bias, ...). The traditional name is kept in a separate
`traditional_name` field — it is metadata about heritage, not a
substitute for the face-set identifier.

Repo conventions (from src/auxetic_lace/parse_to_graph.py):
    - LaceGraph stores DIRECTED arcs in `edges`; each vertex has
      2 in + 2 out arcs (4-regular as undirected).
    - Vertex coords are integer (col, row) with y increasing downward.
    - Edge.polyline is a 2-tuple of points in SOURCE-ANCHORED ABSOLUTE
      lattice coordinates: polyline[0] equals the src vertex's
      coordinate, polyline[-1] equals dst's "extended" position
      (= dst vertex coord + wrap * (n_cols, n_rows)).

Public API:
    family_label(graph)           -> "3_6"
    face_sizes(graph)             -> [3, 3, 6, 6, 6, ...]   (multiset)
    trace_faces(graph)            -> [[dart, dart, ...], ...]
    trace_faces_strict(graph)     -> same; raises if any face has wrap != (0,0)
    build_rotation_system(graph)  -> {v: [dart, ...]}      (sorted by atan2)
    assign_traditional_name(name, old_family) -> str | None

Algorithm:
1. Build a rotation system at each vertex (cyclic order of incident
   "darts" sorted by departure angle). A dart is a directed arc
   (edge_idx, direction) where direction=0 is the forward arc and
   direction=1 is the reverse arc.
2. From an unvisited dart, walk a face: at the destination vertex,
   take the *next* dart in the cyclic order after the reverse of the
   arriving dart.
3. Each closed walk traces one face. Record its length.
4. Collect distinct face lengths, sort, join with '_'.

On a torus, each face walk closes with cumulative wrap (0, 0). Use
trace_faces_strict() to assert this.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math


from .parse_to_graph import LaceGraph, Edge


# ---------------------------------------------------------------------------
# Rotation system
# ---------------------------------------------------------------------------

def build_rotation_system(graph) -> dict:
    """
    For each vertex v, return the cyclic ordering of incident darts,
    sorted by atan2 of the departure direction.

    A dart is identified by `(edge_idx, direction)`:
        direction == 0  -> forward arc (leaves edge.src_idx toward edge.dst_idx)
        direction == 1  -> reverse arc (leaves edge.dst_idx toward edge.src_idx)

    Departure direction:
        forward dart from src: vector from polyline[0] toward polyline[1]
                               (polyline[0] equals src's position)
        reverse dart from dst: vector from polyline[-1] toward polyline[-2]
                               (polyline[-1] equals dst's extended position)

    Sort order is by atan2(dy, dx). Note that the repo uses y-down
    coordinates, so atan2 ordering corresponds to CW visual order;
    chirality of the resulting face trace is consistent throughout
    the module so face SIZES come out correct either way.
    """
    n = len(graph.vertices)
    incidences = defaultdict(list)

    for e_idx, e in enumerate(graph.edges):
        # forward dart: leaves src
        # departure direction = polyline[1] - polyline[0]
        # (polyline[0] is src's position, polyline[1] is next polyline point;
        # for a 2-point polyline this is dst's extended position)
        if not e.polyline or len(e.polyline) < 2:
            # fall back: build a virtual straight polyline
            src_pos = graph.vertices[e.src_idx]
            dst_pos_ext = (graph.vertices[e.dst_idx][0] + e.wrap[0] * graph.n_cols,
                           graph.vertices[e.dst_idx][1] + e.wrap[1] * graph.n_rows)
            p_first, p_second = src_pos, dst_pos_ext
            p_last_minus_one, p_last = src_pos, dst_pos_ext
        else:
            p_first = e.polyline[0]
            p_second = e.polyline[1]
            p_last_minus_one = e.polyline[-2]
            p_last = e.polyline[-1]

        ang_fwd = math.atan2(p_second[1] - p_first[1],
                             p_second[0] - p_first[0])
        incidences[e.src_idx].append((ang_fwd, (e_idx, 0)))

        ang_rev = math.atan2(p_last_minus_one[1] - p_last[1],
                             p_last_minus_one[0] - p_last[0])
        incidences[e.dst_idx].append((ang_rev, (e_idx, 1)))

    rotation = {}
    for v in range(n):
        sorted_darts = [d for _, d in sorted(incidences[v], key=lambda x: x[0])]
        rotation[v] = sorted_darts
    return rotation


# ---------------------------------------------------------------------------
# Face tracing
# ---------------------------------------------------------------------------

def reverse_dart(dart):
    e_idx, d = dart
    return (e_idx, 1 - d)


def dart_endpoints(graph, dart):
    """Return (tail_vertex, head_vertex, wrap) for a dart."""
    e_idx, d = dart
    e = graph.edges[e_idx]
    if d == 0:
        return e.src_idx, e.dst_idx, e.wrap
    return e.dst_idx, e.src_idx, (-e.wrap[0], -e.wrap[1])


def trace_faces(graph):
    """
    Return a list of face walks. Each walk is a list of darts forming
    a closed face boundary.
    """
    walks, _ = _trace_faces_with_wrap(graph)
    return walks


def trace_faces_strict(graph):
    """Like trace_faces, but raises if any face closes with non-zero wrap."""
    walks, wraps = _trace_faces_with_wrap(graph)
    for walk, wrap in zip(walks, wraps):
        if wrap != (0, 0):
            raise ValueError(
                f"Face trace closed with non-zero wrap {wrap}; "
                f"walk length {len(walk)}; first dart {walk[0]}. "
                f"Embedding may be non-planar or wrap accounting is off."
            )
    return walks


def _trace_faces_with_wrap(graph):
    rotation = build_rotation_system(graph)

    pos_in_rotation = {}
    for v, darts in rotation.items():
        for i, d in enumerate(darts):
            pos_in_rotation[(v, d)] = i

    visited = set()
    walks = []
    wraps = []
    for e_idx in range(len(graph.edges)):
        for direction in (0, 1):
            start = (e_idx, direction)
            if start in visited:
                continue
            walk = []
            cur = start
            wrap_acc = (0, 0)
            while True:
                if cur in visited:
                    raise RuntimeError(
                        f"Face trace re-entered visited dart {cur}; "
                        f"rotation system is inconsistent."
                    )
                visited.add(cur)
                walk.append(cur)
                _, head, w = dart_endpoints(graph, cur)
                wrap_acc = (wrap_acc[0] + w[0], wrap_acc[1] + w[1])

                rev = reverse_dart(cur)
                if (head, rev) not in pos_in_rotation:
                    raise RuntimeError(
                        f"Reverse dart {rev} not found at vertex {head}; "
                        f"rotation system is inconsistent."
                    )
                idx = pos_in_rotation[(head, rev)]
                rot = rotation[head]
                nxt = rot[(idx + 1) % len(rot)]

                if nxt == start:
                    walks.append(walk)
                    wraps.append(wrap_acc)
                    break
                cur = nxt
    return walks, wraps


def face_sizes(graph):
    """Return a list of face lengths (one per face)."""
    return [len(walk) for walk in trace_faces(graph)]


def family_label(graph) -> str:
    """
    Canonical face-set family label: sorted distinct face sizes joined
    by '_'. E.g. {3, 6} -> "3_6", {3, 4, 6, 8} -> "3_4_6_8".
    """
    sizes = sorted(set(face_sizes(graph)))
    return "_".join(str(s) for s in sizes)


# ---------------------------------------------------------------------------
# Traditional names
# ---------------------------------------------------------------------------

def assign_traditional_name(ground_name: str, old_family: str):
    """
    Heuristic: any old_family that is NOT a face-set code (purely digits
    separated by underscores) is treated as a traditional name and
    returned. Otherwise None.

    Examples:
        assign_traditional_name("kat",        "cloth")          -> "cloth"
        assign_traditional_name("2x4_8",      "3_6")            -> None
        assign_traditional_name("V9_3x3_001", "taylor_bobbin")  -> "taylor_bobbin"

    NOTE on `taylor_bobbin`: that's provenance, not a folk name. If
    you don't want it propagated as `traditional_name`, filter it out
    in the integration script.
    """
    if not old_family:
        return None
    parts = old_family.split("_")
    if all(p.isdigit() for p in parts):
        return None
    return old_family


# ---------------------------------------------------------------------------
# Self-tests (use TesseLace y-down convention)
# ---------------------------------------------------------------------------

def _verify_degrees(graph):
    """Return undirected degree per vertex (each arc contributes 1 to src
    AND 1 to dst, since arcs are directed half-edges of an undirected edge)."""
    deg = defaultdict(int)
    for e in graph.edges:
        deg[e.src_idx] += 1
        deg[e.dst_idx] += 1
    return dict(deg)


def _make_trihex():
    """
    Trihexagonal (kagome, 3.6.3.6) tiling, expressed using TesseLace's
    integer-lattice y-down convention. Expect family '3_6'.

    We choose a 2-row, 4-col integer lattice (n_rows=2, n_cols=4) with 3
    vertices placed so each vertex has degree 4 and the faces are 2 triangles
    plus 1 hexagon. Edges are directed arcs (2 in + 2 out per vertex).

    Vertex layout (y down, integer coords):
        v0 = (0, 0)
        v1 = (2, 0)
        v2 = (1, 1)

    Arc list (each arc is one of the 2 outgoing arcs from its src):
        v0 -> v1 wrap (0,0)        [along bottom of triangle]
        v0 -> v2 wrap (0,0)        [up to apex]
        v1 -> v2 wrap (0,0)        [up to apex]
        v1 -> v0 wrap (1,0)        [horizontal wrap]
        v2 -> v0 wrap (0,1)        [down-wrap to v0 in row below]
        v2 -> v1 wrap (0,1)        [down-wrap to v1 in row below]

    Each vertex has 2 outgoing arcs. Now check incoming:
        v0 incoming: from v1 wrap(1,0), from v2 wrap(0,1) — 2 ✓
        v1 incoming: from v0 wrap(0,0), from v2 wrap(0,1) — 2 ✓
        v2 incoming: from v0 wrap(0,0), from v1 wrap(0,0) — 2 ✓
    Total degrees (undirected): all 4 ✓
    Faces by Euler on torus: V-E+F=0 → 3-6+F=0 → F=3 (2 triangles + 1 hexagon).
    """
    vertices = [(0, 0), (2, 0), (1, 1)]
    n_rows, n_cols = 2, 4

    def poly(src, dst, wrap):
        src_pos = vertices[src]
        dst_pos_ext = (vertices[dst][0] + wrap[0] * n_cols,
                       vertices[dst][1] + wrap[1] * n_rows)
        return (src_pos, dst_pos_ext)

    arcs = [
        (0, 1, (0, 0)),
        (0, 2, (0, 0)),
        (1, 2, (0, 0)),
        (1, 0, (1, 0)),
        (2, 0, (0, 1)),
        (2, 1, (0, 1)),
    ]
    edges = [Edge(s, d, w, poly(s, d, w)) for s, d, w in arcs]
    return LaceGraph(vertices=vertices, edges=edges,
                     n_rows=n_rows, n_cols=n_cols)


def _make_square_single():
    """Single-vertex square tiling 4.4.4.4. Expect family '4'.

    Vertex 0 at (0,0). Two outgoing arcs: 0->0 wrap (1,0) and 0->0 wrap (0,1).
    Each contributes 1 incoming arc to vertex 0 from itself, total degree 4.
    """
    vertices = [(0, 0)]
    n_rows, n_cols = 1, 1
    def poly(src, dst, wrap):
        src_pos = vertices[src]
        dst_pos_ext = (vertices[dst][0] + wrap[0] * n_cols,
                       vertices[dst][1] + wrap[1] * n_rows)
        return (src_pos, dst_pos_ext)
    arcs = [(0, 0, (1, 0)), (0, 0, (0, 1))]
    edges = [Edge(s, d, w, poly(s, d, w)) for s, d, w in arcs]
    return LaceGraph(vertices=vertices, edges=edges,
                     n_rows=n_rows, n_cols=n_cols)


def _make_square_2x2():
    """2x2 square grid. Expect family '4'.

    Layout (y down):
        v0=(0,0)  v1=(1,0)
        v2=(0,1)  v3=(1,1)

    8 arcs (2 out per vertex):
        v0 -> v1 (0,0), v0 -> v2 (0,0)
        v1 -> v0 (1,0), v1 -> v3 (0,0)
        v2 -> v3 (0,0), v2 -> v0 (0,1)
        v3 -> v2 (1,0), v3 -> v1 (0,1)
    """
    vertices = [(0, 0), (1, 0), (0, 1), (1, 1)]
    n_rows, n_cols = 2, 2
    def poly(src, dst, wrap):
        src_pos = vertices[src]
        dst_pos_ext = (vertices[dst][0] + wrap[0] * n_cols,
                       vertices[dst][1] + wrap[1] * n_rows)
        return (src_pos, dst_pos_ext)
    arcs = [
        (0, 1, (0, 0)), (0, 2, (0, 0)),
        (1, 0, (1, 0)), (1, 3, (0, 0)),
        (2, 3, (0, 0)), (2, 0, (0, 1)),
        (3, 2, (1, 0)), (3, 1, (0, 1)),
    ]
    edges = [Edge(s, d, w, poly(s, d, w)) for s, d, w in arcs]
    return LaceGraph(vertices=vertices, edges=edges,
                     n_rows=n_rows, n_cols=n_cols)


if __name__ == "__main__":
    cases = [
        ("Trihexagonal (kagome) [y-down]", _make_trihex(), "3_6"),
        ("Square tiling, 1-vertex cell",   _make_square_single(), "4"),
        ("Square tiling, 2x2 cell",        _make_square_2x2(), "4"),
    ]
    all_pass = True
    for name, g, expected in cases:
        try:
            got = family_label(g)
            sizes = sorted(face_sizes(g))
        except Exception as exc:
            got = f"<error: {type(exc).__name__}: {exc}>"
            sizes = []
        ok = got == expected
        all_pass &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: family={got} (expected {expected})")
        V = len(g.vertices)
        E = len(g.edges)
        F = len(sizes)
        chi = V - E + F
        print(f"          V={V}, E={E}, F={F}, V-E+F={chi}  (should be 0)")
        print(f"          face sizes: {sizes}")

    print("\nTraditional-name heuristic:")
    cases2 = [
        ("kat", "cloth", "cloth"),
        ("2x4_8", "3_6", None),
        ("R3M3_6x6_1", "3_6", None),
        ("V9_3x3_001", "taylor_bobbin", "taylor_bobbin"),
        ("foo", "3_4_6_8", None),
        ("foo", "", None),
    ]
    for nm, old, expected in cases2:
        got = assign_traditional_name(nm, old)
        ok = got == expected
        all_pass &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  ({nm!r}, {old!r}) -> {got!r}  (expected {expected!r})")

    print("\n" + ("All tests passed." if all_pass else "FAILURES."))
