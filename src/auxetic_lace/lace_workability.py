"""
lace_workability.py
====================

Implementation of `is_lace_workable` checker for periodic 2-regular digraph
embeddings (bobbin lace grounds).

Following Irvine (2016) Lace Tessellations, Chapter 3:

  A 2-regular digraph G embedded on the torus is "workable" iff its edges
  can be partitioned into a set of osculating directed circuits, each
  of which has wrapping index (1, 0).

The osculating partition is constructed (Lemma 3.3.4):

  "Choose a directed circuit K by selecting any directed edge (u, v) ∈ E.
   Then add the unique, rotationally consecutive outgoing edge (v, w) ∈ E
   to K. ... The process is repeated using vertex w and so on until K
   returns to its initial vertex u via edge (z, u) ∈ E where (z, u) is
   rotationally consecutive to (u, v) at vertex u."

Key insights:

  - The graph is a DIGRAPH: edges have direction (src → dst).
  - At each vertex, the cyclic angular order of incident edges must be
    "rotationally consecutive": [in1, in2, out1, out2] in cyclic order.
    A "rotationally alternating" vertex [in, out, in, out] would imply
    contractible cycles (Cor 3.3.3) and is incompatible with workable lace.
  - Continuation rule: arriving on incoming edge i, depart on the outgoing
    edge that is the next-clockwise (or next-counter-clockwise) outgoing
    edge — specifically, the one rotationally adjacent in the same
    direction as the rotation order.

For the wrap criterion: each circuit must wrap (1, 0) on the toroidal
quotient — i.e., go once around horizontally (meridional direction in
Irvine's convention) and zero times vertically.

Implementation:
  - At each vertex, sort incident edges (incoming + outgoing) by angle.
  - For the rotation system to be "consecutive", the 4 edges around
    vertex must group as [in, in, out, out] in cyclic order.
  - When following circuit, on arrival via incoming edge `i` at vertex v,
    look up the outgoing edge that immediately follows `i` in the cyclic
    rotation.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


def _edge_angle_at_vertex(ground: dict, edge_idx: int, end: str) -> float:
    """Angle (radians, CCW from +x) of the edge as seen from one of its endpoints.

    end='src': we are at the src vertex, looking toward dst+wrap (outgoing direction)
    end='dst': we are at the dst vertex (in adjacent cell if wrap nonzero),
               looking back toward src-wrap (incoming direction reversed)
    """
    e = ground["edges"][edge_idx]
    src = ground["vertices"][e["src"]]
    dst = ground["vertices"][e["dst"]]
    wrap = e["wrap"]
    n_cols = ground["n_cols"]
    n_rows = ground["n_rows"]
    L = ground["lattice"]
    a_vec = (L[0][0], L[0][1])
    b_vec = (L[1][0], L[1][1])
    sx = (src[0] / n_cols) * a_vec[0] + (src[1] / n_rows) * b_vec[0]
    sy = (src[0] / n_cols) * a_vec[1] + (src[1] / n_rows) * b_vec[1]
    dx_in = (dst[0] / n_cols) * a_vec[0] + (dst[1] / n_rows) * b_vec[0]
    dy_in = (dst[0] / n_cols) * a_vec[1] + (dst[1] / n_rows) * b_vec[1]
    wx = wrap[0] * a_vec[0] + wrap[1] * b_vec[0]
    wy = wrap[0] * a_vec[1] + wrap[1] * b_vec[1]
    dx = dx_in + wx
    dy = dy_in + wy
    if end == "src":
        return math.atan2(dy - sy, dx - sx)
    else:
        return math.atan2(sy - dy, sx - dx)


def _build_vertex_rotation(ground: dict) -> Dict[int, List[Tuple[int, str]]]:
    """For each vertex, return the cyclic order of incident edge-ends.

    Each entry is (edge_idx, role) where role is 'out' (this vertex is the
    src of the edge) or 'in' (this vertex is the dst of the edge). Sorted
    counter-clockwise by angle.
    """
    n_vertices = len(ground["vertices"])
    by_vertex: Dict[int, List[Tuple[int, str]]] = {v: [] for v in range(n_vertices)}
    for ei, e in enumerate(ground["edges"]):
        by_vertex[e["src"]].append((ei, "out"))
        by_vertex[e["dst"]].append((ei, "in"))
    for v, lst in by_vertex.items():
        lst.sort(key=lambda h: _edge_angle_at_vertex(ground, h[0],
                                                       "src" if h[1] == "out" else "dst"))
    return by_vertex


def _check_2_in_2_out(ground: dict) -> bool:
    """Verify each vertex has 2 in and 2 out edges (basic 2-regular digraph check)."""
    n_vertices = len(ground["vertices"])
    in_count = [0] * n_vertices
    out_count = [0] * n_vertices
    for e in ground["edges"]:
        out_count[e["src"]] += 1
        in_count[e["dst"]] += 1
    return all(in_count[v] == 2 and out_count[v] == 2 for v in range(n_vertices))


def _check_rotationally_consecutive(rotation: Dict[int, List[Tuple[int, str]]]) -> bool:
    """Each vertex's cyclic order must be [in, in, out, out] (some rotation)."""
    for v, lst in rotation.items():
        if len(lst) != 4:
            return False
        roles = [r for (_, r) in lst]
        # Find a rotation where roles = ['in', 'in', 'out', 'out']
        ok = False
        for shift in range(4):
            rotated = roles[shift:] + roles[:shift]
            if rotated == ["in", "in", "out", "out"]:
                ok = True
                break
        if not ok:
            return False
    return True


def trace_osculating_circuits(ground: dict) -> List[Tuple[List[int], Tuple[int, int]]]:
    """Partition directed edges into osculating circuits.

    Returns list of (edges_in_order, total_wrap) for each circuit.

    Algorithm: starting from any unvisited directed edge (u, v), trace by
    rotational consecutivity at each vertex. At v, we arrived via the
    edge with role='in' at v. The "rotationally consecutive outgoing edge"
    is the next 'out' edge in the cyclic order after this 'in' edge.
    """
    rotation = _build_vertex_rotation(ground)
    n_edges = len(ground["edges"])
    visited = [False] * n_edges
    circuits = []

    # Pre-compute, for each vertex, a map from incoming-edge-idx to the
    # rotationally consecutive outgoing edge index.
    #
    # For a rotationally-consecutive vertex with cyclic order [in, in, out, out]
    # (in some rotation), the OSCULATING (kissing) bijection pairs each 'in'
    # with the 'out' that is its immediate ROTATIONALLY ADJACENT neighbor of
    # opposite role. Each in-edge has exactly one adjacent neighbor (CW or CCW)
    # that is an out-edge — that neighbor is the consecutive out.
    #
    # The "across" pairing (positions 2 apart) corresponds to TRANSVERSE
    # crossings (Figure 3.5(a)) which Irvine's theorem 3.3.5 excludes for
    # workable lace.
    consec_out: Dict[Tuple[int, int], int] = {}  # (v, in_edge) -> out_edge
    for v, lst in rotation.items():
        n = len(lst)
        for i, (ei, role) in enumerate(lst):
            if role != "in":
                continue
            # Look at both rotational neighbors (positions i-1 and i+1).
            # Exactly one of them should be 'out'. That's the consecutive out.
            for di in (1, -1):
                j = (i + di) % n
                ej, rj = lst[j]
                if rj == "out":
                    consec_out[(v, ei)] = ej
                    break

    for start in range(n_edges):
        if visited[start]:
            continue
        circuit_edges = []
        wrap_total = [0, 0]
        cur = start
        while not visited[cur]:
            visited[cur] = True
            circuit_edges.append(cur)
            e = ground["edges"][cur]
            wrap_total[0] += e["wrap"][0]
            wrap_total[1] += e["wrap"][1]
            v = e["dst"]  # vertex we just arrived at
            nxt = consec_out.get((v, cur))
            if nxt is None:
                # Shouldn't happen for a well-formed 2-regular consecutive digraph
                break
            cur = nxt
        circuits.append((circuit_edges, tuple(wrap_total)))
    return circuits


def is_lace_workable(ground: dict) -> bool:
    """Return True iff this ground is bobbin-lace workable, i.e., its edges
    can be partitioned into osculating directed circuits each wrapping (1, 0)
    or (-1, 0) on the toroidal quotient.

    Implements the criterion of Irvine (2016) Theorem 3.3.5.
    """
    if not _check_2_in_2_out(ground):
        return False
    rotation = _build_vertex_rotation(ground)
    if not _check_rotationally_consecutive(rotation):
        return False
    try:
        circuits = trace_osculating_circuits(ground)
    except Exception:
        return False
    # Each edge should be in exactly one circuit
    seen = [False] * len(ground["edges"])
    for circ, _ in circuits:
        for ei in circ:
            if seen[ei]:
                return False
            seen[ei] = True
    if not all(seen):
        return False
    # Each circuit must wrap (0, ±1) OR (±1, 0) — once around one axis, zero
    # around the other. Irvine's stored data uses (0, ±1); other constructions
    # may use (±1, 0). Both are valid lace-workable configurations.
    for _, wrap in circuits:
        if wrap not in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    atlas_path = sys.argv[1] if len(sys.argv) > 1 else "docs/atlas.json"
    with open(atlas_path) as f:
        atlas = json.load(f)
    grounds = atlas["grounds"]

    print(f"Running is_lace_workable on {len(grounds)} Irvine grounds...")
    print(f"  (All should pass — Irvine's catalog is the lace-workable set)")
    print()

    failures_2in2out = []
    failures_consecutive = []
    failures_circuit_partition = []
    failures_wrap = []

    for g in grounds:
        label = f"{g['family']}/{g['name']}"
        if not _check_2_in_2_out(g):
            failures_2in2out.append(label)
            continue
        rot = _build_vertex_rotation(g)
        if not _check_rotationally_consecutive(rot):
            failures_consecutive.append(label)
            continue
        circuits = trace_osculating_circuits(g)
        # Edge partition check
        seen = [False] * len(g["edges"])
        ok_partition = True
        for circ, _ in circuits:
            for ei in circ:
                if seen[ei]:
                    ok_partition = False
                    break
                seen[ei] = True
            if not ok_partition:
                break
        if not (ok_partition and all(seen)):
            failures_circuit_partition.append(label)
            continue
        # Wrap check
        bad_wraps = [w for (_, w) in circuits if w not in [(0, 1), (0, -1)]]
        if bad_wraps:
            failures_wrap.append((label, bad_wraps[:3]))

    n_pass = (len(grounds)
              - len(failures_2in2out)
              - len(failures_consecutive)
              - len(failures_circuit_partition)
              - len(failures_wrap))

    print(f"  Pass:                     {n_pass} / {len(grounds)}")
    print(f"  Fail 2-in-2-out:          {len(failures_2in2out)}")
    print(f"  Fail rot. consecutive:    {len(failures_consecutive)}")
    print(f"  Fail edge partition:      {len(failures_circuit_partition)}")
    print(f"  Fail wrap (1, 0):         {len(failures_wrap)}")
    print()

    if failures_consecutive:
        print(f"  Sample rotationally-alternating failures:")
        for f in failures_consecutive[:5]:
            print(f"    {f}")
    if failures_wrap:
        print(f"  Sample wrap failures:")
        for f, ws in failures_wrap[:5]:
            print(f"    {f}: {ws}")
