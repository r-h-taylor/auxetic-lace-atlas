"""
planarity.py
==============

Real planarity checker for periodic 4-regular graphs. Tests whether
the straight-line embedding (using the polyline data and wrap info)
has any edge crossings in the universal cover except at common
vertex endpoints.

For Irvine's catalog this is checked once and serves as a sanity
check (all should be planar by construction). For enumerator output
it's the principal filter that distinguishes "physically realizable
as lace" from "valid as an abstract graph but not embeddable without
crossings."

Algorithm:

  1. For each edge, expand its polyline into 2D coordinates including
     periodic wrap. An edge from src to (dst + wrap) becomes a
     sequence of straight line segments through the polyline points.
     Polyline points are in "extended" coordinates so they already
     account for wrap.

  2. For every pair of edges (i, j), test if any segment of edge i
     crosses any segment of edge j at a point that is not a shared
     vertex endpoint.

  3. Endpoint-on-segment cases (T-intersection): tested separately;
     a vertex of edge i lying on the interior of edge j's segment is
     also a crossing (would be a degree-5+ vertex in the planar
     drawing).

Returns True iff no crossings found.
"""

from __future__ import annotations
import math
from typing import List, Tuple


# Tolerance for floating-point comparisons.
EPS = 1e-9


# ---------------------------------------------------------------------------
# Segment geometry
# ---------------------------------------------------------------------------

def _orient(a, b, c) -> float:
    """Sign of the (signed) area of triangle abc.
    Positive if counter-clockwise, negative if clockwise, zero if colinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    """True iff p lies on segment a-b (assuming colinear), inclusive of endpoints."""
    return (min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS)


def _segments_intersect_interior(a1, a2, b1, b2,
                                    endpoint_set: set) -> bool:
    """Test if segments a1-a2 and b1-b2 cross at a non-endpoint interior point.

    `endpoint_set` is a set of points (rounded tuples) that are considered
    legitimate shared endpoints — intersections AT these points don't count.

    Returns True iff there is a crossing strictly in the interior of at
    least one of the segments at a point not in endpoint_set.
    """
    o1 = _orient(a1, a2, b1)
    o2 = _orient(a1, a2, b2)
    o3 = _orient(b1, b2, a1)
    o4 = _orient(b1, b2, a2)

    # General case: segments straddle each other
    if ((o1 > EPS and o2 < -EPS) or (o1 < -EPS and o2 > EPS)) \
        and ((o3 > EPS and o4 < -EPS) or (o3 < -EPS and o4 > EPS)):
        # They cross at exactly one interior point. Compute it.
        # Parametric: a1 + t*(a2-a1) = b1 + u*(b2-b1)
        denom = (a2[0] - a1[0]) * (b2[1] - b1[1]) \
              - (a2[1] - a1[1]) * (b2[0] - b1[0])
        if abs(denom) < EPS:
            return False  # parallel; degenerate
        t = ((b1[0] - a1[0]) * (b2[1] - b1[1])
             - (b1[1] - a1[1]) * (b2[0] - b1[0])) / denom
        ix = a1[0] + t * (a2[0] - a1[0])
        iy = a1[1] + t * (a2[1] - a1[1])
        # Round to grid coordinates for endpoint set comparison
        key = (_round(ix), _round(iy))
        if key in endpoint_set:
            return False  # crossing happens to be at a shared vertex, OK
        return True

    # Colinear/touching cases: check if any endpoint of one segment lies on
    # the interior of the other. T-intersections.
    if abs(o1) < EPS and _on_segment(a1, a2, b1):
        key = (_round(b1[0]), _round(b1[1]))
        # If b1 is interior to a1-a2 (not at a1 or a2), it's a T-intersection
        if not _is_close(b1, a1) and not _is_close(b1, a2):
            return True
    if abs(o2) < EPS and _on_segment(a1, a2, b2):
        if not _is_close(b2, a1) and not _is_close(b2, a2):
            return True
    if abs(o3) < EPS and _on_segment(b1, b2, a1):
        if not _is_close(a1, b1) and not _is_close(a1, b2):
            return True
    if abs(o4) < EPS and _on_segment(b1, b2, a2):
        if not _is_close(a2, b1) and not _is_close(a2, b2):
            return True

    return False


def _is_close(p, q):
    return abs(p[0] - q[0]) < EPS and abs(p[1] - q[1]) < EPS


def _round(x):
    """Round to a grid-friendly precision for hashable equality."""
    return round(x, 6)


# ---------------------------------------------------------------------------
# Edge geometry from the periodic graph
# ---------------------------------------------------------------------------

def _edge_segments(ground: dict, edge_idx: int
                    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return the geometric segments composing an edge.

    For an edge stored with src, dst, wrap, polyline:
      - The polyline points are in "extended" coordinates already
        accounting for wrap (Irvine convention).
      - For atlases lacking polyline data (e.g., minimally-built records),
        we synthesize a straight segment from src.coord to dst.coord+wrap.

    Returns list of segments [(p1, p2), ...].
    """
    e = ground["edges"][edge_idx]
    n_cols = ground["n_cols"]
    n_rows = ground["n_rows"]
    src_v = ground["vertices"][e["src"]]
    dst_v = ground["vertices"][e["dst"]]
    wrap = e["wrap"]
    # Coordinates of src and (dst + wrap) in extended (un-wrapped) coords.
    src = (float(src_v[0]), float(src_v[1]))
    dst_wrap = (float(dst_v[0]) + wrap[0] * n_cols,
                 float(dst_v[1]) + wrap[1] * n_rows)
    poly = e.get("polyline")
    if not poly or len(poly) < 2:
        # Synthesize straight segment
        return [(src, dst_wrap)]
    # Use polyline points as given. We trust they're in extended coordinates.
    pts = [(float(p[0]), float(p[1])) for p in poly]
    segs = []
    for i in range(len(pts) - 1):
        segs.append((pts[i], pts[i + 1]))
    return segs


def _vertex_endpoint_set(ground: dict) -> set:
    """All vertex coordinates including periodic copies that any edge visits.

    For each edge we record the rounded src and (dst + wrap) coordinates.
    These are the legitimate "endpoints" — intersections at any of these
    are vertex-touchings, not crossings.
    """
    n_cols = ground["n_cols"]
    n_rows = ground["n_rows"]
    pts = set()
    for e in ground["edges"]:
        sv = ground["vertices"][e["src"]]
        dv = ground["vertices"][e["dst"]]
        w = e["wrap"]
        src_pt = (_round(float(sv[0])), _round(float(sv[1])))
        dst_pt = (_round(float(dv[0]) + w[0] * n_cols),
                   _round(float(dv[1]) + w[1] * n_rows))
        pts.add(src_pt)
        pts.add(dst_pt)
    return pts


# ---------------------------------------------------------------------------
# The planarity checker
# ---------------------------------------------------------------------------

def is_planar(ground: dict) -> bool:
    """Return True iff the straight-line (or polyline) embedding of the
    periodic graph has no edge crossings except at shared vertex endpoints.

    Tests the embedding as drawn in the universal cover using the polyline
    data and wrap info. This is the right notion of planarity for a
    periodic graph drawn in the infinite plane: no rod crosses another
    rod except at a common pin.
    """
    n_edges = len(ground.get("edges", []))
    if n_edges < 2:
        return True

    # Pre-compute segments and endpoint set
    segs_per_edge = [_edge_segments(ground, i) for i in range(n_edges)]
    vertex_pts = _vertex_endpoint_set(ground)

    # Test all pairs
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            for s1 in segs_per_edge[i]:
                for s2 in segs_per_edge[j]:
                    if _segments_intersect_interior(s1[0], s1[1], s2[0], s2[1],
                                                     vertex_pts):
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

    print(f"Running is_planar on {len(grounds)} grounds...")
    print()
    print("NOTE: This straight-line checker is the right test for taylor")
    print("grounds (synthesized straight polylines). For Irvine grounds it")
    print("over-reports non-planarity because Irvine uses curved polylines")
    print("we don't have access to in atlas.json. Treat Irvine results as")
    print("diagnostic only.")
    print()

    n_irvine_pass = 0
    n_irvine_fail = 0
    n_taylor_pass = 0
    n_taylor_fail = 0
    irvine_passes = []
    taylor_passes = []

    for g in grounds:
        ok = is_planar(g)
        is_irvine = (g.get("provenance", {}).get("source") == "irvine")
        is_taylor = (str(g.get("provenance", {}).get("source", ""))
                       .startswith("taylor_"))
        if is_irvine:
            if ok:
                n_irvine_pass += 1
                irvine_passes.append(f"{g['family']}/{g['name']}")
            else:
                n_irvine_fail += 1
        elif is_taylor:
            if ok:
                n_taylor_pass += 1
                taylor_passes.append(f"{g['family']}/{g['name']}")
            else:
                n_taylor_fail += 1

    n_irvine = n_irvine_pass + n_irvine_fail
    n_taylor = n_taylor_pass + n_taylor_fail
    print(f"  Irvine (straight-line check, diagnostic):")
    print(f"     {n_irvine_pass} pass / {n_irvine} total "
          f"({100 * n_irvine_pass / max(n_irvine, 1):.1f}%)")
    print(f"  Taylor (real check):")
    print(f"     {n_taylor_pass} pass / {n_taylor} total "
          f"({100 * n_taylor_pass / max(n_taylor, 1):.1f}%)")
    print()
    if taylor_passes:
        print(f"  Taylor planar grounds ({len(taylor_passes)}):")
        for f in taylor_passes[:30]:
            print(f"    {f}")
        if len(taylor_passes) > 30:
            print(f"    ... and {len(taylor_passes) - 30} more")
    if irvine_passes and len(irvine_passes) < 50:
        print(f"\n  Irvine planar by straight-line drawing ({len(irvine_passes)}):")
        for f in irvine_passes[:30]:
            print(f"    {f}")
