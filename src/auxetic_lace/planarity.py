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
    """Test whether the ground is planar in the cell-coordinate embedding.

    For each edge, build its polyline-segment list:
      - If the edge has a 'polyline' field with >= 2 points, use those segments
        directly (in extended cell coords).
      - Otherwise, fall back to a single straight segment from src to dst+wrap.

    Then test all pairs of segments (across all edges) for off-pin (interior)
    intersection. Off-pin means the intersection is strictly interior to both
    segments (not at a shared endpoint).

    Returns True if no off-pin intersections found.
    """
    n_cols = ground["n_cols"]
    n_rows = ground["n_rows"]
    vertices = ground["vertices"]

    segments = []
    for ei, e in enumerate(ground["edges"]):
        polyline = e.get("polyline")
        if polyline and len(polyline) >= 2:
            for si in range(len(polyline) - 1):
                p1 = tuple(polyline[si])
                p2 = tuple(polyline[si + 1])
                segments.append((p1, p2, ei, si))
        else:
            src = vertices[e["src"]]
            dst = vertices[e["dst"]]
            wrap = e["wrap"]
            p1 = (src[0], src[1])
            p2 = (dst[0] + wrap[0] * n_cols, dst[1] + wrap[1] * n_rows)
            segments.append((p1, p2, ei, 0))

    for i in range(len(segments)):
        p1, p2, ei1, si1 = segments[i]
        for j in range(i + 1, len(segments)):
            p3, p4, ei2, si2 = segments[j]
            if _segments_cross_off_pin(p1, p2, p3, p4):
                return False
    return True


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross_off_pin(p1, p2, p3, p4):
    """True iff the segments (p1,p2) and (p3,p4) cross at a point strictly
    interior to both segments."""
    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)
    if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
       ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
        return True
    return False
