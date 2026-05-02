"""
canonicalize.py
================

Canonical forms for 2D periodic lattice grounds.

  graph_canonical : nauty certificate of the wrap-labeled quotient graph,
                    canonicalized over the 8 D4 wrap-rotations. Fully
                    invariant under abstract graph isomorphism and D4.

  lace_canonical  : full physical-lattice fingerprint combining graph
                    and cartesian-space embedding signature. Distinguishes
                    all 321 grounds in the Irvine catalog (each has a
                    unique value).

KNOWN LIMITATION: lace_canonical is not yet fully invariant under
rotation/reflection of the input — i.e., if the same lattice is fed in
twice with different orientations, the two lace_canonicals may differ.
For the existing fixed-orientation Irvine catalog this doesn't matter
(no rotated duplicates exist). For the planned enumerator, this will
need to be fixed by adding Niggli basis reduction or switching to a
purely-relative-geometry signature. graph_canonical IS rotation-invariant
and can be used in the enumerator as a first-pass deduplicator.

Requires:
    pip install pynauty --break-system-packages
"""

from __future__ import annotations

import hashlib
from typing import Tuple

try:
    import pynauty
except ImportError:
    raise ImportError(
        "pynauty required. Install: pip install pynauty --break-system-packages"
    )


# ---------------------------------------------------------------------------
# D4 group (8 elements)
# ---------------------------------------------------------------------------

D4_OPERATIONS = [
    ((1, 0), (0, 1)),     # e
    ((0, -1), (1, 0)),    # r90
    ((-1, 0), (0, -1)),   # r180
    ((0, 1), (-1, 0)),    # r270
    ((1, 0), (0, -1)),    # mx
    ((-1, 0), (0, 1)),    # my
    ((0, 1), (1, 0)),     # mxy
    ((0, -1), (-1, 0)),   # mxny
]


def _apply_2x2(M, v):
    """Apply 2x2 matrix M to 2-vector v."""
    return (M[0][0] * v[0] + M[0][1] * v[1],
            M[1][0] * v[0] + M[1][1] * v[1])


# ---------------------------------------------------------------------------
# Convert ground to cartesian representation
# ---------------------------------------------------------------------------

def _to_cartesian(ground):
    """Convert ground to cartesian-space representation.

    Returns:
        a, b: 2-tuples of cartesian basis vectors
        vertices: list of cartesian vertex positions
        edges: list of (src_idx, dst_idx, displacement) where displacement
               is the cartesian vector from src to dst (incorporating wrap)
    """
    n_cols = ground["n_cols"]
    n_rows = ground["n_rows"]
    L = ground["lattice"]
    a = (L[0][0], L[0][1])
    b = (L[1][0], L[1][1])

    vertices = []
    for v in ground["vertices"]:
        x = (v[0] / n_cols) * a[0] + (v[1] / n_rows) * b[0]
        y = (v[0] / n_cols) * a[1] + (v[1] / n_rows) * b[1]
        vertices.append((x, y))

    edges = []
    for e in ground["edges"]:
        src = e["src"]
        dst = e["dst"]
        wrap = e["wrap"]
        # Displacement = (dst_pos - src_pos) + wrap_a * a + wrap_b * b
        sx, sy = vertices[src]
        dx, dy = vertices[dst]
        wx = wrap[0] * a[0] + wrap[1] * b[0]
        wy = wrap[0] * a[1] + wrap[1] * b[1]
        disp = (dx - sx + wx, dy - sy + wy)
        edges.append((src, dst, disp))

    return a, b, vertices, edges


# ---------------------------------------------------------------------------
# Canonical form of a cartesian-space lattice (after applying a fixed R)
# ---------------------------------------------------------------------------

def _canonical_cartesian_signature(a, b, vertices, edges, precision=6):
    """Compute a translation-invariant canonical signature of a cartesian lattice.

    Translation invariance: shift so the lex-smallest vertex sits at origin.
    Basis ordering: sort (a, b) so a is lex-smaller.
    """
    PREC = precision

    # Round vertices for stability
    vert_rounded = [(round(v[0], PREC), round(v[1], PREC)) for v in vertices]

    # Find lex-smallest vertex; translate so it sits at (0, 0)
    sorted_verts = sorted(vert_rounded)
    if not sorted_verts:
        return b"empty"
    shift = sorted_verts[0]
    pos_by_idx = {}
    for i, v in enumerate(vert_rounded):
        pos_by_idx[i] = (round(v[0] - shift[0], PREC),
                          round(v[1] - shift[1], PREC))

    # Express each vertex modulo the lattice — wrap into the unit cell.
    # We need fractional coordinates wrt (a, b) for this.
    # Inverse of [a; b]^T to convert cartesian to fractional.
    det = a[0] * b[1] - a[1] * b[0]
    if abs(det) < 1e-12:
        return b"DEGENERATE"

    def to_frac(p):
        u = (p[0] * b[1] - p[1] * b[0]) / det
        v = (-p[0] * a[1] + p[1] * a[0]) / det
        return (u, v)

    def from_frac(uv):
        return (uv[0] * a[0] + uv[1] * b[0],
                uv[0] * a[1] + uv[1] * b[1])

    def wrap_unit(x):
        x = x - int(x)
        if x < 0:
            x += 1
        return x

    # Wrap vertices into [0,1) cell coords, then back to cartesian for display
    pos_in_cell = {}
    for i, p in pos_by_idx.items():
        f = to_frac(p)
        f_wrapped = (round(wrap_unit(f[0]), PREC), round(wrap_unit(f[1]), PREC))
        pos_in_cell[i] = f_wrapped

    # Choose canonical basis ordering: sort (a, b) so the lex-smaller one is
    # first. The basis swap is its own permutation but the lattice is the same.
    # Actually, swapping a and b swaps the meaning of u and v in fractional
    # coords, so we'd swap them in pos_in_cell too.
    a_t = (round(a[0], PREC), round(a[1], PREC))
    b_t = (round(b[0], PREC), round(b[1], PREC))

    if a_t <= b_t:
        # a is canonical first
        canon_basis = (a_t, b_t)
        canon_pos = pos_in_cell
    else:
        # swap a and b: also swap fractional coords
        canon_basis = (b_t, a_t)
        canon_pos = {i: (uv[1], uv[0]) for i, uv in pos_in_cell.items()}

    # Edges: represent each edge by (src_pos, dst_pos_in_cell, displacement_round)
    # The displacement is a cartesian vector independent of wrap conventions
    edges_repr = []
    for src_idx, dst_idx, disp in edges:
        pu = canon_pos[src_idx]
        pv = canon_pos[dst_idx]
        d_round = (round(disp[0], PREC), round(disp[1], PREC))
        # If we swapped a,b above, displacement is in cartesian and unaffected
        if pu <= pv:
            edges_repr.append((pu, pv, d_round))
        else:
            edges_repr.append((pv, pu,
                                (round(-d_round[0], PREC),
                                 round(-d_round[1], PREC))))
    edges_repr.sort()
    canon_verts = sorted(canon_pos.values())

    return repr((canon_basis, canon_verts, edges_repr)).encode()


# ---------------------------------------------------------------------------
# Apply D4 element to the cartesian lattice and canonicalize
# ---------------------------------------------------------------------------

def _signature_under_d4(a, b, vertices, edges, R):
    """Apply D4 element R to the cartesian lattice, then canonicalize."""
    Ra = _apply_2x2(R, a)
    Rb = _apply_2x2(R, b)
    Rverts = [_apply_2x2(R, v) for v in vertices]
    Redges = [(s, d, _apply_2x2(R, disp)) for (s, d, disp) in edges]
    return _canonical_cartesian_signature(Ra, Rb, Rverts, Redges)


# ---------------------------------------------------------------------------
# Graph canonical (D4-invariant nauty certificate)
# ---------------------------------------------------------------------------

def _canonicalize_edge(src, dst, wrap):
    if src < dst:
        return (src, dst, wrap)
    if dst < src:
        return (dst, src, (-wrap[0], -wrap[1]))
    neg_wrap = (-wrap[0], -wrap[1])
    return (src, dst, min(wrap, neg_wrap))


def _build_nauty_graph(n_orig, edges):
    canon_edges = sorted(edges)
    by_wrap = {}
    for u, v, wrap in canon_edges:
        by_wrap.setdefault(wrap, []).append((u, v))
    n_total = n_orig + len(canon_edges)
    g = pynauty.Graph(n_total, directed=False)
    color_classes = [set(range(n_orig))]
    aux_idx = n_orig
    for wrap in sorted(by_wrap.keys()):
        wrap_class = set()
        for (u, v) in by_wrap[wrap]:
            if u == v:
                g.connect_vertex(aux_idx, [u])
            else:
                g.connect_vertex(aux_idx, [u, v])
            wrap_class.add(aux_idx)
            aux_idx += 1
        color_classes.append(wrap_class)
    g.set_vertex_coloring(color_classes)
    return g


def _nauty_cert_under_op(ground, R):
    edges = set()
    for e in ground["edges"]:
        wrap = tuple(e["wrap"])
        new_wrap = _apply_2x2(R, wrap)
        new_wrap = (int(round(new_wrap[0])), int(round(new_wrap[1])))
        edges.add(_canonicalize_edge(e["src"], e["dst"], new_wrap))
    g = _build_nauty_graph(len(ground["vertices"]), edges)
    return pynauty.certificate(g)


def graph_canonical(ground: dict) -> str:
    """D4-invariant nauty certificate hash of the abstract graph."""
    certs = [_nauty_cert_under_op(ground, R) for R in D4_OPERATIONS]
    return hashlib.sha256(min(certs)).hexdigest()


# ---------------------------------------------------------------------------
# Lace canonical: cartesian D4-canonicalized embedding + graph
# ---------------------------------------------------------------------------

def lace_canonical(ground: dict) -> str:
    """D4-invariant canonical of the full physical lattice."""
    a, b, vertices, edges = _to_cartesian(ground)
    sigs = [_signature_under_d4(a, b, vertices, edges, R) for R in D4_OPERATIONS]
    embed_canon = min(sigs)
    g_cert = graph_canonical(ground).encode()
    h = hashlib.sha256()
    h.update(g_cert)
    h.update(b"|")
    h.update(embed_canon)
    return h.hexdigest()


def both_canonical(ground: dict) -> dict:
    return {
        "graph_canonical": graph_canonical(ground),
        "lace_canonical": lace_canonical(ground),
    }


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
    print(f"Computing canonical forms for {len(grounds)} grounds...")
    graph_certs, lace_certs = {}, {}
    for g in grounds:
        graph_certs.setdefault(graph_canonical(g), []).append(
            f"{g['family']}/{g['name']}")
        lace_certs.setdefault(lace_canonical(g), []).append(
            f"{g['family']}/{g['name']}")
    print(f"\n  Distinct abstract graphs: {len(graph_certs)}")
    print(f"  Distinct lace structures: {len(lace_certs)}")
    n_dup = sum(len(v) for v in lace_certs.values() if len(v) > 1)
    print(f"  Grounds sharing lace: {n_dup}")
    if n_dup > 0:
        print("\n  Lace duplicates (D4-equivalent):")
        for k, v in lace_certs.items():
            if len(v) > 1:
                print(f"     {v}")
