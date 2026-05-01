"""
extract_features.py
===================

Compute structural and mechanical descriptors for every ground in atlas.json
and write them to features.csv (one row per ground, one column per feature).

Three feature groups:

  Tier 1 — face profile (counts of triangles, squares, ..., decagons; mean
  face size; n_distinct_face_sizes).

  Tier 2 — geometry of the toroidal embedding (mean / variance of edge
  length; mean / variance of vertex interior angles; n_reentrant vertices,
  defined as vertices with at least one interior angle > pi).

  Tier 3 — derived mechanical descriptors from C_voigt at the default beam
  AR (planar bulk modulus K, planar shear modulus G, K/G ratio, anisotropy
  index measured as nu_max - nu_min).

Plus the labels we already have: nu_min and nu_max in the spring and beam
defaults, classification.

The hard part is face-walking on the torus. Algorithm:

  1. Build half-edges. Each Edge(src, dst, wrap) becomes two half-edges:
       outgoing: from src, direction = vertices[dst]+wrap*lattice - vertices[src]
       incoming partner: from dst (in image -wrap), the reverse direction.
     Each half-edge is (origin_vertex_idx, direction_vec, twin_index).

  2. Rotation system. For each vertex v, list all incident half-edges (the
     ones whose origin is v). Sort them by angle. This is the cyclic
     order around v.

  3. Face walk. For a starting half-edge h, the face on its LEFT is found by
       next(h) = rotation_at(h.dst).cw_neighbor(twin(h))
     i.e., arrive along twin(h), step CW one slot in the rotation at the
     destination, take that as the next half-edge. Walk until we return to h.
     Each unvisited half-edge starts a new face.

  Conventions: TesseLace uses y-down (column, row). For angle calculations
  we treat (col, row) as a planar (x, y) embedding; choice of CW vs CCW
  doesn't matter as long as we're consistent.

  Sanity check: V - E + F = 0 on a torus, and sum of face sizes = 2E.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def edge_direction(g: Dict, e: Dict) -> Tuple[float, float]:
    """Real-space direction vector of an edge in the toroidal embedding,
    using the lattice basis. Returns (dx, dy).

    g['vertices'] gives in-period (col, row) integer coords.
    g['lattice'] gives the 2x2 basis [[a1x, a1y], [a2x, a2y]] where
    a1 wraps n_cols and a2 wraps n_rows. (For all 321 grounds in the
    atlas these are axis-aligned identity-scaled, so col->x, row->y, but
    we honor the basis just in case.)
    """
    src = g["vertices"][e["src"]]
    dst = g["vertices"][e["dst"]]
    wrap = e["wrap"]
    a1, a2 = g["lattice"][0], g["lattice"][1]
    # vertex coords are integer (col, row); the lattice maps col->x_axis, row->y_axis
    # so a1 contributes per-column displacement and a2 per-row.
    # Source and destination in cartesian:
    #   pos(v) = v[0] * (a1/n_cols) + v[1] * (a2/n_rows)
    # but more simply: each integer step in col is a1/n_cols, etc.
    # However the "wrap" is in units of full periods, so wrap[0] adds a1, wrap[1] adds a2.
    n_cols = g["n_cols"]
    n_rows = g["n_rows"]
    sx = src[0] * a1[0] / n_cols + src[1] * a2[0] / n_rows
    sy = src[0] * a1[1] / n_cols + src[1] * a2[1] / n_rows
    dx_in = dst[0] * a1[0] / n_cols + dst[1] * a2[0] / n_rows
    dy_in = dst[0] * a1[1] / n_cols + dst[1] * a2[1] / n_rows
    dx = dx_in + wrap[0] * a1[0] + wrap[1] * a2[0] - sx
    dy = dy_in + wrap[0] * a1[1] + wrap[1] * a2[1] - sy
    return dx, dy


# ----------------------------------------------------------------------------
# Half-edge construction and face walking
# ----------------------------------------------------------------------------

def build_half_edges(g: Dict) -> Tuple[List[Dict], List[List[int]]]:
    """Returns (half_edges, rotation) where:

      half_edges[h] = {
        'origin': vertex idx where this half-edge starts,
        'dst':    vertex idx where it ends (in some periodic image),
        'dx', 'dy': cartesian direction vector,
        'angle':  atan2(dy, dx),
        'twin':   index of the partner half-edge (same edge, opposite direction)
      }

      rotation[v] = list of half-edge indices originating at v, sorted CCW
                    by angle.
    """
    half_edges: List[Dict] = []
    for e in g["edges"]:
        dx, dy = edge_direction(g, e)
        ang = math.atan2(dy, dx)
        h_fwd_idx = len(half_edges)
        half_edges.append({
            "origin": e["src"], "dst": e["dst"],
            "dx": dx, "dy": dy, "angle": ang, "twin": h_fwd_idx + 1,
        })
        # Twin: starts at the destination (in its periodic image, but the
        # rotation system at a vertex doesn't care about which image — we
        # rotate around the in-period vertex), with the reverse direction.
        h_rev_idx = len(half_edges)
        half_edges.append({
            "origin": e["dst"], "dst": e["src"],
            "dx": -dx, "dy": -dy, "angle": math.atan2(-dy, -dx), "twin": h_fwd_idx,
        })

    n_v = len(g["vertices"])
    rotation: List[List[int]] = [[] for _ in range(n_v)]
    for h_idx, h in enumerate(half_edges):
        rotation[h["origin"]].append(h_idx)
    for v in range(n_v):
        rotation[v].sort(key=lambda h_idx: half_edges[h_idx]["angle"])

    return half_edges, rotation


def walk_faces(half_edges: List[Dict],
               rotation: List[List[int]]) -> List[List[int]]:
    """Walk all faces. Returns list of faces, each face being the list of
    half-edge indices traversed in order.

    Rule: from half-edge h with origin u and dst v, the next half-edge in
    the face on h's LEFT is the half-edge at v whose angle is the angle of
    twin(h) rotated CW by one slot in v's rotation. (Equivalently: in the
    CCW-sorted rotation at v, take the index *before* twin(h).)
    """
    n_h = len(half_edges)
    visited = [False] * n_h

    # Build "position in rotation" lookup: rot_pos[h] = index of h in
    # rotation[half_edges[h]['origin']]
    rot_pos = [0] * n_h
    for v, rot in enumerate(rotation):
        for i, h_idx in enumerate(rot):
            rot_pos[h_idx] = i

    faces: List[List[int]] = []
    for start in range(n_h):
        if visited[start]:
            continue
        face = []
        h = start
        # Loop with a hard cap to prevent runaway in case of bugs
        for _ in range(2 * n_h + 1):
            if visited[h]:
                break
            visited[h] = True
            face.append(h)
            twin = half_edges[h]["twin"]
            v = half_edges[twin]["origin"]   # = half_edges[h]['dst']
            rot = rotation[v]
            i = rot_pos[twin]
            # CW step = previous in the CCW-sorted list
            next_h = rot[(i - 1) % len(rot)]
            h = next_h
            if h == start:
                break
        else:
            raise RuntimeError("face walk did not terminate (graph malformed?)")
        faces.append(face)

    return faces


def face_size_distribution(faces: List[List[int]]) -> Counter:
    """Counter from face-size (n) to count of n-gon faces."""
    return Counter(len(f) for f in faces)


# ----------------------------------------------------------------------------
# Vertex angles, edge lengths
# ----------------------------------------------------------------------------

def vertex_interior_angles(half_edges: List[Dict],
                           rotation: List[List[int]]) -> List[List[float]]:
    """For each vertex, list the angles between successive incident
    half-edges in the rotation. These are the interior angles of the
    faces meeting at that vertex (sum to 2*pi).
    """
    angles_per_vertex = []
    for v, rot in enumerate(rotation):
        if not rot:
            angles_per_vertex.append([])
            continue
        angs = [half_edges[h]["angle"] for h in rot]
        gaps = []
        for i in range(len(angs)):
            a1 = angs[i]
            a2 = angs[(i + 1) % len(angs)]
            d = a2 - a1
            if d <= 0:
                d += 2 * math.pi
            gaps.append(d)
        angles_per_vertex.append(gaps)
    return angles_per_vertex


def edge_lengths(g: Dict) -> List[float]:
    return [math.hypot(*edge_direction(g, e)) for e in g["edges"]]


# ----------------------------------------------------------------------------
# Mechanical descriptors from C_voigt
# ----------------------------------------------------------------------------

def planar_K_G(C_voigt: List[List[float]]) -> Tuple[float, float]:
    """Compute planar bulk modulus K and planar shear modulus G from the
    plane-stress 3x3 stiffness in Voigt notation. We use the orientation-
    averaged (isotropic-projection) values, which match Soyarslan et al.

      K = (C11 + C22 + 2*C12) / 4
      G = (C11 + C22 - 2*C12 + 4*C33) / 8

    For elastically isotropic 2D materials these reduce to the usual
    K = C11 - G, G = C33 identities. For anisotropic ones they are the
    Voigt-averaged moduli.

    Returns (K, G) or (nan, nan) if C is singular / invalid.
    """
    try:
        C = np.asarray(C_voigt, dtype=float)
        if C.shape != (3, 3) or not np.all(np.isfinite(C)):
            return float("nan"), float("nan")
        C11, C22, C12, C33 = C[0, 0], C[1, 1], C[0, 1], C[2, 2]
        K = (C11 + C22 + 2 * C12) / 4.0
        G = (C11 + C22 - 2 * C12 + 4 * C33) / 8.0
        return float(K), float(G)
    except Exception:
        return float("nan"), float("nan")


# ----------------------------------------------------------------------------
# Per-ground feature extraction
# ----------------------------------------------------------------------------

# Face-size buckets we'll record explicitly. Faces larger than 12 are very
# rare in this catalog; we lump them into "n_faces_ge13".
FACE_SIZE_BUCKETS = list(range(3, 13))   # 3..12


def extract_features(g: Dict, atlas_meta: Dict) -> Dict:
    name = g["name"]
    family = g["family"]
    n_v = g["n_vertices"]
    n_e = g["n_edges"]

    half_edges, rotation = build_half_edges(g)
    faces = walk_faces(half_edges, rotation)
    n_f = len(faces)

    # Sanity invariants
    euler = n_v - n_e + n_f                       # 0 on torus
    sum_face_sizes = sum(len(f) for f in faces)   # = 2 E
    invariants_ok = (euler == 0) and (sum_face_sizes == 2 * n_e)

    fsd = face_size_distribution(faces)
    face_size_features = {f"n_faces_{k}": fsd.get(k, 0) for k in FACE_SIZE_BUCKETS}
    face_size_features["n_faces_ge13"] = sum(c for k, c in fsd.items() if k >= 13)
    face_sizes_list = [len(f) for f in faces]
    face_size_features["mean_face_size"] = float(np.mean(face_sizes_list))
    face_size_features["std_face_size"] = float(np.std(face_sizes_list))
    face_size_features["n_distinct_face_sizes"] = len(set(face_sizes_list))

    # Geometry features
    angles = vertex_interior_angles(half_edges, rotation)
    flat_angles = [a for av in angles for a in av]
    n_reentrant_angles = sum(1 for a in flat_angles if a > math.pi + 1e-9)
    n_reentrant_vertices = sum(
        1 for av in angles if any(a > math.pi + 1e-9 for a in av)
    )

    elens = edge_lengths(g)
    geom_features = {
        "mean_edge_length": float(np.mean(elens)),
        "std_edge_length": float(np.std(elens)),
        "min_edge_length": float(np.min(elens)),
        "max_edge_length": float(np.max(elens)),
        "n_reentrant_angles": n_reentrant_angles,
        "n_reentrant_vertices": n_reentrant_vertices,
        "frac_reentrant_vertices": n_reentrant_vertices / max(1, n_v),
        "mean_vertex_min_angle": float(np.mean(
            [min(av) for av in angles if av]
        )) if any(angles) else float("nan"),
        "mean_vertex_max_angle": float(np.mean(
            [max(av) for av in angles if av]
        )) if any(angles) else float("nan"),
        "cell_area": float(g["cell_area"]),
        "edge_density": n_e / max(1e-12, float(g["cell_area"])),
    }

    # Mechanical-derived features at default beam AR
    beam_idx = atlas_meta.get("beam_default_idx", 1)
    spring_idx = atlas_meta.get("spring_default_idx", 2)
    C_beam = g["beam"]["C_voigt"][beam_idx]
    K_b, G_b = planar_K_G(C_beam)
    nu_min_b = g["beam"]["nu_min"][beam_idx]
    nu_max_b = g["beam"]["nu_max"][beam_idx]
    nu_min_s = g["spring"]["nu_min"][spring_idx]
    nu_max_s = g["spring"]["nu_max"][spring_idx]
    cls_b = g["beam"]["classification"][beam_idx]
    cls_s = g["spring"]["classification"][spring_idx]

    mech_features = {
        "K_beam": K_b,
        "G_beam": G_b,
        "K_over_G_beam": (K_b / G_b) if (G_b not in (0.0,) and not math.isnan(G_b)
                                          and abs(G_b) > 1e-12) else float("nan"),
        "nu_anisotropy_beam": (nu_max_b - nu_min_b) if (nu_max_b is not None
                                                          and nu_min_b is not None) else float("nan"),
    }

    # Family-token features (binary indicators for each polygon size mentioned
    # in the family label string, e.g. "3_4_6_8" -> has_face_3, has_face_4, ...)
    family_tokens = set()
    for tok in family.split("_"):
        try:
            family_tokens.add(int(tok))
        except ValueError:
            pass
    family_features = {f"fam_token_{k}": int(k in family_tokens)
                       for k in FACE_SIZE_BUCKETS}

    out = {
        "name": name,
        "family": family,
        "n_rows": g["n_rows"],
        "n_cols": g["n_cols"],
        "n_vertices": n_v,
        "n_edges": n_e,
        "n_faces": n_f,
        "euler_check": euler,
        "handshake_check": sum_face_sizes - 2 * n_e,
        "invariants_ok": int(invariants_ok),
    }
    out.update(face_size_features)
    out.update(geom_features)
    out.update(mech_features)
    out.update(family_features)
    # Labels last, for readability of the CSV
    out.update({
        "nu_min_spring": nu_min_s,
        "nu_max_spring": nu_max_s,
        "spring_classification": cls_s,
        "nu_min_beam": nu_min_b,
        "nu_max_beam": nu_max_b,
        "beam_classification": cls_b,
        "is_homogeneous_auxetic": int(
            cls_b == "homogeneously_auxetic"
        ),
        "is_directional_auxetic": int(
            cls_b == "directionally_auxetic"
        ),
    })
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "atlas.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "features.csv"

    with open(in_path) as f:
        atlas = json.load(f)
    meta = atlas.get("metadata", {})
    grounds = atlas["grounds"]

    rows = []
    n_invariant_fail = 0
    for i, g in enumerate(grounds):
        try:
            row = extract_features(g, meta)
        except Exception as exc:
            print(f"  [{i}] FAIL {g.get('family')}/{g.get('name')}: {exc}",
                  file=sys.stderr)
            continue
        rows.append(row)
        if not row["invariants_ok"]:
            n_invariant_fail += 1
            print(f"  [{i}] INVARIANT VIOLATION {g['family']}/{g['name']}: "
                  f"euler={row['euler_check']}, handshake={row['handshake_check']}",
                  file=sys.stderr)
        if (i + 1) % 50 == 0 or i == len(grounds) - 1:
            print(f"  {i+1}/{len(grounds)} processed", flush=True)

    if not rows:
        print("No rows produced.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"Wrote {len(rows)} rows × {len(fieldnames)} columns -> {out_path}")
    if n_invariant_fail:
        print(f"WARNING: {n_invariant_fail} grounds had invariant violations "
              f"(see stderr).")
    else:
        print("All grounds passed Euler + handshake invariants.")


if __name__ == "__main__":
    main()
