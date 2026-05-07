"""
pore_response.py
================

Compute the directional pore-aperture response of a periodic ground.

For each loading angle θ, apply a small uniaxial strain ε along θ,
deform the cell using the beam-element homogenization, then compute
the mean inscribed-circle radius across all faces. The inscribed
circle radius is the largest disc that fits inside a face polygon —
a proxy for the characteristic flow aperture through that pore.

The profile r̄(θ) and its anisotropy are useful design metrics for
applications that care about pore geometry:
  - Permeability for fluid/gas flow
  - Light transmission
  - Sweat wicking in textiles

Public API
----------
  pore_response(graph, EA=1.0, aspect_ratio=10.0, n_angles=90,
                 strain_magnitude=0.01) -> Dict[str, Any]

Returns:
  inscribed_profile    : list of length n_angles, mean r at each θ
  inscribed_at_rest    : mean r at zero strain
  inscribed_relative   : list of length n_angles, r̄(θ) / r̄_rest - 1
                          (fractional change per applied strain unit)
  inscribed_max_change : max(abs(inscribed_relative))
  inscribed_anisotropy : (max - min) / (max + min) of profile, in [0, 1]
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog

from .parse_to_graph import LaceGraph
from .compute_family import _trace_faces_with_wrap, dart_endpoints
from .mechanics import default_lattice_vectors
from .mechanics_beam import homogenize_beam


# --------------------------------------------------------------------- #
# Inscribed-circle radius via Chebyshev-center LP
# --------------------------------------------------------------------- #

def inscribed_radius(polygon: List[Tuple[float, float]]) -> float:
    """Largest inscribed-circle radius in a polygon (assumes convex
    or near-convex; lace faces are typically convex)."""
    pts = list(polygon)
    n = len(pts)
    if n < 3:
        return 0.0

    sa = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        sa += x1 * y2 - x2 * y1
    if abs(sa) < 1e-12:
        return 0.0
    if sa < 0:
        pts = list(reversed(pts))

    A_rows, b_rhs = [], []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        elen = (ex * ex + ey * ey) ** 0.5
        if elen < 1e-12:
            return 0.0
        nx = ey / elen
        ny = -ex / elen
        A_rows.append([nx, ny, 1.0])
        b_rhs.append(nx * x1 + ny * y1)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rhs)
    c_obj = np.array([0.0, 0.0, -1.0])

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                       bounds=[(None, None), (None, None), (0, None)],
                       method='highs')
    except Exception:
        return 0.0
    if not res.success:
        return 0.0
    return float(res.x[2])


# --------------------------------------------------------------------- #
# Face polygons (unwrapped via cumulative wrap from face walk)
# --------------------------------------------------------------------- #

def face_polygons(graph: LaceGraph,
                    deformed_positions: np.ndarray,
                    L: np.ndarray) -> List[List[Tuple[float, float]]]:
    """Build face polygons in Cartesian coords from deformed vertex
    positions (lattice coords). Each polygon is unwrapped using the
    face's cumulative wrap so it forms a topologically simple closed
    curve in the plane.
    """
    walks, wraps = _trace_faces_with_wrap(graph)
    polys = []
    for walk, total_wrap in zip(walks, wraps):
        cumulative_wrap = (0, 0)
        verts_in_face = []
        for dart in walk:
            tail, _, w = dart_endpoints(graph, dart)
            tail_x = (deformed_positions[tail][0]
                       + cumulative_wrap[0] * graph.n_cols)
            tail_y = (deformed_positions[tail][1]
                       + cumulative_wrap[1] * graph.n_rows)
            cart = L @ np.array([tail_x, tail_y])
            verts_in_face.append((float(cart[0]), float(cart[1])))
            cumulative_wrap = (cumulative_wrap[0] + w[0],
                                cumulative_wrap[1] + w[1])
        if len(verts_in_face) >= 3:
            polys.append(verts_in_face)
    return polys


def per_face_inscribed_radii(graph: LaceGraph,
                                positions: np.ndarray,
                                L: np.ndarray) -> List[float]:
    """Return list of inscribed radii, one per face (excluding
    degenerate faces with r=0). Order matches face_polygons()."""
    polys = face_polygons(graph, positions, L)
    return [inscribed_radius(p) for p in polys]


def mean_inscribed_radius(polygons: List[List[Tuple[float, float]]]) -> float:
    """Mean of inscribed_radius across polygons (positive ones only)."""
    if not polygons:
        return 0.0
    rs = [inscribed_radius(p) for p in polygons]
    rs = [r for r in rs if r > 0]
    if not rs:
        return 0.0
    return float(np.mean(rs))


def mean_abs_relative_change(rest_radii: List[float],
                              deformed_radii: List[float]) -> float:
    """Mean of |Δr_i / r_i| across faces — captures total pore-aperture
    activity regardless of sign. This metric does NOT cancel between
    opening and closing faces, unlike a simple mean.

    Designed for auxetic structures where some faces open while
    others close in coupled mechanisms (e.g., kagome under stretch).
    """
    if not rest_radii or len(rest_radii) != len(deformed_radii):
        return 0.0
    changes = []
    for r0, r in zip(rest_radii, deformed_radii):
        if r0 > 1e-9:
            changes.append(abs((r - r0) / r0))
    if not changes:
        return 0.0
    return float(np.mean(changes))


# --------------------------------------------------------------------- #
# Deformed configurations from beam homogenization
# --------------------------------------------------------------------- #

def deformed_under_uniaxial(graph: LaceGraph,
                              u_full: np.ndarray,
                              theta: float,
                              strain_magnitude: float = 0.01) -> np.ndarray:
    """Deformed vertex positions under unit-strain at angle θ.

    Args:
      graph: LaceGraph
      u_full: (3N, 3) from homogenize_beam — internal DOF response
        per strain mode (εxx, εyy, 2εxy in Voigt).
      theta: loading angle in radians (uniaxial along this direction)
      strain_magnitude: applied strain magnitude (default 1%)

    The Voigt strain for unit uniaxial loading along θ̂:
        ε_voigt = strain_magnitude * [cos²θ, sin²θ, 2 sin(θ) cos(θ)]
    """
    N = len(graph.vertices)
    cos = np.cos(theta)
    sin = np.sin(theta)
    eps_voigt = strain_magnitude * np.array([
        cos * cos, sin * sin, 2 * cos * sin
    ])

    # u_full has 3 DOFs per vertex (ux, uy, theta) — keep first 2
    u_internal = u_full @ eps_voigt  # (3N,)
    u_internal = u_internal.reshape(N, 3)
    u_xy = u_internal[:, :2]

    rest = np.array(graph.vertices, dtype=float)
    eps_tensor = np.array([
        [eps_voigt[0], 0.5 * eps_voigt[2]],
        [0.5 * eps_voigt[2], eps_voigt[1]],
    ])
    macro_disp = rest @ eps_tensor.T
    return rest + u_xy + macro_disp


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #

def pore_response(graph: LaceGraph,
                   EA: float = 1.0,
                   aspect_ratio: float = 10.0,
                   n_angles: int = 90,
                   strain_magnitude: float = 0.01,
                   L: np.ndarray = None) -> Dict[str, Any]:
    """Compute directional pore-aperture profile for a ground.

    Primary metric: mean absolute relative change of inscribed-circle
    radius per face, per loading angle. This captures pore-aperture
    activity even when some faces open while others close (the
    typical auxetic mechanism), which a simple mean-radius would
    average to zero.

    Returns dict with:
      activity_profile     : mean |Δr/r| at each θ ∈ [0, π)
      activity_max         : max of profile
      activity_anisotropy  : (max - min) / (max + min) of profile
      mean_radius_profile  : auxiliary — mean r at each θ (often flat)
      mean_radius_at_rest  : baseline mean r
    """
    if L is None:
        L = default_lattice_vectors()

    try:
        C_voigt, A, u_full = homogenize_beam(
            graph, L_lattice=L, EA=EA, aspect_ratio=aspect_ratio)
    except Exception:
        return _nan_result(n_angles)

    rest = np.array(graph.vertices, dtype=float)
    rest_radii = per_face_inscribed_radii(graph, rest, L)
    if not rest_radii or all(r <= 0 for r in rest_radii):
        return _nan_result(n_angles)

    r_rest_mean = float(np.mean([r for r in rest_radii if r > 0]))

    angles = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    activity_profile = np.zeros(n_angles)
    mean_radius_profile = np.zeros(n_angles)

    for i, theta in enumerate(angles):
        try:
            deformed = deformed_under_uniaxial(
                graph, u_full, float(theta), strain_magnitude)
            def_radii = per_face_inscribed_radii(graph, deformed, L)
            activity_profile[i] = mean_abs_relative_change(
                rest_radii, def_radii)
            valid = [r for r in def_radii if r > 0]
            mean_radius_profile[i] = (
                float(np.mean(valid)) if valid else 0.0)
        except Exception:
            activity_profile[i] = 0.0
            mean_radius_profile[i] = r_rest_mean

    a_max = float(activity_profile.max())
    a_min = float(activity_profile.min())
    a_mean = float(activity_profile.mean())
    a_std = float(activity_profile.std())
    # Coefficient of variation: std / mean. Behaves well across all
    # signal levels — for flat profiles (cloth, isotropic), CV ≈ 0
    # regardless of overall activity. For directional auxetics,
    # CV > 0 captures the angular variation.
    a_cv = a_std / a_mean if a_mean > 1e-9 else 0.0

    return {
        "activity_profile": activity_profile.tolist(),
        "activity_max": a_max,
        "activity_min": a_min,
        "activity_mean": a_mean,
        "activity_anisotropy": float(a_cv),
        "mean_radius_profile": mean_radius_profile.tolist(),
        "mean_radius_at_rest": r_rest_mean,
    }


def _nan_result(n_angles: int) -> Dict[str, Any]:
    return {
        "activity_profile": [float('nan')] * n_angles,
        "activity_max": float('nan'),
        "activity_min": float('nan'),
        "activity_mean": float('nan'),
        "activity_anisotropy": float('nan'),
        "mean_radius_profile": [float('nan')] * n_angles,
        "mean_radius_at_rest": float('nan'),
    }


# --------------------------------------------------------------------- #
# Self-test (LP correctness)
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    # Test inscribed-circle LP on simple shapes
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    r = inscribed_radius(sq)
    print(f"Unit square: r = {r:.4f}  (expected 0.5)")

    tri = [(0, 0), (3, 0), (0, 4)]
    r = inscribed_radius(tri)
    print(f"3-4-5 right triangle: r = {r:.4f}  (expected 1.0)")

    hex_unit = [
        (1, 0), (0.5, 0.866), (-0.5, 0.866),
        (-1, 0), (-0.5, -0.866), (0.5, -0.866),
    ]
    r = inscribed_radius(hex_unit)
    print(f"Unit regular hexagon: r = {r:.4f}  (expected ~0.866)")
