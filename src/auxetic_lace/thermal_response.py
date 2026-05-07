"""
thermal_response.py
===================

Compute the macroscopic thermal expansion of a periodic ground when
struts are made of two materials with different coefficients of
thermal expansion (CTE): horizontal struts have α_h, vertical struts
have α_v. Diagonals split the difference (axial expansion proportional
to the strut's projection onto each axis).

Physical model: each strut's rest length depends on temperature
through axial eigenstrain ε_th = α · ΔT. The cell deforms to relieve
the resulting incompatibility, producing macroscopic strain
ε_eff = α_eff · ΔT.

For a unit differential CTE (α_h = +1, α_v = -1, ΔT = 1), we extract:
  alpha_xx, alpha_yy, alpha_xy : effective cell-strain components
  alpha_amplification          : sqrt(αxx² + αyy² + 2 αxy²) — magnitude
                                  of cell deformation per unit |α_h - α_v|

Implementation parallels humidity.py: assemble cell stiffness, build
forcing terms (f_u, f_eps) from per-strut axial eigenstrain, solve the
zero-macro-stress homogenization for ε_eff.

NOTE: Unlike humidity.py (which uses perpendicular swelling and
requires k_angular > 0), thermal expansion via axial strut
elongation works in the pin-jointed limit. We use the spring
mechanics module for consistency with the rest of the atlas, with
default k_angular = 0.01.

Public API
----------
  thermal_response(graph, k_per_unit_length=1.0, k_angular=0.01)
      -> Dict[str, float]

Returns:
  alpha_xx, alpha_yy, alpha_xy : cell strain components per unit
                                  (α_h - α_v) ΔT
  alpha_amplification          : magnitude of cell deformation
  alpha_anisotropic_signature  : alpha_xx - alpha_yy (positive = warp
                                  expands more, negative = weft does)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .parse_to_graph import LaceGraph
from .mechanics import (
    default_lattice_vectors,
    edge_geometry,
    assemble_stiffness,
    fix_translation,
)


def _strut_cte_class(unit: np.ndarray) -> float:
    """Classify a strut by direction and return its 'warp coefficient'
    in [0, 1] where 1 = pure horizontal (warp), 0 = pure vertical
    (weft), and 0.5 = diagonal.

    Specifically, returns u_x² (the x-projection-squared of the unit
    vector). For E or W struts → 1. For N or S → 0. For NE/NW/SE/SW
    → 0.5. This gives a smooth weighted blend for the diagonals.
    """
    return float(unit[0] ** 2)


def _thermal_forcing(graph: LaceGraph,
                       L: np.ndarray,
                       k_per_unit_length: float,
                       alpha_h: float,
                       alpha_v: float,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Build (f_u, f_eps) for warp-vs-weft thermal eigenstrain.

    Each strut elongates axially by α_strut * ΔT, where:
        α_strut = warp_coeff * α_h + (1 - warp_coeff) * α_v
        warp_coeff = u_x² (projection of strut direction onto x)

    For a strut along unit u with length L, axial eigenstrain
    ε_th = α_strut * ΔT contributes:
        f_u[src] -= k_axial * L * α_strut * u
        f_u[dst] += k_axial * L * α_strut * u
        f_eps    += k_axial * L * α_strut * (u . B_eps)
    where k_axial = k_per_unit_length / L (so k_axial * L =
    k_per_unit_length, uniform across struts of any length).

    We use ΔT = 1 throughout; final result scales linearly with ΔT.
    """
    N = len(graph.vertices)
    f_u = np.zeros(2 * N)
    f_eps = np.zeros(3)

    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        _delta_geom, length, unit = edge_geometry(graph, e_idx, L)
        if length < 1e-12:
            continue
        warp_coeff = _strut_cte_class(unit)
        alpha_strut = warp_coeff * alpha_h + (1.0 - warp_coeff) * alpha_v
        coeff = k_per_unit_length * alpha_strut

        wrap_shift = L @ np.array(
            [e.wrap[0] * graph.n_cols, e.wrap[1] * graph.n_rows],
            dtype=float,
        )
        wx, wy = wrap_shift[0], wrap_shift[1]
        B_eps = np.array(
            [[wx, 0.0, 0.5 * wy],
             [0.0, wy, 0.5 * wx]]
        )

        i_src = 2 * e.src_idx
        i_dst = 2 * e.dst_idx
        f_u[i_src:i_src + 2] -= coeff * unit
        f_u[i_dst:i_dst + 2] += coeff * unit
        f_eps += coeff * (unit @ B_eps)

    return f_u, f_eps


def _solve_strain_for(graph: LaceGraph,
                       L: np.ndarray,
                       K_red: np.ndarray,
                       G_red: np.ndarray,
                       H: np.ndarray,
                       kept: list,
                       k_per_unit_length: float,
                       alpha_h: float,
                       alpha_v: float,
                       ) -> np.ndarray:
    """Solve for ε_voigt under given (α_h, α_v) thermal pattern.

    Mirrors humidity._solve_strain_for. The Schur-complement form
    (H - G^T K^{-1} G) is exactly the homogenized stiffness — by
    design, it does not depend on the forcing terms. The forcing
    (f_eps - G^T K^{-1} f_u) is the effective thermal stress.
    Solving gives the strain that relaxes that stress to zero
    (free thermal expansion at zero macro stress).
    """
    f_u, f_eps = _thermal_forcing(graph, L, k_per_unit_length,
                                    alpha_h, alpha_v)
    f_u_red = f_u[kept]

    rank = np.linalg.matrix_rank(K_red, tol=1e-9)
    has_mech = rank < K_red.shape[0]
    if has_mech:
        K_inv_G = np.linalg.pinv(K_red, rcond=1e-9) @ G_red
        K_inv_fu = np.linalg.pinv(K_red, rcond=1e-9) @ f_u_red
    else:
        K_inv_G = np.linalg.solve(K_red, G_red)
        K_inv_fu = np.linalg.solve(K_red, f_u_red)

    C_eff = H - G_red.T @ K_inv_G
    C_eff = 0.5 * (C_eff + C_eff.T)
    f_eff = f_eps - G_red.T @ K_inv_fu

    eigvals = np.linalg.eigvalsh(C_eff)
    largest = float(eigvals[-1])
    softest = float(eigvals[0])
    if largest <= 0:
        return np.full(3, np.nan)
    if softest < 1e-9 * max(1.0, largest):
        return np.linalg.lstsq(C_eff, f_eff, rcond=1e-9)[0]
    return np.linalg.solve(C_eff, f_eff)


def thermal_response(graph: LaceGraph,
                      k_per_unit_length: float = 1.0,
                      k_angular: float = 0.01,
                      L: np.ndarray = None,
                      ) -> Dict[str, float]:
    """Compute the warp-vs-weft thermal expansion descriptors.

    The probe is a unit differential CTE: α_h = +1, α_v = -1, ΔT = 1.
    The response is linear, so applying any (α_h, α_v) gives a strain
    that scales as (α_h - α_v) for the differential signature, plus
    a uniform component proportional to (α_h + α_v)/2.

    Returns:
      alpha_xx, alpha_yy, alpha_xy : cell strain per (α_h - α_v) ΔT
      alpha_amplification          : magnitude of full strain tensor
      alpha_anisotropic_signature  : alpha_xx - alpha_yy (positive =
                                      warp axis expands more)
    """
    if L is None:
        L = default_lattice_vectors()

    K, G, H, A = assemble_stiffness(graph, L, k_per_unit_length, k_angular)
    K_red, G_red, kept = fix_translation(K, G)

    # Differential CTE: α_h = +1, α_v = -1
    eps = _solve_strain_for(graph, L, K_red, G_red, H, kept,
                              k_per_unit_length,
                              alpha_h=+1.0, alpha_v=-1.0)
    if np.any(np.isnan(eps)):
        return _nan_result()

    alpha_xx = float(eps[0])
    alpha_yy = float(eps[1])
    alpha_xy = 0.5 * float(eps[2])  # Voigt 2εxy → εxy
    alpha_amp = float(np.sqrt(alpha_xx ** 2 + alpha_yy ** 2
                                 + 2 * alpha_xy ** 2))
    alpha_aniso_sig = alpha_xx - alpha_yy

    return {
        "alpha_xx": alpha_xx,
        "alpha_yy": alpha_yy,
        "alpha_xy": alpha_xy,
        "alpha_amplification": alpha_amp,
        "alpha_anisotropic_signature": alpha_aniso_sig,
    }


def _nan_result() -> Dict[str, float]:
    return {
        "alpha_xx": float("nan"),
        "alpha_yy": float("nan"),
        "alpha_xy": float("nan"),
        "alpha_amplification": float("nan"),
        "alpha_anisotropic_signature": float("nan"),
    }
