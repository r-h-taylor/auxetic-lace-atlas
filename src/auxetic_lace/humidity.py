"""
humidity.py
===========

Compute the macroscopic strain response of a periodic 2D lattice to uniform
swelling of every strut. This characterizes how much the lattice topology
amplifies (or suppresses) bulk material swelling — a metric relevant for
humidity-adaptive 3D-printed meshes made from hydrophilic polymers.

Physics
-------
When every strut wants to elongate by a uniform fractional amount delta
(an axial eigenstrain), the lattice deforms its unit cell to relieve the
internal incompatibility. The resulting macroscopic strain eps_eff/delta
is independent of delta in the linear regime and depends only on the
lattice topology + elastic tensor.

Internally we extend the existing periodic-homogenization scheme. The
swelling enters as a forcing term on both the internal displacement DOFs
u and the macro strain DOFs eps. With the unit cell in zero macro stress
(unconstrained boundary), the equilibrium [K G; G^T H][u; eps] = [f_u; f_eps]
yields:

    eps_eff = (H - G^T K^{-1} G)^{-1} (f_eps - G^T K^{-1} f_u) / delta

The first factor is exactly the homogenized compliance S = C^{-1}, so the
swelling response is:

    eps_eff/delta = S * (effective swelling stress)

where the "effective swelling stress" is what bulk swelling would exert
on the homogenized continuum.

Public API
----------
    humidity_features(graph, k_angular=0.01, k_per_unit_length=1.0,
                      strut_radius=0.05) -> Dict[str, float]

Returns 8 scalar descriptors:
    eta_xx, eta_yy, eta_xy   : macroscopic strain components, normalized by delta
    eta_mean                 : (eta_xx + eta_yy) / 2  — mean amplification
    eta_aniso                : eta_xx - eta_yy        — directional anisotropy
    eta_max, eta_min         : principal strains      — frame-invariant
    eta_anisotropy_ratio     : |eta_max / eta_min|    — dimensionless ranking
    eta_pore                 : pore-area amplification (depends on strut_radius)

For an isotropic, geometrically compatible structure: eta_mean = 1,
eta_aniso = 0. Numbers significantly above 1 (or significantly anisotropic)
indicate topology-driven amplification of bulk swelling.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .parse_to_graph import LaceGraph
from .mechanics import (
    default_lattice_vectors,
    edge_geometry,
    cell_area,
    assemble_stiffness,
    fix_translation,
)


def _swelling_forcing(graph: LaceGraph,
                       L: np.ndarray,
                       k_per_unit_length: float,
                       k_angular: float,
                       delta_x: float = 1.0,
                       delta_y: float = 1.0,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Build (f_u, f_eps) for perpendicular (radial) strut eigenstrain.

    Physical model: a strut is like a wooden stick or fiber with grain
    along its length. Water absorbs into the cell walls and swells the
    strut perpendicular to its axis (radial swelling), with negligible
    axial elongation. The eigenstrain on each strut is therefore in the
    perpendicular direction, not along the axis.

    For a strut along unit u = (u_x, u_y), the perpendicular is
    perp = (-u_y, u_x). The eigenstrain magnitude can depend on strut
    orientation:
        delta_perp(u) = delta_x * u_x^2 + delta_y * u_y^2
    where (delta_x, delta_y) characterizes the anisotropy of the bulk
    polymer's swelling. With delta_x = delta_y = 1, every strut swells
    perpendicular by the same amount, but this is NOT geometrically
    compatible (unlike axial uniform swelling) because the perpendicular
    direction depends on strut orientation.

    The forcing acts through the angular/perpendicular spring component:
      f_u[src] -= k_an * length * delta_perp * perp
      f_u[dst] += k_an * length * delta_perp * perp
      f_eps    += k_an * length * delta_perp * (perp . B_eps)

    Note: the response scales with k_angular. A pure pin-jointed lattice
    (k_angular = 0) cannot transmit perpendicular swelling forces and
    will give a zero response.
    """
    N = len(graph.vertices)
    f_u = np.zeros(2 * N)
    f_eps = np.zeros(3)

    if k_angular <= 0:
        return f_u, f_eps  # Pin-jointed: no perpendicular response

    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        _delta_geom, length, unit = edge_geometry(graph, e_idx, L)
        if length < 1e-12:
            continue
        perp = np.array([-unit[1], unit[0]])
        # Anisotropic swelling magnitude based on strut orientation
        cos2 = unit[0] * unit[0]
        sin2 = unit[1] * unit[1]
        delta_strut = delta_x * cos2 + delta_y * sin2
        # k_an * length = k_angular, so the prefactor is uniform per edge
        coeff = k_angular * delta_strut

        wrap_shift = L @ np.array(
            [e.wrap[0] * graph.n_cols, e.wrap[1] * graph.n_rows], dtype=float
        )
        wx, wy = wrap_shift[0], wrap_shift[1]
        B_eps = np.array(
            [[wx, 0.0, 0.5 * wy],
             [0.0, wy, 0.5 * wx]]
        )

        i_src = 2 * e.src_idx
        i_dst = 2 * e.dst_idx
        f_u[i_src:i_src + 2] -= coeff * perp
        f_u[i_dst:i_dst + 2] += coeff * perp
        f_eps += coeff * (perp @ B_eps)

    return f_u, f_eps


def _solve_strain_for(graph: LaceGraph,
                       L: np.ndarray,
                       K_red: np.ndarray,
                       G_red: np.ndarray,
                       H: np.ndarray,
                       kept: list,
                       k_per_unit_length: float,
                       k_angular: float,
                       delta_x: float,
                       delta_y: float,
                       ) -> np.ndarray:
    """Solve for eps_voigt under a given (delta_x, delta_y) swelling pattern."""
    f_u, f_eps = _swelling_forcing(graph, L, k_per_unit_length, k_angular,
                                     delta_x, delta_y)
    f_u_red = f_u[kept]

    rank = np.linalg.matrix_rank(K_red, tol=1e-9)
    has_mech = rank < K_red.shape[0]
    if has_mech:
        K_inv_G = np.linalg.pinv(K_red, rcond=1e-9) @ G_red
        K_inv_fu = np.linalg.pinv(K_red, rcond=1e-9) @ f_u_red
    else:
        K_inv_G = np.linalg.solve(K_red, G_red)
        K_inv_fu = np.linalg.solve(K_red, f_u_red)

    C_eff_unscaled = H - G_red.T @ K_inv_G
    C_eff_unscaled = 0.5 * (C_eff_unscaled + C_eff_unscaled.T)
    f_eff = f_eps - G_red.T @ K_inv_fu

    eigvals = np.linalg.eigvalsh(C_eff_unscaled)
    largest = float(eigvals[-1])
    softest = float(eigvals[0])
    if largest <= 0:
        return np.full(3, np.nan)
    if softest < 1e-9 * max(1.0, largest):
        return np.linalg.lstsq(C_eff_unscaled, f_eff, rcond=1e-9)[0]
    return np.linalg.solve(C_eff_unscaled, f_eff)


def humidity_features(graph: LaceGraph,
                       k_per_unit_length: float = 1.0,
                       k_angular: float = 0.01,
                       strut_radius: float = 0.05,
                       L: np.ndarray = None,
                       ) -> Dict[str, float]:
    """Compute the swelling-response descriptors.

    Physical model: each strut is treated as a fiber/wood-like element with
    grain along its axis. Water absorption swells the strut perpendicular to
    its long axis (radial swelling), with negligible axial elongation.
    This is appropriate for cotton thread, bamboo splints, oriented 3D-printed
    struts (deposition along the strut), and fiber-reinforced composites.

    The probe is uniform isotropic perpendicular swelling (delta_x = delta_y
    = 1): every strut swells perpendicular to itself by the same amount.
    Unlike uniform axial swelling (which is geometrically compatible and
    yields a trivial response), uniform perpendicular swelling is NOT
    compatible because the perpendicular direction varies with strut
    orientation. This produces a topology-dependent macroscopic strain that
    is the cleanest measure of how the lattice geometry couples to bulk
    swelling.

    The response scales with k_angular. With k_angular = 0 (pin-jointed
    framework), the response is identically zero — there is no mechanism
    for a strut's perpendicular swelling to drive vertex motion. We use
    k_angular = 0.01 (the spring-default value) for consistency with the
    rest of the atlas.

    Returns 11 scalars:
        eta_xx, eta_yy, eta_xy : strain components, isotropic perp swelling
        eta_xx_x, eta_yy_x, eta_xy_x : strain components when only x-perp
                                        struts swell (delta_x=1, delta_y=0)
        eta_aniso              : eta_xx - eta_yy (cell-axis anisotropy)
        eta_max, eta_min       : principal strains
        eta_anisotropy_ratio   : |eta_max / eta_min|, frame-invariant
        eta_pore               : pore-area amplification, isotropic mode
    """
    if L is None:
        L = default_lattice_vectors()

    K, G, H, A = assemble_stiffness(graph, L, k_per_unit_length, k_angular)
    K_red, G_red, kept = fix_translation(K, G)

    # Mode 1: isotropic perpendicular swelling (every strut swells perp by 1)
    eps_iso = _solve_strain_for(graph, L, K_red, G_red, H, kept,
                                 k_per_unit_length, k_angular,
                                 delta_x=1.0, delta_y=1.0)
    if np.any(np.isnan(eps_iso)):
        return _nan_result()

    eta_xx = float(eps_iso[0])
    eta_yy = float(eps_iso[1])
    eta_xy = 0.5 * float(eps_iso[2])

    # Mode 2: anisotropic — only x-aligned struts swell perpendicular
    eps_xa = _solve_strain_for(graph, L, K_red, G_red, H, kept,
                                k_per_unit_length, k_angular,
                                delta_x=1.0, delta_y=0.0)
    eta_xx_x = float(eps_xa[0]) if not np.any(np.isnan(eps_xa)) else float("nan")
    eta_yy_x = float(eps_xa[1]) if not np.any(np.isnan(eps_xa)) else float("nan")
    eta_xy_x = (0.5 * float(eps_xa[2])
                  if not np.any(np.isnan(eps_xa)) else float("nan"))

    eta_aniso = eta_xx - eta_yy

    eps_tensor = np.array([[eta_xx, eta_xy], [eta_xy, eta_yy]])
    pvals = np.linalg.eigvalsh(eps_tensor)
    eta_min = float(pvals[0])
    eta_max = float(pvals[1])
    if abs(eta_min) > 1e-12:
        eta_anisotropy_ratio = abs(eta_max / eta_min)
    else:
        eta_anisotropy_ratio = float("inf") if abs(eta_max) > 1e-12 else 1.0

    # Pore-area amplification under isotropic perpendicular swelling.
    # Cell-area change: (eta_xx + eta_yy) * delta * A
    # Strut footprint change: each strut's radius grows by delta (every
    # strut is swelling radially), and length stays roughly constant. So
    # footprint = 2 * radius * total_length grows linearly in delta.
    if strut_radius > 0:
        total_strut_length = 0.0
        for e_idx in range(len(graph.edges)):
            _, length, _ = edge_geometry(graph, e_idx, L)
            total_strut_length += length
        footprint = 2.0 * strut_radius * total_strut_length
        A_pore = max(A - footprint, 1e-12)
        # d(A_cell)/delta = (eta_xx + eta_yy) * A
        # d(footprint)/delta = 2 * strut_radius * total_length = footprint
        d_pore = (eta_xx + eta_yy) * A - footprint
        eta_pore = float(d_pore / A_pore)
    else:
        eta_pore = float(eta_xx + eta_yy)

    return {
        "eta_xx": eta_xx,
        "eta_yy": eta_yy,
        "eta_xy": eta_xy,
        "eta_xx_x": eta_xx_x,
        "eta_yy_x": eta_yy_x,
        "eta_xy_x": eta_xy_x,
        "eta_aniso": eta_aniso,
        "eta_max": eta_max,
        "eta_min": eta_min,
        "eta_anisotropy_ratio": eta_anisotropy_ratio,
        "eta_pore": eta_pore,
    }


def _nan_result() -> Dict[str, float]:
    """Return a dictionary with all-NaN values when computation fails."""
    return {
        "eta_xx": float("nan"),
        "eta_yy": float("nan"),
        "eta_xy": float("nan"),
        "eta_xx_x": float("nan"),
        "eta_yy_x": float("nan"),
        "eta_xy_x": float("nan"),
        "eta_aniso": float("nan"),
        "eta_max": float("nan"),
        "eta_min": float("nan"),
        "eta_anisotropy_ratio": float("nan"),
        "eta_pore": float("nan"),
    }
