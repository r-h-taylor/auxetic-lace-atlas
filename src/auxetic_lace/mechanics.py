"""
mechanics.py
============

Periodic bar-and-pin mechanical homogenization for TesseLace ground graphs.

For each LaceGraph (parsed from a TesseLace .txt template), we treat:
  - Vertices = pin joints (free to translate, no moments)
  - Edges    = linear axial springs of natural length |dst - src|
               (no bending stiffness)
  - Periodic = wrap vectors in the LaceGraph encode toroidal connectivity

We solve four periodic-boundary problems to extract the 2D effective
elastic tensor C_ij:
  - epsilon_xx = 1, others zero
  - epsilon_yy = 1, others zero
  - 2*epsilon_xy = 1, others zero (engineering shear)
  - identity reference

For each strain mode, the macroscopic deformation gradient
  F = I + epsilon
defines an "affine" target displacement at each periodic image of every
vertex. We solve for the internal periodic perturbation u(v) at each
vertex v in the unit cell that minimizes total spring energy. The
resulting stress tensor (per unit cell area) gives the elastic response.

Outputs:
  - Effective elastic tensor C (in 2D Voigt: 3x3 matrix)
  - Compliance S = C^{-1}
  - Poisson ratio nu(theta) over in-plane directions theta in [0, pi)
  - nu_min, nu_max, anisotropy
  - Classification: directionally auxetic (nu_min < 0),
                    homogeneously auxetic (nu_max < 0)

This is the Borcea-Streinu / Maxwell-Calladine bar-and-pin model, the
standard reference framework for geometric auxetics. It captures
mechanism-driven auxetic behavior (rotating squares, re-entrant
honeycombs, etc.) cleanly.

Usage:
    python3 mechanics.py --single tesselace_catalog/tl/4/3x3_19.txt
    python3 mechanics.py --catalog tesselace_catalog --output results.csv

Attribution: TesseLace ground patterns by Veronika Irvine (CC-BY 4.0).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .parse_to_graph import LaceGraph, parse_file, parse_manifest


# -----------------------------------------------------------------------
# Lattice and embedding
# -----------------------------------------------------------------------

def default_lattice_vectors() -> np.ndarray:
    """Default lattice basis: unit square, a1 = (1, 0), a2 = (0, 1).
    Returns a 2x2 matrix with a1, a2 as columns:
        L = [[a1x, a2x], [a1y, a2y]]
    so that a real-space position of vertex v in tile (tx, ty) is
        p = L @ [v_col + tx * n_cols, v_row + ty * n_rows]^T
    """
    return np.eye(2)


def vertex_position(graph: LaceGraph, idx: int,
                    L: np.ndarray = None) -> np.ndarray:
    """Return cartesian position of vertex idx in the (0,0) tile."""
    if L is None:
        L = default_lattice_vectors()
    v = graph.vertices[idx]
    return L @ np.array([v[0], v[1]], dtype=float)


def cell_vectors(graph: LaceGraph, L: np.ndarray = None) -> np.ndarray:
    """Period parallelogram basis vectors.
    Returns 2x2 matrix [b1 | b2] where b1 spans n_cols in the col direction,
    b2 spans n_rows in the row direction (in cartesian coordinates)."""
    if L is None:
        L = default_lattice_vectors()
    b1 = L @ np.array([graph.n_cols, 0.0])
    b2 = L @ np.array([0.0, graph.n_rows])
    return np.column_stack([b1, b2])


def cell_area(graph: LaceGraph, L: np.ndarray = None) -> float:
    """Area of the period parallelogram."""
    B = cell_vectors(graph, L)
    return abs(np.linalg.det(B))


# -----------------------------------------------------------------------
# Stiffness assembly
# -----------------------------------------------------------------------

def edge_geometry(graph: LaceGraph, edge_idx: int,
                  L: np.ndarray = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return (delta, length, unit_vector) for an edge.

    delta is the cartesian vector from src to dst, accounting for periodic
    wrapping (i.e. the edge points across the period if wrap != (0,0)).
    """
    if L is None:
        L = default_lattice_vectors()
    e = graph.edges[edge_idx]
    p_src = vertex_position(graph, e.src_idx, L)
    p_dst = vertex_position(graph, e.dst_idx, L)
    # Apply wrap: dst is in tile (wrap_col, wrap_row), so add the
    # corresponding shift.
    wrap_shift = L @ np.array([e.wrap[0] * graph.n_cols,
                                e.wrap[1] * graph.n_rows], dtype=float)
    delta = (p_dst + wrap_shift) - p_src
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        # Degenerate; should not happen for a proper ground
        unit = np.zeros(2)
    else:
        unit = delta / length
    return delta, length, unit


def assemble_stiffness(graph: LaceGraph,
                       L: np.ndarray = None,
                       k_per_unit_length: float = 1.0,
                       k_angular: float = 0.0,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble the periodic stiffness matrix and the strain-coupling
    matrix.

    Degrees of freedom: 2N where N is the number of vertices in the unit
    cell (each vertex contributes ux, uy). The DOF vector u stacks them
    in vertex order.

    Two stiffness contributions:

    1. AXIAL (always present): each edge from src to dst is a linear axial
       spring with stiffness k = k_per_unit_length / length, contributing
       a standard truss element to K. The bond elongation under macro
       strain epsilon plus internal perturbation u is:
           e_axial = unit . [u(dst) + epsilon . wrap_shift - u(src)]
       The energy is (1/2) k e_axial^2.

    2. ANGULAR (optional, controlled by k_angular): for each vertex we
       penalize deviations of every bond direction from its rest direction
       by a quadratic spring on the perpendicular displacement. This is
       a simplified beam-bending regularization that breaks pin-jointed
       mechanism modes without committing to a full beam-element model.

       For each bar at vertex v with rest unit vector u_hat, the
       perpendicular component of the relative displacement is
           e_perp = u_perp_hat . [u(other_end) - u(v)]
       and we add (1/2) k_angular * e_perp^2 / length to the energy.
       This is identical in form to the axial term but with the
       perpendicular unit vector and a separate stiffness scale.

    Setting k_angular = 0 reproduces pure pin-jointed behavior. Setting
    k_angular > 0 gives a regularized framework with finite stiffness
    in all macroscopic strain modes, suitable for extracting Poisson
    ratios that reflect the geometric mechanism + bending response.

    Returns (K, G, H, A) where A is the cell area.
    """
    if L is None:
        L = default_lattice_vectors()

    N = len(graph.vertices)
    K = np.zeros((2 * N, 2 * N))
    G = np.zeros((2 * N, 3))     # u-eps coupling, eps in Voigt [exx, eyy, 2exy]
    H = np.zeros((3, 3))         # eps-eps quadratic
    A = cell_area(graph, L)

    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        delta, length, unit = edge_geometry(graph, e_idx, L)
        if length < 1e-12:
            continue
        k_ax = k_per_unit_length / length

        # Wrap shift in cartesian coordinates
        wrap_shift = L @ np.array([e.wrap[0] * graph.n_cols,
                                    e.wrap[1] * graph.n_rows], dtype=float)
        wx, wy = wrap_shift[0], wrap_shift[1]

        # Voigt strain-to-displacement coupling for the wrap shift:
        # epsilon . wrap = [exx*wx + exy*wy, eyy*wy + exy*wx]
        # = B_eps . eps_voigt (eps_voigt = [exx, eyy, 2exy])
        B_eps = np.array([
            [wx, 0.0, 0.5 * wy],
            [0.0, wy, 0.5 * wx],
        ])

        i_src = 2 * e.src_idx
        i_dst = 2 * e.dst_idx

        # ----- Axial contribution -----
        # bond elongation = unit . (u_dst - u_src + eps . wrap)
        nn = np.outer(unit, unit)
        K[i_src:i_src+2, i_src:i_src+2] += k_ax * nn
        K[i_dst:i_dst+2, i_dst:i_dst+2] += k_ax * nn
        K[i_src:i_src+2, i_dst:i_dst+2] -= k_ax * nn
        K[i_dst:i_dst+2, i_src:i_src+2] -= k_ax * nn
        unit_dot_B = unit @ B_eps
        G[i_src:i_src+2, :] -= k_ax * np.outer(unit, unit_dot_B)
        G[i_dst:i_dst+2, :] += k_ax * np.outer(unit, unit_dot_B)
        H += k_ax * np.outer(unit_dot_B, unit_dot_B)

        # ----- Angular (bending) contribution -----
        if k_angular > 0:
            k_an = k_angular / length
            # Perpendicular unit vector (rotate unit by 90 degrees CCW)
            perp = np.array([-unit[1], unit[0]])
            # Same form as axial but with perp instead of unit
            pp = np.outer(perp, perp)
            K[i_src:i_src+2, i_src:i_src+2] += k_an * pp
            K[i_dst:i_dst+2, i_dst:i_dst+2] += k_an * pp
            K[i_src:i_src+2, i_dst:i_dst+2] -= k_an * pp
            K[i_dst:i_dst+2, i_src:i_src+2] -= k_an * pp
            perp_dot_B = perp @ B_eps
            G[i_src:i_src+2, :] -= k_an * np.outer(perp, perp_dot_B)
            G[i_dst:i_dst+2, :] += k_an * np.outer(perp, perp_dot_B)
            H += k_an * np.outer(perp_dot_B, perp_dot_B)

    return K, G, H, A


# -----------------------------------------------------------------------
# Homogenization: solve for effective elastic tensor
# -----------------------------------------------------------------------

def fix_translation(K: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Remove the rigid-body translation null space by fixing one vertex.

    Returns (K_reduced, G_reduced, kept_indices) where kept_indices lists
    the DOFs that remain (i.e. all DOFs except the first vertex's ux, uy
    which are pinned to zero).
    """
    n = K.shape[0]
    # Pin vertex 0 (DOFs 0 and 1)
    keep = list(range(2, n))
    K_red = K[np.ix_(keep, keep)]
    G_red = G[keep, :]
    return K_red, G_red, keep


def homogenize(graph: LaceGraph,
               L: np.ndarray = None,
               k_per_unit_length: float = 1.0,
               k_angular: float = 0.0,
               ) -> Dict:
    """Compute the homogenized 2D elastic tensor for a periodic ground.

    Parameters
    ----------
    k_per_unit_length : axial spring stiffness per unit length (the bar
                        stiffness scale; k_axial = k_per_unit_length / length)
    k_angular         : bending/angular stiffness scale. With k_angular = 0
                        we have a pure pin-jointed bar framework, which
                        for most lace grounds is mechanism-floppy.
                        With k_angular > 0 we add a perpendicular spring
                        at each bond that breaks mechanism modes and
                        gives a finite stiffness in all macroscopic
                        strain directions. A typical regularization
                        choice is k_angular = 0.01 * k_per_unit_length,
                        making bending much softer than axial.

    Procedure (standard periodic homogenization):
      1. Assemble K, G, H, A.
      2. Pin one vertex to fix rigid-body translation.
      3. Solve for the periodic perturbation u^* given each unit strain:
            K u^* = -G eps
         giving u^* = -K^{-1} G eps for each eps in Voigt basis.
      4. Effective elastic tensor (per unit area):
            C_eff = (H + G^T u^*) / A
                  = (H - G^T K^{-1} G) / A
    """
    K, G, H, A = assemble_stiffness(graph, L, k_per_unit_length, k_angular)
    K_red, G_red, kept = fix_translation(K, G)

    rank = np.linalg.matrix_rank(K_red, tol=1e-9)
    n_dof = K_red.shape[0]
    n_internal_mechanisms = n_dof - rank

    if n_internal_mechanisms > 0:
        K_inv_G = np.linalg.pinv(K_red, rcond=1e-9) @ G_red
    else:
        K_inv_G = np.linalg.solve(K_red, G_red)

    C = (H - G_red.T @ K_inv_G) / A
    C = 0.5 * (C + C.T)

    C_eigvals = np.linalg.eigvalsh(C)
    C_eigvals_desc = np.sort(C_eigvals)[::-1]
    if C_eigvals_desc[0] > 0:
        soft_threshold = max(1e-9, 1e-6 * C_eigvals_desc[0])
    else:
        soft_threshold = 1e-9
    n_macro_soft_modes = int(np.sum(C_eigvals < soft_threshold))

    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        S = None

    return {
        "C": C,
        "S": S,
        "A": A,
        "n_dof": n_dof,
        "n_internal_mechanisms": n_internal_mechanisms,
        "n_macro_soft_modes": n_macro_soft_modes,
        "C_eigvals": C_eigvals_desc,
    }


# -----------------------------------------------------------------------
# Poisson ratio analysis
# -----------------------------------------------------------------------

def voigt_to_tensor(c_voigt: np.ndarray) -> np.ndarray:
    """Convert 3x3 Voigt C (engineering shear) to 4-index symmetric form.
    Returns a 2x2x2x2 tensor with C_ijkl symmetries."""
    # Voigt indexing: 1<->xx, 2<->yy, 3<->xy (engineering, gamma_xy = 2*eps_xy)
    Cijkl = np.zeros((2, 2, 2, 2))
    Cijkl[0, 0, 0, 0] = c_voigt[0, 0]   # C_xxxx
    Cijkl[1, 1, 1, 1] = c_voigt[1, 1]   # C_yyyy
    Cijkl[0, 0, 1, 1] = c_voigt[0, 1]   # C_xxyy
    Cijkl[1, 1, 0, 0] = c_voigt[0, 1]
    # shear terms: C_voigt[2,2] = C_xyxy * 4 due to engineering shear convention
    # Actually: with eps_voigt = [exx, eyy, 2*exy] and Voigt stress
    # sig_voigt = [sxx, syy, sxy], the relation sig = C eps gives
    # C_voigt[2,2] = C_xyxy (not multiplied), but careful:
    # sig_xy = 2 * C_xyxy * eps_xy  -- in engineering shear,
    # sig_xy = C_voigt[2,2] * (2 * eps_xy) so C_xyxy = C_voigt[2,2].
    Cijkl[0, 1, 0, 1] = c_voigt[2, 2]
    Cijkl[1, 0, 0, 1] = c_voigt[2, 2]
    Cijkl[0, 1, 1, 0] = c_voigt[2, 2]
    Cijkl[1, 0, 1, 0] = c_voigt[2, 2]
    # cross shear-axial: C_voigt[0,2] = C_xxxy * 2
    Cijkl[0, 0, 0, 1] = c_voigt[0, 2]
    Cijkl[0, 0, 1, 0] = c_voigt[0, 2]
    Cijkl[0, 1, 0, 0] = c_voigt[0, 2]
    Cijkl[1, 0, 0, 0] = c_voigt[0, 2]
    Cijkl[1, 1, 0, 1] = c_voigt[1, 2]
    Cijkl[1, 1, 1, 0] = c_voigt[1, 2]
    Cijkl[0, 1, 1, 1] = c_voigt[1, 2]
    Cijkl[1, 0, 1, 1] = c_voigt[1, 2]
    return Cijkl


def rotate_voigt(c_voigt: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a Voigt C matrix by angle theta (radians).

    Returns the C tensor in a frame rotated by theta. We use the
    Bond/Voigt rotation matrix M(theta) such that C' = M C M^T."""
    c, s = np.cos(theta), np.sin(theta)
    M = np.array([
        [c*c,    s*s,     2*c*s],
        [s*s,    c*c,    -2*c*s],
        [-c*s,   c*s,    c*c - s*s],
    ])
    return M @ c_voigt @ M.T


def poisson_ratio_at_angle(c_voigt: np.ndarray, theta: float) -> float:
    """Poisson ratio for uniaxial stress applied at angle theta (radians).

    Standard definition: nu(theta) = -eps_perp / eps_parallel under
    uniaxial stress sigma in direction theta. Equivalent to
    nu(theta) = -S_12'(theta) / S_11'(theta) in the rotated frame."""
    try:
        S = np.linalg.inv(c_voigt)
    except np.linalg.LinAlgError:
        return np.nan
    S_rot = rotate_voigt(S, theta)
    if abs(S_rot[0, 0]) < 1e-15:
        return np.nan
    return -S_rot[0, 1] / S_rot[0, 0]


def poisson_profile(c_voigt: np.ndarray,
                    n_samples: int = 180) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Poisson ratio over angles theta in [0, pi).
    Returns (thetas, nu_values).
    """
    thetas = np.linspace(0, np.pi, n_samples, endpoint=False)
    nus = np.array([poisson_ratio_at_angle(c_voigt, t) for t in thetas])
    return thetas, nus


def analyze(graph: LaceGraph,
            L: np.ndarray = None,
            k_per_unit_length: float = 1.0,
            k_angular: float = 0.01,
            n_angle_samples: int = 180) -> Dict:
    """Full mechanical analysis of one ground.

    Default k_angular = 0.01 means bending is 100x softer than axial,
    a regularization that breaks pin-jointed mechanism modes without
    dominating the response. Set k_angular = 0 for the pure pin-jointed
    framework (Borcea-Streinu geometric auxetic limit), in which most
    grounds will be reported as mechanism-floppy.
    """
    h = homogenize(graph, L, k_per_unit_length, k_angular)
    C = h["C"]
    n_macro_soft = h["n_macro_soft_modes"]

    if n_macro_soft >= 3:
        # Fully floppy: every macroscopic strain mode is soft.
        nu_min = np.nan
        nu_max = np.nan
        anisotropy = np.nan
        classification = "fully_floppy"
        thetas = nus = None
    else:
        thetas, nus = poisson_profile(C, n_angle_samples)
        # Filter out angles where the framework's response in that direction
        # is dominated by soft modes -- numerically these show up as
        # extreme nu values that reflect 1/(small_eigenvalue), not real
        # Poisson behavior. We use a robustness filter: keep only angles
        # where S_11(theta) is bounded.
        try:
            S = np.linalg.inv(C)
            # Compute rotated S_11 at each theta directly
            S11_rot = np.array([rotate_voigt(S, t)[0, 0] for t in thetas])
            # Keep angles where rotated S_11 is finite and not dominated
            # by soft-mode contribution
            S_max = np.median(np.abs(S11_rot)) * 100  # heuristic
            valid_mask = np.abs(S11_rot) < S_max
            valid_mask &= ~np.isnan(nus)
        except (np.linalg.LinAlgError, ValueError):
            valid_mask = ~np.isnan(nus)

        if not np.any(valid_mask):
            nu_min = np.nan
            nu_max = np.nan
            anisotropy = np.nan
            classification = "ill_conditioned"
        else:
            nus_valid = nus[valid_mask]
            nu_min = float(np.min(nus_valid))
            nu_max = float(np.max(nus_valid))
            anisotropy = nu_max - nu_min
            if n_macro_soft >= 1:
                # Mechanism-driven response: C has soft modes. Poisson is
                # well-defined only in the rigid subspace.
                if nu_max < 0:
                    classification = "homogeneously_auxetic_with_mechanism"
                elif nu_min < 0:
                    classification = "directionally_auxetic_with_mechanism"
                else:
                    classification = "non_auxetic_with_mechanism"
            else:
                if nu_max < 0:
                    classification = "homogeneously_auxetic"
                elif nu_min < 0:
                    classification = "directionally_auxetic"
                else:
                    classification = "non_auxetic"

    return {
        "name": graph.name,
        "family": graph.family,
        "keyword": graph.keyword,
        "n_rows": graph.n_rows,
        "n_cols": graph.n_cols,
        "n_vertices": len(graph.vertices),
        "n_edges": len(graph.edges),
        "C": C,
        "C11": float(C[0, 0]),
        "C22": float(C[1, 1]),
        "C12": float(C[0, 1]),
        "C66": float(C[2, 2]),
        "C16": float(C[0, 2]),
        "C26": float(C[1, 2]),
        "C_eigvals": h["C_eigvals"],
        "C_eig_max": float(h["C_eigvals"][0]),
        "C_eig_min": float(h["C_eigvals"][-1]),
        "nu_min": nu_min,
        "nu_max": nu_max,
        "anisotropy": anisotropy,
        "classification": classification,
        "n_internal_mechanisms": h["n_internal_mechanisms"],
        "n_macro_soft_modes": n_macro_soft,
        "cell_area": h["A"],
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _format_C(C: np.ndarray) -> str:
    return ("[[{:+.4f} {:+.4f} {:+.4f}]\n"
            " [{:+.4f} {:+.4f} {:+.4f}]\n"
            " [{:+.4f} {:+.4f} {:+.4f}]]").format(
        C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2],
        C[2,0], C[2,1], C[2,2])


def main():
    ap = argparse.ArgumentParser(
        description="Mechanical homogenization of TesseLace bobbin lace "
                    "ground graphs. Computes effective elastic tensor and "
                    "Poisson ratio profile.")
    ap.add_argument("--catalog", "-c", default="tesselace_catalog",
                    help="catalog directory (default: tesselace_catalog)")
    ap.add_argument("--single", "-s", default=None,
                    help="analyze only this single .txt file (debug)")
    ap.add_argument("--output", "-o", default="mechanics_results.csv",
                    help="output CSV (default: mechanics_results.csv)")
    ap.add_argument("--k-angular", type=float, default=0.01,
                    help="bending/angular stiffness ratio (default: 0.01). "
                         "Set 0 for pure pin-jointed framework (will "
                         "show many grounds as mechanism-floppy).")
    args = ap.parse_args()

    if args.single:
        g = parse_file(args.single)
        if g is None:
            print(f"Failed to parse {args.single}")
            return 1
        result = analyze(g, k_angular=args.k_angular)
        print(f"Ground: {result['family']}/{result['name']}")
        print(f"  size: {result['n_rows']} x {result['n_cols']}")
        print(f"  vertices: {result['n_vertices']}, edges: {result['n_edges']}")
        print(f"  cell area: {result['cell_area']:.4f}")
        print(f"  internal mechanism modes: {result['n_internal_mechanisms']}")
        print(f"  macroscopic soft modes:   {result['n_macro_soft_modes']}")
        print(f"  C eigenvalues (descending): {result['C_eigvals']}")
        print(f"  effective C (Voigt):")
        print(_format_C(result['C']))
        print(f"  C_11 = {result['C11']:+.4f}")
        print(f"  C_22 = {result['C22']:+.4f}")
        print(f"  C_12 = {result['C12']:+.4f}")
        print(f"  C_66 = {result['C66']:+.4f}")
        if not np.isnan(result['nu_min']):
            print(f"  Poisson:")
            print(f"    nu_min = {result['nu_min']:+.4f}")
            print(f"    nu_max = {result['nu_max']:+.4f}")
            print(f"    anisotropy = {result['anisotropy']:.4f}")
        else:
            print(f"  Poisson: undefined (fully floppy)")
        print(f"  classification: {result['classification']}")
        return 0

    # Catalog mode
    manifest = os.path.join(args.catalog, "manifest.csv")
    if not os.path.exists(manifest):
        print(f"manifest not found at {manifest}")
        print("Run scrape_tesselace.py first.")
        return 1

    print(f"Parsing catalog from {manifest}...", flush=True)
    graphs = parse_manifest(manifest, base_dir="")
    print(f"  parsed {len(graphs)} grounds", flush=True)
    print()

    print(f"Running mechanical analysis on each ground...", flush=True)
    results = []
    t0 = time.time()
    for i, g in enumerate(graphs, 1):
        try:
            r = analyze(g, k_angular=args.k_angular)
            results.append(r)
        except Exception as exc:
            print(f"  failed on {g.family}/{g.name}: {exc}", flush=True)
        if i % 25 == 0 or i == len(graphs):
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  {i}/{len(graphs)} ({elapsed:.1f}s, {rate:.1f}/s)",
                  flush=True)

    # Summary
    print()
    print(f"Analyzed {len(results)} grounds successfully.")
    classes: Dict[str, int] = {}
    for r in results:
        c = r["classification"]
        classes[c] = classes.get(c, 0) + 1
    print(f"  Classification counts:")
    for c in sorted(classes):
        print(f"    {c:<45s} {classes[c]}")
    n_with_macro_softmodes = sum(1 for r in results if r["n_macro_soft_modes"] > 0)
    print(f"  Grounds with macroscopic soft modes: {n_with_macro_softmodes}")
    print()

    # Sort by nu_min ascending (most-auxetic first) and show top 10
    valid_results = [r for r in results if not np.isnan(r["nu_min"])]
    sorted_by_min = sorted(valid_results, key=lambda r: r["nu_min"])
    print("Top 10 most-auxetic grounds (by nu_min, where defined):")
    print(f"  {'family':<12s} {'name':<14s} {'nu_min':>9s} {'nu_max':>9s} "
          f"{'soft':>4s} {'class':>40s}")
    for r in sorted_by_min[:10]:
        print(f"  {r['family']:<12s} {r['name']:<14s} "
              f"{r['nu_min']:+9.4f} {r['nu_max']:+9.4f} "
              f"{r['n_macro_soft_modes']:>4d} {r['classification']:>40s}")

    # CSV
    field_order = [
        "family", "name", "keyword", "n_rows", "n_cols",
        "n_vertices", "n_edges", "cell_area",
        "C11", "C22", "C12", "C66", "C16", "C26",
        "C_eig_max", "C_eig_min",
        "nu_min", "nu_max", "anisotropy", "classification",
        "n_internal_mechanisms", "n_macro_soft_modes",
    ]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
