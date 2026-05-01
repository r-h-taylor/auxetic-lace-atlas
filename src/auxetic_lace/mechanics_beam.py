"""
mechanics_beam.py
=================

Periodic Euler-Bernoulli beam-frame mechanical homogenization for
TesseLace ground graphs.

This is a refinement of `mechanics.py` (axial spring + perpendicular
regularization) to a proper frame-element model:

  - Vertices = rigid junctions with 3 DOFs (ux, uy, theta)
  - Edges    = Euler-Bernoulli beam elements with axial stiffness EA/L
               and flexural stiffness EI/L^3
  - Periodic = wrap vectors encode toroidal connectivity

Each beam element couples 6 DOFs (3 at each end). The standard
local-frame stiffness matrix is

  k_local =
    [ EA/L     0           0          -EA/L     0           0          ]
    [ 0        12 EI/L^3   6 EI/L^2    0       -12 EI/L^3   6 EI/L^2   ]
    [ 0        6 EI/L^2    4 EI/L      0       -6 EI/L^2    2 EI/L     ]
    [ -EA/L    0           0           EA/L     0           0          ]
    [ 0       -12 EI/L^3  -6 EI/L^2    0        12 EI/L^3  -6 EI/L^2   ]
    [ 0        6 EI/L^2    2 EI/L      0       -6 EI/L^2    4 EI/L     ]

with DOF order (u1, v1, theta1, u2, v2, theta2) where u is along the
beam, v is transverse, theta is the rotation. We then transform to
global coordinates via the rotation matrix R(alpha) where alpha is
the beam's orientation in the lab frame.

The beam aspect ratio r = L/h (length over cross-section thickness)
parameterizes the EI/EA ratio:
  - For a rectangular cross-section of height h and width 1:
    EI/EA = h^2 / 12. So EI/(EA*L^2) = (h/L)^2 / 12 = 1/(12 r^2).
  - For a circular cross-section of diameter h:
    EI/EA = h^2 / 16. So EI/(EA*L^2) = 1/(16 r^2).
  - Setting r -> infinity recovers the axial-spring (pin-jointed) limit.
  - Setting r ~ 5-10 corresponds to chunky 3D-printed rods where
    bending and stretching stiffnesses are comparable.

Outputs same as mechanics.py: 2D effective elastic tensor C_ij,
nu(theta), nu_min, nu_max, classification.

Author: clean-room implementation, Apr 2026.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .parse_to_graph import LaceGraph, parse_file, parse_manifest
from .mechanics import (
    default_lattice_vectors, vertex_position, cell_vectors, cell_area,
    edge_geometry, voigt_to_tensor, rotate_voigt, poisson_ratio_at_angle,
    poisson_profile,
)


# =====================================================================
# Beam-element stiffness assembly
# =====================================================================

def beam_element_stiffness_local(L: float, EA: float, EI: float
                                  ) -> np.ndarray:
    """6x6 Euler-Bernoulli beam stiffness in local (beam-aligned) frame.

    DOF order: [u1, v1, theta1, u2, v2, theta2] where u is along the
    beam, v is transverse, theta is the cross-section rotation.
    """
    a = EA / L
    b = 12.0 * EI / L**3
    c = 6.0 * EI / L**2
    d = 4.0 * EI / L
    e = 2.0 * EI / L
    return np.array([
        [ a,  0,  0, -a,  0,  0],
        [ 0,  b,  c,  0, -b,  c],
        [ 0,  c,  d,  0, -c,  e],
        [-a,  0,  0,  a,  0,  0],
        [ 0, -b, -c,  0,  b, -c],
        [ 0,  c,  e,  0, -c,  d],
    ])


def beam_rotation_matrix(unit_x: float, unit_y: float) -> np.ndarray:
    """6x6 rotation matrix that maps local-frame DOFs to global-frame
    DOFs. The local x-axis is the beam axis (unit vector
    `(unit_x, unit_y)` in global coords); local y is perpendicular
    (`(-unit_y, unit_x)`); theta is the same in both frames.
    """
    cs = unit_x
    sn = unit_y
    R = np.zeros((6, 6))
    # Endpoint 1
    R[0, 0] = cs;  R[0, 1] = sn
    R[1, 0] = -sn; R[1, 1] = cs
    R[2, 2] = 1.0
    # Endpoint 2
    R[3, 3] = cs;  R[3, 4] = sn
    R[4, 3] = -sn; R[4, 4] = cs
    R[5, 5] = 1.0
    return R


def assemble_stiffness_beam(graph: LaceGraph,
                             L_lattice: np.ndarray = None,
                             EA: float = 1.0,
                             aspect_ratio: float = 10.0,
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble the periodic stiffness matrix and strain-coupling
    matrix for a 2D Euler-Bernoulli frame.

    DOFs per vertex: (ux, uy, theta), so total DOFs = 3 * N.
    Bond elongation under macro strain epsilon plus internal perturbation:
        relative_disp_global = u(dst) + epsilon . wrap_shift - u(src)
        relative_disp_local  = R_2x2 . relative_disp_global
        local axial = (u2_local - u1_local)
        local transverse + theta1, theta2 -> bending energy

    The cross-section parameter `aspect_ratio = L_bar / h` controls
    EI/EA via EI = EA * L_bar^2 / (12 * aspect_ratio^2) for a rectangular
    cross-section of height h (assumes width = 1, modulus E factored
    into EA).

    `EA` is in the same units as the spring stiffness in mechanics.py
    (default EA = 1.0, so direct comparisons can be made).

    Returns (K, G, H, A) where K is the 3N x 3N stiffness, G is the
    coupling to macro strain (Voigt 3-vector), H is the macro-strain
    quadratic, and A is the cell area.
    """
    if L_lattice is None:
        L_lattice = default_lattice_vectors()

    N = len(graph.vertices)
    K = np.zeros((3 * N, 3 * N))
    G = np.zeros((3 * N, 3))      # u <-> eps coupling, eps Voigt [exx, eyy, 2exy]
    H = np.zeros((3, 3))           # eps-eps quadratic
    A = cell_area(graph, L_lattice)

    for e_idx in range(len(graph.edges)):
        e = graph.edges[e_idx]
        delta, L_bar, unit = edge_geometry(graph, e_idx, L_lattice)
        if L_bar < 1e-12:
            continue
        # EI from aspect ratio: rectangular cross-section h x 1, where
        # h = L_bar / aspect_ratio. EI = E * h^3 / 12 with E*1*h = EA
        # -> EI/EA = h^2/12 -> EI = EA * h^2/12 = EA * L_bar^2 / (12 * AR^2)
        h = L_bar / aspect_ratio
        EI = EA * h**2 / 12.0

        # Local-frame stiffness
        k_local = beam_element_stiffness_local(L_bar, EA, EI)
        # Rotation to global frame
        R = beam_rotation_matrix(unit[0], unit[1])
        # Global-frame stiffness for this beam, 6x6
        k_global = R.T @ k_local @ R

        # Wrap shift in cartesian coords
        wrap_shift = L_lattice @ np.array(
            [e.wrap[0] * graph.n_cols, e.wrap[1] * graph.n_rows], dtype=float)
        wx, wy = wrap_shift[0], wrap_shift[1]

        # B_eps maps eps_voigt to global displacement at endpoint 2
        # (relative to endpoint 1) via wrap_shift contribution:
        # delta_u_from_strain = epsilon . wrap_shift = [wx*exx + wy*0.5*2exy, wy*eyy + wx*0.5*2exy]
        # In Voigt vector form with eps = [exx, eyy, 2exy]:
        B_eps = np.array([
            [wx, 0.0, 0.5 * wy],   # delta u_x
            [0.0, wy, 0.5 * wx],   # delta u_y
            [0.0, 0.0, 0.0],       # delta theta = 0 (affine has no rotation jump)
        ])

        # The bond's 6-vector "elongation" (in DOF order) due to internal u:
        # d = [-I_3, +I_3] @ [u1; u2]. Energy = (1/2) d^T k_global d.
        # Adding wrap-shift contribution: d_total = d_internal + d_wrap
        # where d_wrap has zeros at endpoint 1 and B_eps . eps at endpoint 2.

        # Index slices for the 6 DOFs of this edge
        i1 = 3 * e.src_idx
        i2 = 3 * e.dst_idx

        # Build the (6, 3N) "endpoint extraction matrix" implicitly.
        # K_global has block structure [k11 k12; k21 k22] where each block is 3x3.
        k11 = k_global[0:3, 0:3]
        k12 = k_global[0:3, 3:6]
        k21 = k_global[3:6, 0:3]
        k22 = k_global[3:6, 3:6]

        # Internal DOF coupling
        K[i1:i1+3, i1:i1+3] += k11
        K[i1:i1+3, i2:i2+3] += k12
        K[i2:i2+3, i1:i1+3] += k21
        K[i2:i2+3, i2:i2+3] += k22

        # u-eps coupling: d^T k_global d expanded with d = [-u1; u2 + B_eps eps]
        # = u^T K u  +  2 * eps^T B_eps^T (k22 + k21? actually let's redo carefully)
        #
        # Let v = [-u1; u2 + d_wrap_2]. Then v^T k_global v has cross terms.
        # v^T k_global v = u^T K u + 2 d_wrap_2^T (k22 u2 - k21 u1) wait let me be careful.
        # Actually v_1 = -u1 means v_1 stands for "endpoint 1's displacement contribution",
        # but the displacements at the two endpoints relative to the beam ARE u1 and u2 +
        # wrap. Let me re-derive.
        #
        # Total displacement at endpoint 1 = u1 (in global frame).
        # Total displacement at endpoint 2 = u2 + epsilon . wrap_shift (in global
        # frame, since we're imposing macro affine deformation on the wrap).
        # Strain energy = (1/2) [u1; u2 + Eps] k_global [u1; u2 + Eps]
        # where Eps = (B_eps eps, 0_theta) = a 3-vector with first 2 entries from B_eps
        # and 0 for theta.
        #
        # Expanding:
        #   = (1/2) [u1 u2] K [u1; u2] + [u2] (k22 + k22^T)/2 * Eps (cross term)
        #   + ... wait, let me be more careful.

        # Let's expand directly:
        # Energy = (1/2) [u1; u2+Eps]^T [k11 k12; k21 k22] [u1; u2+Eps]
        #        = (1/2) u1^T k11 u1 + u1^T k12 u2 + u1^T k12 Eps
        #          + (1/2) u2^T k22 u2 + u2^T k22 Eps + (1/2) Eps^T k22 Eps
        # Wait that's not symmetric in u1, u2. Let me redo:
        # [a b]^T [k11 k12; k21 k22] [a b] = a^T k11 a + a^T k12 b + b^T k21 a + b^T k22 b
        # If K is symmetric, k12 = k21^T, so this = a^T k11 a + 2 a^T k12 b + b^T k22 b.
        # So with a = u1, b = u2 + Eps:
        # 2 * Energy = u1^T k11 u1 + 2 u1^T k12 (u2 + Eps) + (u2 + Eps)^T k22 (u2 + Eps)
        #            = u1^T k11 u1 + 2 u1^T k12 u2 + u2^T k22 u2  [internal K terms]
        #            + 2 u1^T k12 Eps + 2 u2^T k22 Eps             [G terms]
        #            + Eps^T k22 Eps                                 [H term]

        # So the strain coupling for this edge is:
        # G_edge_u1 = k12 . Eps  (added to G[i1:i1+3, :])
        # G_edge_u2 = k22 . Eps  (added to G[i2:i2+3, :])
        # H_edge   = B_eps^T (only first 2 rows used) k22[0:2, 0:2] B_eps
        #            (since Eps's theta is 0)

        # Build the 3x3 "Eps from eps" matrix: B_eps full (3x3, with zero theta row)
        # Already built as B_eps above. Note B_eps is 3x3 here, with the 3rd row
        # being zeros.
        G[i1:i1+3, :] += k12 @ B_eps
        G[i2:i2+3, :] += k22 @ B_eps
        H += B_eps.T @ k22 @ B_eps

    return K, G, H, A


# =====================================================================
# Homogenization
# =====================================================================

def fix_translation_beam(K: np.ndarray, G: np.ndarray, n_verts: int
                          ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Pin translation DOFs of vertex 0 to remove translation null space.
    Rotation DOFs are NOT pinned globally — they need to be free at every
    vertex for the frame to deform correctly.

    Returns (K_reduced, G_reduced, free_idx).
    """
    # Pin DOFs 0 (ux of vertex 0) and 1 (uy of vertex 0). Keep 2 (theta)
    # of vertex 0 free.
    n_dofs = 3 * n_verts
    fixed = [0, 1]
    free_idx = [i for i in range(n_dofs) if i not in fixed]
    K_red = K[np.ix_(free_idx, free_idx)]
    G_red = G[free_idx, :]
    return K_red, G_red, free_idx


def homogenize_beam(graph: LaceGraph,
                    L_lattice: Optional[np.ndarray] = None,
                    EA: float = 1.0,
                    aspect_ratio: float = 10.0,
                    rcond: float = 1e-12,
                    ) -> Tuple[np.ndarray, float, np.ndarray]:
    """Solve the periodic-frame homogenization problem.

    Returns (C_voigt, A, internal_disp) where:
      - C_voigt is the 3x3 effective elastic tensor in Voigt notation
      - A is cell area
      - internal_disp[k] is the internal DOF response to strain mode k
        (3*N entries: ux1, uy1, theta1, ux2, uy2, theta2, ...).
    """
    if L_lattice is None:
        L_lattice = default_lattice_vectors()
    N = len(graph.vertices)

    K, G, H, A = assemble_stiffness_beam(graph, L_lattice, EA, aspect_ratio)
    K_red, G_red, free_idx = fix_translation_beam(K, G, N)

    # Solve K_red u = -G_red eps for each strain mode
    # The stiffness can still have null modes (e.g. global rotation if
    # angular springs don't couple it), so use lstsq with rcond.
    u_red, _, _, _ = np.linalg.lstsq(K_red, -G_red, rcond=rcond)

    # Embed back into full DOF vector (with vertex 0 ux, uy = 0)
    u_full = np.zeros((3 * N, 3))
    u_full[free_idx, :] = u_red

    # Effective stiffness: C = (H + G^T u_full) / A
    C_voigt = (H + G.T @ u_full) / A

    # Symmetrize (numerical noise)
    C_voigt = 0.5 * (C_voigt + C_voigt.T)

    return C_voigt, A, u_full


# =====================================================================
# Analyze and classify
# =====================================================================

@dataclass
class BeamAnalysisResult:
    name: str
    family: str
    n_rows: int
    n_cols: int
    n_vertices: int
    n_edges: int
    aspect_ratio: float
    cell_area: float
    C_voigt: np.ndarray
    nu_min: float
    nu_max: float
    nu_min_angle: float
    nu_max_angle: float
    anisotropy: float
    classification: str  # 'non_auxetic' | 'directionally_auxetic' | 'homogeneously_auxetic'


def analyze_beam(graph: LaceGraph,
                  name: str = "",
                  family: str = "",
                  EA: float = 1.0,
                  aspect_ratio: float = 10.0,
                  ) -> BeamAnalysisResult:
    """Compute homogenized elastic tensor and Poisson profile."""
    L_lattice = default_lattice_vectors()
    C, A, _ = homogenize_beam(graph, L_lattice, EA, aspect_ratio)

    # Sweep theta over [0, pi) on a 181-point grid for nu_min/max
    thetas = np.linspace(0.0, np.pi, 181, endpoint=False)
    nus = np.array([poisson_ratio_at_angle(C, t) for t in thetas])
    finite = np.isfinite(nus)
    if not finite.any():
        nu_min = nu_max = float('nan')
        nu_min_angle = nu_max_angle = 0.0
    else:
        nu_min_idx = np.argmin(nus[finite])
        nu_max_idx = np.argmax(nus[finite])
        valid_thetas = thetas[finite]
        valid_nus = nus[finite]
        nu_min = float(valid_nus[nu_min_idx])
        nu_max = float(valid_nus[nu_max_idx])
        nu_min_angle = float(valid_thetas[nu_min_idx])
        nu_max_angle = float(valid_thetas[nu_max_idx])

    if nu_max < 0:
        classification = "homogeneously_auxetic"
    elif nu_min < 0:
        classification = "directionally_auxetic"
    else:
        classification = "non_auxetic"

    return BeamAnalysisResult(
        name=name, family=family,
        n_rows=graph.n_rows, n_cols=graph.n_cols,
        n_vertices=len(graph.vertices), n_edges=len(graph.edges),
        aspect_ratio=aspect_ratio,
        cell_area=A,
        C_voigt=C,
        nu_min=nu_min, nu_max=nu_max,
        nu_min_angle=nu_min_angle, nu_max_angle=nu_max_angle,
        anisotropy=nu_max - nu_min,
        classification=classification,
    )


# =====================================================================
# Validation: in the high-aspect-ratio limit, beam should approach
# the pin-jointed (axial-only) result of mechanics.py.
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Beam-element mechanics for lace grounds.")
    ap.add_argument("--single", type=str, default=None,
                    help="Path to a single .txt template")
    ap.add_argument("--catalog", type=str, default=None,
                    help="Path to a tesselace_catalog directory")
    ap.add_argument("--aspect-ratio", type=float, default=10.0,
                    help="Beam length / cross-section thickness")
    ap.add_argument("--EA", type=float, default=1.0)
    ap.add_argument("--output", type=str, default=None,
                    help="Output CSV path (for catalog mode)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.single:
        graph = parse_file(args.single)
        result = analyze_beam(graph, name=os.path.basename(args.single),
                                aspect_ratio=args.aspect_ratio, EA=args.EA)
        print(f"\n{result.name}")
        print(f"  family: {result.family}, cell: {result.n_rows}x{result.n_cols}")
        print(f"  vertices: {result.n_vertices}, edges: {result.n_edges}")
        print(f"  aspect ratio: {result.aspect_ratio}")
        print(f"  C (Voigt):\n{result.C_voigt}")
        print(f"  nu_min: {result.nu_min:+.4f} (theta = {np.degrees(result.nu_min_angle):.1f} deg)")
        print(f"  nu_max: {result.nu_max:+.4f}")
        print(f"  classification: {result.classification}")

    elif args.catalog:
        manifest = parse_manifest(os.path.join(args.catalog, "manifest.csv"))
        rows = []
        t0 = time.time()
        for i, entry in enumerate(manifest):
            try:
                graph = parse_file(os.path.join(args.catalog, entry['path']))
                result = analyze_beam(graph, name=entry['name'],
                                       family=entry['family'],
                                       aspect_ratio=args.aspect_ratio,
                                       EA=args.EA)
                rows.append({
                    'family': result.family,
                    'name': result.name,
                    'n_rows': result.n_rows,
                    'n_cols': result.n_cols,
                    'n_vertices': result.n_vertices,
                    'n_edges': result.n_edges,
                    'aspect_ratio': result.aspect_ratio,
                    'cell_area': result.cell_area,
                    'C11': result.C_voigt[0, 0],
                    'C22': result.C_voigt[1, 1],
                    'C12': result.C_voigt[0, 1],
                    'C66': result.C_voigt[2, 2],
                    'C16': result.C_voigt[0, 2],
                    'C26': result.C_voigt[1, 2],
                    'nu_min': result.nu_min,
                    'nu_max': result.nu_max,
                    'classification': result.classification,
                })
                if args.verbose and i % 20 == 0:
                    print(f"  {i}/{len(manifest)}: {entry['name']} -> nu_min={result.nu_min:+.3f}")
            except Exception as exc:
                print(f"  FAIL on {entry['name']}: {exc}")
        elapsed = time.time() - t0
        print(f"\nProcessed {len(rows)} grounds in {elapsed:.1f}s")
        if args.output:
            with open(args.output, "w") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {args.output}")
    else:
        print("Provide --single or --catalog")


if __name__ == "__main__":
    main()
