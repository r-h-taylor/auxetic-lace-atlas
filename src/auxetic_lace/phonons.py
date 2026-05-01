"""
phonons.py
==========

Phonon spectrum computation for a periodic lace ground.

Treats each LaceGraph as a 2D ball-and-spring network: unit point mass at
every vertex; the same axial + angular springs as `mechanics.py`. The
dynamical matrix D(k) is the Bloch-summed stiffness with phase factors on
inter-cell hopping. Eigenvalues of D(k) are omega^2(k); the lowest two
bands are the acoustic branches (going to zero at Gamma by translational
symmetry).

Public API:

    omegas_along_path(graph, k_path, k_angular=0.01) -> ndarray (M, 2N)
        Diagonalize D(k) at each k in k_path; return sorted omegas.

    high_symmetry_path(graph, npts_per_seg=60) -> (k_path, distances, markers)
        Standard Gamma -> X -> M -> Gamma path on the rectangular BZ.

    bz_grid(graph, n_per_axis=16) -> ndarray (n_per_axis^2, 2)
        Uniform grid sampling the first Brillouin zone.

    dispersion_features(graph, k_angular=0.01, n_grid=16) -> dict
        22 scalar phonon descriptors (acoustic min/max/mean over BZ;
        acoustic-branch slopes near Gamma along x and y; optical-band
        gap; soft-mode location). Suitable for atlas record inclusion
        and downstream regression / surrogate training.

Conventions: the LaceGraph stores integer (col, row) vertices; we treat
(col, row) as planar (x, y) for the dispersion calculation. Inter-cell
shifts are computed via the lattice basis from `mechanics.default_lattice_vectors`.

Cross-checks (asserted in tests/test_phonons.py):
  - D(0) has exactly 2 zero eigenvalues (translational symmetry).
  - Acoustic-velocity slopes near Gamma match the static elastic tensor
    from `mechanics.assemble_stiffness` to ~1% on canonical test grounds.

Spring-only: this module mirrors `mechanics.py`, not `mechanics_beam.py`.
The beam-mode dynamical matrix has additional rotational DOFs that we do
not include here; for static elasticity the spring + angular regularization
already captures the relevant auxetic mechanisms cheaply.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .parse_to_graph import LaceGraph
from .mechanics import default_lattice_vectors, edge_geometry


# =============================================================================
# Per-edge stiffness blocks (cached) for fast Bloch summation
# =============================================================================

def _precompute_blocks(graph: LaceGraph,
                       L: np.ndarray,
                       k_per_len: float = 1.0,
                       k_angular: float = 0.01,
                       ) -> Dict:
    """Build per-edge cartesian stiffness blocks once. Returns a dict with
    arrays suitable for D(k) assembly across many k.

    Each edge contributes a 2x2 cartesian block K_e = k_ax * nn + k_an * pp,
    where nn = unit unit^T and pp = perp perp^T. The wrap shift R_cart =
    (wrap_col * a1) + (wrap_row * a2) carries the Bloch phase.
    """
    N = len(graph.vertices)
    n_e = len(graph.edges)
    Kblocks = np.zeros((n_e, 2, 2))
    src_idx = np.zeros(n_e, dtype=int)
    dst_idx = np.zeros(n_e, dtype=int)
    R_cart = np.zeros((n_e, 2))

    a1 = L @ np.array([graph.n_cols, 0.0])    # full row period in cartesian
    a2 = L @ np.array([0.0, graph.n_rows])    # full col period in cartesian

    for e_i in range(n_e):
        e = graph.edges[e_i]
        _delta, length, unit = edge_geometry(graph, e_i, L)
        if length < 1e-12:
            continue
        k_ax = k_per_len / length
        nn = np.outer(unit, unit)
        block = k_ax * nn
        if k_angular > 0.0:
            perp = np.array([-unit[1], unit[0]])
            pp = np.outer(perp, perp)
            block = block + (k_angular / length) * pp
        Kblocks[e_i] = block
        src_idx[e_i] = e.src_idx
        dst_idx[e_i] = e.dst_idx
        R_cart[e_i] = e.wrap[0] * a1 + e.wrap[1] * a2

    return {
        "N": N,
        "n_e": n_e,
        "Kblocks": Kblocks,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "R_cart": R_cart,
    }


def _D_at_k(blocks: Dict, k_vec: np.ndarray) -> np.ndarray:
    """Hermitian dynamical matrix D(k), unit vertex mass."""
    N = blocks["N"]
    Kblocks = blocks["Kblocks"]
    src_idx = blocks["src_idx"]
    dst_idx = blocks["dst_idx"]
    R_cart = blocks["R_cart"]

    D = np.zeros((2 * N, 2 * N), dtype=complex)
    phases = np.exp(1j * (R_cart @ k_vec))
    for e_i in range(blocks["n_e"]):
        Kb = Kblocks[e_i]
        i, j = 2 * src_idx[e_i], 2 * dst_idx[e_i]
        D[i:i+2, i:i+2] += Kb
        D[j:j+2, j:j+2] += Kb
        D[i:i+2, j:j+2] -= Kb * np.conj(phases[e_i])
        D[j:j+2, i:i+2] -= Kb * phases[e_i]
    # Hermitize against round-off
    D = 0.5 * (D + D.conj().T)
    return D


def _omegas(blocks: Dict, k_vec: np.ndarray) -> np.ndarray:
    """Sorted ascending omega = sqrt(max(0, eig)) at this k."""
    D = _D_at_k(blocks, k_vec)
    w = np.linalg.eigvalsh(D)
    return np.sqrt(np.maximum(w, 0.0))


# =============================================================================
# Public: dispersion along a k-path
# =============================================================================

def omegas_along_path(graph: LaceGraph,
                       k_path: np.ndarray,
                       k_angular: float = 0.01,
                       k_per_len: float = 1.0,
                       L: np.ndarray = None,
                       ) -> np.ndarray:
    """Evaluate omega(k) along the given (M, 2) array of k-points.
    Returns omegas of shape (M, 2N), each row sorted ascending."""
    if L is None:
        L = default_lattice_vectors()
    blocks = _precompute_blocks(graph, L, k_per_len, k_angular)
    M = k_path.shape[0]
    N = blocks["N"]
    out = np.zeros((M, 2 * N))
    for m in range(M):
        out[m] = _omegas(blocks, k_path[m])
    return out


# =============================================================================
# Brillouin zone helpers
# =============================================================================

def reciprocal_basis(B: np.ndarray) -> np.ndarray:
    """2D reciprocal-lattice basis. B has rows a1, a2; returns rows b1, b2
    with ai . bj = 2*pi*delta_ij."""
    a1, a2 = B[0], B[1]
    rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    a1p = rot90 @ a1
    a2p = rot90 @ a2
    b1 = 2 * np.pi * a2p / float(np.dot(a1, a2p))
    b2 = 2 * np.pi * a1p / float(np.dot(a2, a1p))
    return np.vstack([b1, b2])


def _direct_basis(graph: LaceGraph, L: np.ndarray) -> np.ndarray:
    """Cartesian basis vectors of the unit cell as rows."""
    a1 = L @ np.array([graph.n_cols, 0.0])
    a2 = L @ np.array([0.0, graph.n_rows])
    return np.vstack([a1, a2])


def high_symmetry_path(graph: LaceGraph,
                        npts_per_seg: int = 60,
                        L: np.ndarray = None,
                        ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build a Gamma -> X -> M -> Gamma path on the rectangular BZ.

      Gamma = (0, 0)
      X     = b1 / 2
      M     = (b1 + b2) / 2

    Returns (k_path, distances, markers) where:
      k_path     : (M, 2) array of cartesian k-points
      distances  : (M,) cumulative euclidean path length, for plotting
      markers    : indices of the high-symmetry vertices in k_path
    """
    if L is None:
        L = default_lattice_vectors()
    B = _direct_basis(graph, L)
    G = reciprocal_basis(B)
    Gamma = np.zeros(2)
    X = G[0] / 2
    M = (G[0] + G[1]) / 2
    segments = [(Gamma, X), (X, M), (M, Gamma)]

    k_list, dist_list, markers = [], [], [0]
    cumulative = 0.0
    last_seg = len(segments) - 1
    for s_i, (k0, k1) in enumerate(segments):
        seg_len = float(np.linalg.norm(k1 - k0))
        if s_i == last_seg:
            ts = np.linspace(0.0, 1.0, npts_per_seg + 1, endpoint=True)
        else:
            ts = np.linspace(0.0, 1.0, npts_per_seg, endpoint=False)
        for t in ts:
            k = k0 + t * (k1 - k0)
            k_list.append(k)
            dist_list.append(cumulative + t * seg_len)
        cumulative += seg_len
        markers.append(len(k_list) - 1)
    return np.array(k_list), np.array(dist_list), markers


def bz_grid(graph: LaceGraph,
             n_per_axis: int = 16,
             L: np.ndarray = None,
             ) -> np.ndarray:
    """Uniform sampling of the first BZ; returns shape (n_per_axis**2, 2)."""
    if L is None:
        L = default_lattice_vectors()
    G = reciprocal_basis(_direct_basis(graph, L))
    fracs = (np.arange(n_per_axis) + 0.5) / n_per_axis - 0.5  # in (-0.5, 0.5)
    f1, f2 = np.meshgrid(fracs, fracs, indexing="ij")
    flat = np.stack([f1.ravel(), f2.ravel()], axis=1)
    return flat @ G


# =============================================================================
# Public: per-ground scalar descriptors for the atlas record
# =============================================================================

def dispersion_features(graph: LaceGraph,
                         k_angular: float = 0.01,
                         k_per_len: float = 1.0,
                         n_grid: int = 16,
                         L: np.ndarray = None,
                         ) -> Dict[str, float]:
    """22 scalar phonon descriptors per ground.

    Keys (all keys present, NaN where undefined):

      n_bands              total bands (= 2 * n_vertices)
      acoustic_min         lowest band's BZ-grid minimum (= 0 at Gamma)
      acoustic_max         max of lowest two bands over BZ grid
      acoustic_mean        mean of lowest two bands over BZ grid
      flat_acoustic_score  1 - (acoustic_max - acoustic_min)/acoustic_max
      softmode_omega       lowest band's BZ-grid min away from Gamma
      softmode_kx, softmode_ky : that minimum's location
      v_x_lo, v_x_hi       acoustic-branch slopes near Gamma along +x
      v_y_lo, v_y_hi       same along +y
      v_min, v_max         min/max of those four
      v_anisotropy         (v_max - v_min) / v_max
      v_ratio_xy           v_y_hi / v_x_hi
      optical_min_gamma    omega of lowest optical band at Gamma
      optical_min_X        same at X
      optical_min_M        same at M
      optical_min_BZ       lowest optical band over BZ grid
      acoustic_optical_gap optical_min_BZ - acoustic_max  (>0 = clean gap)
      has_gap              1 if positive gap, else 0

    For grounds with only 2 bands (single-vertex cells), all optical_* are NaN
    and has_gap = 0.
    """
    if L is None:
        L = default_lattice_vectors()
    blocks = _precompute_blocks(graph, L, k_per_len, k_angular)
    n_bands = 2 * blocks["N"]

    # BZ grid
    G = reciprocal_basis(_direct_basis(graph, L))
    fracs = (np.arange(n_grid) + 0.5) / n_grid - 0.5
    f1, f2 = np.meshgrid(fracs, fracs, indexing="ij")
    flat = np.stack([f1.ravel(), f2.ravel()], axis=1)
    k_grid = flat @ G
    omegas_grid = np.zeros((k_grid.shape[0], n_bands))
    for m in range(k_grid.shape[0]):
        omegas_grid[m] = _omegas(blocks, k_grid[m])

    acoustic = omegas_grid[:, :2]
    acoustic_min = float(acoustic.min())
    acoustic_max = float(acoustic.max())
    acoustic_mean = float(acoustic.mean())

    # Soft-mode: lowest band at the k farthest from Gamma where it dips lowest
    k_norms = np.linalg.norm(k_grid, axis=1)
    k_norm_med = float(np.median(k_norms))
    far_mask = k_norms > 0.2 * k_norm_med
    if np.any(far_mask):
        far_lowest = omegas_grid[far_mask, 0]
        idx_min = int(np.argmin(far_lowest))
        softmode_omega = float(far_lowest[idx_min])
        sm_k = k_grid[far_mask][idx_min]
        softmode_kx = float(sm_k[0])
        softmode_ky = float(sm_k[1])
    else:
        softmode_omega = float("nan")
        softmode_kx = float("nan")
        softmode_ky = float("nan")

    flat_acoustic_score = 1.0 - (acoustic_max - acoustic_min) / max(1e-12, acoustic_max)

    # Acoustic velocities at small k near Gamma
    eps = 1e-4
    om_x = _omegas(blocks, np.array([eps, 0.0]))
    om_y = _omegas(blocks, np.array([0.0, eps]))
    v_x_lo, v_x_hi = float(om_x[0]) / eps, float(om_x[1]) / eps
    v_y_lo, v_y_hi = float(om_y[0]) / eps, float(om_y[1]) / eps
    vs = np.array([v_x_lo, v_x_hi, v_y_lo, v_y_hi])
    v_min = float(vs.min())
    v_max = float(vs.max())
    v_anisotropy = (v_max - v_min) / max(1e-12, v_max)
    v_ratio_xy = v_y_hi / max(1e-12, v_x_hi)

    # Optical band stats (only meaningful if n_bands >= 3)
    if n_bands >= 3:
        om_gamma = _omegas(blocks, np.zeros(2))
        om_X = _omegas(blocks, G[0] / 2)
        om_M = _omegas(blocks, (G[0] + G[1]) / 2)
        optical_min_gamma = float(om_gamma[2])
        optical_min_X = float(om_X[2])
        optical_min_M = float(om_M[2])
        optical_min_BZ = float(omegas_grid[:, 2].min())
        gap = optical_min_BZ - acoustic_max
        has_gap = int(gap > 0)
    else:
        optical_min_gamma = float("nan")
        optical_min_X = float("nan")
        optical_min_M = float("nan")
        optical_min_BZ = float("nan")
        gap = float("nan")
        has_gap = 0

    raw = {
        "n_bands": n_bands,
        "acoustic_min": acoustic_min,
        "acoustic_max": acoustic_max,
        "acoustic_mean": acoustic_mean,
        "flat_acoustic_score": flat_acoustic_score,
        "softmode_omega": softmode_omega,
        "softmode_kx": softmode_kx,
        "softmode_ky": softmode_ky,
        "v_x_lo": v_x_lo,
        "v_x_hi": v_x_hi,
        "v_y_lo": v_y_lo,
        "v_y_hi": v_y_hi,
        "v_min": v_min,
        "v_max": v_max,
        "v_anisotropy": v_anisotropy,
        "v_ratio_xy": v_ratio_xy,
        "optical_min_gamma": optical_min_gamma,
        "optical_min_X": optical_min_X,
        "optical_min_M": optical_min_M,
        "optical_min_BZ": optical_min_BZ,
        "acoustic_optical_gap": gap,
        "has_gap": has_gap,
    }
    # Replace NaN with None so json.dump emits null (valid JSON; NaN is not).
    import math as _math
    return {k: (None if isinstance(v, float) and _math.isnan(v) else v)
            for k, v in raw.items()}


# =============================================================================
# Sanity-check helpers (used by tests, not part of the atlas pipeline)
# =============================================================================

def gamma_zero_modes(graph: LaceGraph,
                      k_angular: float = 0.01,
                      k_per_len: float = 1.0,
                      tol: float = 1e-6,
                      L: np.ndarray = None,
                      ) -> int:
    """Count eigenvalues of D(0) below tol * max-eig. Should be exactly 2
    (the two 2D translations). Extras would indicate genuine zero-energy
    mechanisms (floppy modes)."""
    if L is None:
        L = default_lattice_vectors()
    blocks = _precompute_blocks(graph, L, k_per_len, k_angular)
    D = _D_at_k(blocks, np.zeros(2))
    w = np.linalg.eigvalsh(D)
    s = max(1.0, float(np.max(np.abs(w))))
    return int(np.sum(w < tol * s))
