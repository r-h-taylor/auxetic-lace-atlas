"""
build_atlas.py
==============

Builds the atlas.json data file for the Auxetic Lace Atlas visualizer.

Sweeps both mechanical models over a parameter grid for every ground in
the TesseLace catalog:

  * Spring model (mechanics.py) sweeps angular regularization k_ang
    over {0.0, 0.001, 0.01, 0.1, 1.0}
  * Beam model (mechanics_beam.py) sweeps aspect ratio AR over
    {5, 10, 20, 50, 200}

For each (ground, model, param) combination, we record:
  * The 3x3 elastic tensor C in Voigt notation
  * nu_min, nu_max, classification
  * Cell area

For each ground (one-time, model-independent):
  * Vertex positions (in Cartesian, in unit-cell coords)
  * Edge connectivity (src_idx, dst_idx, wrap)
  * Lattice vectors a1, a2 (default identity)
  * Period parallelogram (n_rows × n_cols)
  * Family, name, hole shapes

The visualizer can compute the response under any applied stress
sigma in JS via:
    epsilon = C_voigt^{-1} . sigma_voigt
    F = I + epsilon  (deformation gradient)
    deformed_position[v] = F . rest_position[v]
plus the internal periodic perturbation, which we ALSO store: for each
(ground, model, param) we save the 3 strain-mode response vectors
u_xx, u_yy, u_xy (one per strain mode), so the deformed configuration
under any applied strain epsilon = (e_xx, e_yy, 2*e_xy) is

    deformed[v] = F . rest[v] + (e_xx * u_xx[v] + e_yy * u_yy[v] + 2*e_xy * u_xy[v])

This is the affine + internal-relaxation decomposition standard in
periodic homogenization.

USAGE:
    # Default: read catalog from ./tesselace_catalog/, write atlas.json
    python3 build_atlas.py

    # Custom paths
    python3 build_atlas.py --catalog tesselace_catalog/ --output atlas.json

    # Smoke test on first N grounds
    python3 build_atlas.py --limit 10

    # Subset of grounds matching a family name
    python3 build_atlas.py --filter-family 3_6

OUTPUT:
    atlas.json structure:
    {
      "metadata": {
        "n_grounds": 321,
        "spring_k_ang_grid": [...],
        "beam_AR_grid": [...],
        "theta_grid_deg": [...],   // angles for nu(theta) profile
        "build_date": "...",
        "attribution": "TesseLace catalog by Veronika Irvine, CC-BY 4.0"
      },
      "grounds": [
        {
          "name": "2x4_86", "family": "3_6",
          "n_rows": 4, "n_cols": 2,
          "vertices": [[col, row], ...],   // unit-cell coords
          "edges": [{"src": 0, "dst": 1, "wrap": [0, 0]}, ...],
          "lattice": [[1, 0], [0, 1]],
          "cell_area": 8.0,
          "spring": {
            "k_ang": [0.0, 0.001, 0.01, 0.1, 1.0],
            "C_voigt": [[[...]], [[...]], ...],  // 5 matrices
            "nu_profile": [[...], [...], ...],   // 5 angle profiles
            "nu_min": [...], "nu_max": [...],
            "classification": [...],
            "u_strain": [[[...]], ...]   // (n_params, 3 modes, 2N) internal disp
          },
          "beam": {
            "AR": [5, 10, 20, 50, 200],
            "C_voigt": [...], ...
            "u_strain": [...],   // (n_params, 3 modes, 3N) for beam (with theta DOF)
            "u_strain_translation": [...] // (n_params, 3 modes, 2N) just the (ux, uy)
          }
        },
        ...
      ]
    }

Note: u_strain is stored as JSON arrays. To keep file size reasonable,
we only store the 3 strain-mode responses (xx, yy, xy), not all
intermediate internal states.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .parse_to_graph import LaceGraph, parse_file, parse_manifest
from .mechanics import (
    default_lattice_vectors, vertex_position, cell_vectors, cell_area,
    edge_geometry, homogenize, poisson_ratio_at_angle, poisson_profile,
    fix_translation, assemble_stiffness, voigt_to_tensor, rotate_voigt,
)
from .mechanics_beam import (
    homogenize_beam, assemble_stiffness_beam, fix_translation_beam,
    analyze_beam,
)
from .phonons import dispersion_features
from .canonicalize import both_canonical
from .manufacturability import manufacturability_block, provenance_block
from .humidity import humidity_features


# -----------------------------------------------------------------------
# Parameter grids
# -----------------------------------------------------------------------

SPRING_K_ANG_GRID = [0.0, 0.001, 0.01, 0.1, 1.0]
BEAM_AR_GRID = [5.0, 10.0, 20.0, 50.0, 200.0]
THETA_GRID_DEG = list(range(0, 180, 2))     # 0, 2, ..., 178 degrees (90 angles)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def classify(nu_min: float, nu_max: float) -> str:
    """Standard auxetic classification."""
    if not np.isfinite(nu_min) or not np.isfinite(nu_max):
        return "fully_floppy"
    if nu_max < 0:
        return "homogeneously_auxetic"
    if nu_min < 0:
        return "directionally_auxetic"
    return "non_auxetic"


def nu_profile_from_C(C: np.ndarray, theta_deg_grid: List[float]
                       ) -> List[float]:
    """Compute nu(theta) on the fixed angle grid."""
    out = []
    for td in theta_deg_grid:
        t = np.radians(td)
        nu = poisson_ratio_at_angle(C, t)
        if not np.isfinite(nu):
            # Cap extreme values for JSON serializability
            out.append(None)
        else:
            # Clip extreme values
            nu_clipped = max(-100.0, min(100.0, float(nu)))
            out.append(nu_clipped)
    return out


def nu_extrema(nu_profile: List[Optional[float]]) -> tuple:
    """Get (nu_min, nu_max, classification) from a profile."""
    finite = [v for v in nu_profile if v is not None]
    if not finite:
        return (None, None, "fully_floppy")
    nu_min = min(finite)
    nu_max = max(finite)
    return (nu_min, nu_max, classify(nu_min, nu_max))


# -----------------------------------------------------------------------
# Compute spring model on a parameter grid
# -----------------------------------------------------------------------

def compute_spring_for_ground(graph: LaceGraph,
                               k_ang_grid: List[float] = None,
                               theta_grid: List[float] = None
                               ) -> Dict[str, Any]:
    """For each k_ang in the grid, compute C_voigt, nu profile, and the
    strain-response internal displacements."""
    if k_ang_grid is None:
        k_ang_grid = SPRING_K_ANG_GRID
    if theta_grid is None:
        theta_grid = THETA_GRID_DEG

    L = default_lattice_vectors()
    N = len(graph.vertices)
    out: Dict[str, Any] = {
        "k_ang": list(k_ang_grid),
        "C_voigt": [],
        "nu_profile": [],
        "nu_min": [],
        "nu_max": [],
        "classification": [],
        "u_strain": [],   # list per k_ang; each is list of 3 mode responses
                          # each mode response is list of 2N floats
    }
    for k_ang in k_ang_grid:
        K, G, H, A = assemble_stiffness(
            graph, L=L, k_per_unit_length=1.0, k_angular=k_ang)
        K_red, G_red, free_idx = fix_translation(K, G)
        # Solve K_red u = -G_red eps for the 3 strain modes
        try:
            u_red, _, _, _ = np.linalg.lstsq(K_red, -G_red, rcond=1e-12)
        except np.linalg.LinAlgError:
            u_red = np.zeros((K_red.shape[0], 3))
        u_full = np.zeros((2 * N, 3))
        u_full[free_idx, :] = u_red
        # C tensor
        C = (H + G.T @ u_full) / A
        C = 0.5 * (C + C.T)

        nu_prof = nu_profile_from_C(C, theta_grid)
        nu_min, nu_max, cls = nu_extrema(nu_prof)

        out["C_voigt"].append(C.tolist())
        out["nu_profile"].append(nu_prof)
        out["nu_min"].append(nu_min)
        out["nu_max"].append(nu_max)
        out["classification"].append(cls)
        # Save internal displacements as (3, 2N) array — 3 modes
        out["u_strain"].append(u_full.T.tolist())  # shape (3, 2N)

    return out


# -----------------------------------------------------------------------
# Compute beam model on a parameter grid
# -----------------------------------------------------------------------

def compute_beam_for_ground(graph: LaceGraph,
                              AR_grid: List[float] = None,
                              theta_grid: List[float] = None
                              ) -> Dict[str, Any]:
    """For each AR in the grid, compute C_voigt, nu profile, internal
    displacements (translation only — we drop theta DOFs from the output
    since the visualizer doesn't need them)."""
    if AR_grid is None:
        AR_grid = BEAM_AR_GRID
    if theta_grid is None:
        theta_grid = THETA_GRID_DEG

    L = default_lattice_vectors()
    N = len(graph.vertices)
    out: Dict[str, Any] = {
        "AR": list(AR_grid),
        "C_voigt": [],
        "nu_profile": [],
        "nu_min": [],
        "nu_max": [],
        "classification": [],
        "u_strain": [],   # list per AR, shape (3 modes, 2N) — translation only
    }
    for AR in AR_grid:
        try:
            C, A, u_full = homogenize_beam(
                graph, L_lattice=L, EA=1.0, aspect_ratio=AR)
        except Exception as exc:
            # Failure: store NaN
            C = np.full((3, 3), np.nan)
            u_full = np.zeros((3 * N, 3))
        # Extract translational DOFs only (drop theta = every 3rd DOF)
        # u_full has shape (3N, 3). Reshape and select first 2 of every 3.
        u_full_resh = u_full.reshape(N, 3, 3)  # (vertex, dof_at_vertex, mode)
        u_trans = u_full_resh[:, 0:2, :]        # (vertex, 2, mode)
        u_trans_flat = u_trans.transpose(2, 0, 1).reshape(3, 2 * N)  # (mode, 2N)

        nu_prof = nu_profile_from_C(C, theta_grid)
        nu_min, nu_max, cls = nu_extrema(nu_prof)

        out["C_voigt"].append(C.tolist() if np.all(np.isfinite(C)) else
                               [[None, None, None]] * 3)
        out["nu_profile"].append(nu_prof)
        out["nu_min"].append(nu_min)
        out["nu_max"].append(nu_max)
        out["classification"].append(cls)
        out["u_strain"].append(u_trans_flat.tolist())

    return out


# -----------------------------------------------------------------------
# Per-ground assembly
# -----------------------------------------------------------------------

def build_ground_record(graph: LaceGraph, name: str, family: str,
                          theta_grid: List[float] = None,
                          thumbnail_dir: Optional[str] = None,
                          ) -> Dict[str, Any]:
    """Build the full atlas record for one ground.

    If `thumbnail_dir` is given, expected PNGs at
    {thumbnail_dir}/{family}/{name}/lace.png and .../deformed.png are
    referenced by relative path in the record so the visualizer can
    display them inline. Files are NOT validated to exist; missing
    thumbnails will be reflected as 404s in the visualizer.
    """
    if theta_grid is None:
        theta_grid = THETA_GRID_DEG

    L = default_lattice_vectors()

    # Geometry
    vertices_uc = [[v[0], v[1]] for v in graph.vertices]   # (col, row)
    edges = [{"src": e.src_idx, "dst": e.dst_idx,
              "wrap": [e.wrap[0], e.wrap[1]]}
             for e in graph.edges]
    A = cell_area(graph, L)

    record: Dict[str, Any] = {
        "name": name,
        "family": family,
        "n_rows": graph.n_rows,
        "n_cols": graph.n_cols,
        "n_vertices": len(graph.vertices),
        "n_edges": len(graph.edges),
        "vertices": vertices_uc,
        "edges": edges,
        "lattice": L.tolist(),
        "cell_area": A,
    }

    if thumbnail_dir is not None:
        # Store relative paths the visualizer can use as URLs
        record["thumbnails"] = {
            "lace": f"{thumbnail_dir}/{family}/{name}/lace.png",
            "deformed": f"{thumbnail_dir}/{family}/{name}/deformed.png",
            "dispersion": f"{thumbnail_dir}/{family}/{name}/dispersion.png",
        }

    # Mechanics
    record["spring"] = compute_spring_for_ground(graph, theta_grid=theta_grid)
    record["beam"] = compute_beam_for_ground(graph, theta_grid=theta_grid)

    # Phonons (spring-model dispersion: 22 scalar descriptors)
    record["phonon"] = dispersion_features(graph, k_angular=0.01)

    # Humidity / swelling response (perpendicular eigenstrain on each strut)
    record["humidity"] = humidity_features(graph, k_angular=0.01)

    # Canonical-form fingerprints for graph and lace
    record.update(both_canonical(record))

    # Manufacturability properties + provenance metadata
    record["manufacturability"] = manufacturability_block(record, source="irvine")
    record["provenance"] = provenance_block(
        source="irvine",
        irvine_label=f"{record['family']}/{record['name']}",
    )

    return record


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def _write_features_csv(grounds, csv_path,
                          spring_k_ang_grid, beam_AR_grid,
                          spring_default_idx, beam_default_idx):
    """Flatten each ground to a single CSV row of scalar fields.

    Includes:
      - Identification (family, name, n_rows, n_cols, n_vertices, n_edges, cell_area)
      - Default-parameter mechanics (spring nu_min/nu_max at default k_ang,
        beam nu_min/nu_max at default AR, classifications)
      - All phonon scalar descriptors (if present)
      - All humidity scalar descriptors (if present)
    """
    import csv

    # Discover the field set from the first ground that has each block
    sample = grounds[0] if grounds else {}
    phonon_keys = sorted((sample.get("phonon") or {}).keys())
    humidity_keys = sorted((sample.get("humidity") or {}).keys())

    headers = [
        "family", "name", "n_rows", "n_cols",
        "n_vertices", "n_edges", "cell_area",
        "spring_nu_min", "spring_nu_max", "spring_classification",
        "beam_nu_min", "beam_nu_max", "beam_classification",
    ]
    headers += [f"phonon_{k}" for k in phonon_keys]
    headers += [f"humidity_{k}" for k in humidity_keys]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for g in grounds:
            row = [
                g.get("family", ""), g.get("name", ""),
                g.get("n_rows", ""), g.get("n_cols", ""),
                g.get("n_vertices", ""), g.get("n_edges", ""),
                g.get("cell_area", ""),
            ]
            sp = g.get("spring", {})
            bm = g.get("beam", {})
            row.extend([
                _safe_get(sp.get("nu_min"), spring_default_idx),
                _safe_get(sp.get("nu_max"), spring_default_idx),
                _safe_get(sp.get("classification"), spring_default_idx, ""),
                _safe_get(bm.get("nu_min"), beam_default_idx),
                _safe_get(bm.get("nu_max"), beam_default_idx),
                _safe_get(bm.get("classification"), beam_default_idx, ""),
            ])
            ph = g.get("phonon") or {}
            row.extend(ph.get(k, "") for k in phonon_keys)
            hm = g.get("humidity") or {}
            row.extend(hm.get(k, "") for k in humidity_keys)
            writer.writerow(row)


def _safe_get(seq, idx, default=""):
    if seq is None:
        return default
    try:
        v = seq[idx]
    except (IndexError, TypeError):
        return default
    return default if v is None else v


def main():
    ap = argparse.ArgumentParser(description="Build atlas.json from TesseLace catalog.")
    ap.add_argument("--catalog", default="tesselace_catalog",
                    help="Path to scraped catalog directory containing manifest.csv")
    ap.add_argument("--output", default="atlas.json",
                    help="Output JSON path")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N grounds (smoke test)")
    ap.add_argument("--filter-family", type=str, default=None,
                    help="Only process grounds in this family (e.g. 3_6)")
    ap.add_argument("--thumbnail-dir", type=str, default="thumbnails",
                    help="Relative URL prefix where the visualizer will find "
                         "pre-rendered thumbnails. Set empty to omit thumbnail "
                         "links from the output. Default: 'thumbnails'.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Load manifest
    manifest_path = os.path.join(args.catalog, "manifest.csv")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("       Run scrape_tesselace.py first.", file=sys.stderr)
        sys.exit(1)
    manifest = parse_manifest(manifest_path)

    # Filter
    if args.filter_family:
        manifest = [m for m in manifest if m['family'] == args.filter_family]
    if args.limit is not None:
        manifest = manifest[:args.limit]

    print(f"Processing {len(manifest)} grounds...")

    grounds: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, graph in enumerate(manifest):
        try:
            record = build_ground_record(
                graph, name=graph.name, family=graph.family,
                thumbnail_dir=args.thumbnail_dir if args.thumbnail_dir else None)
            grounds.append(record)
            if args.verbose and i % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(manifest) - i - 1)
                k_ang_idx = SPRING_K_ANG_GRID.index(0.01) if 0.01 in SPRING_K_ANG_GRID else 1
                nu_min = record["spring"]["nu_min"][k_ang_idx]
                nu_str = f"{nu_min:+.3f}" if nu_min is not None else "N/A"
                print(f"  [{i+1}/{len(manifest)}] {graph.family}/{graph.name} "
                      f"nu_min(spring,k=0.01)={nu_str}  "
                      f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
        except Exception as exc:
            failed.append({
                "name": graph.name,
                "family": graph.family,
                "error": str(exc),
            })
            if args.verbose:
                print(f"  FAIL on {graph.name}: {exc}")

    elapsed = time.time() - t0
    print(f"\nProcessed {len(grounds)} grounds in {elapsed:.1f}s "
          f"({len(failed)} failures)")

    # Build summary index for visualizer convenience: list of grounds sorted
    # by family / name, with per-ground summary stats at the default settings
    # (spring k_ang=0.01, beam AR=10).
    spring_default_idx = (SPRING_K_ANG_GRID.index(0.01)
                           if 0.01 in SPRING_K_ANG_GRID else 1)
    beam_default_idx = (BEAM_AR_GRID.index(10.0)
                         if 10.0 in BEAM_AR_GRID else 1)
    summary = []
    for i, g in enumerate(grounds):
        s = {
            "idx": i,
            "name": g["name"],
            "family": g["family"],
            "n_rows": g["n_rows"],
            "n_cols": g["n_cols"],
            "n_vertices": g["n_vertices"],
            "spring_default_nu_min": g["spring"]["nu_min"][spring_default_idx],
            "spring_default_nu_max": g["spring"]["nu_max"][spring_default_idx],
            "spring_default_classification": g["spring"]["classification"][spring_default_idx],
            "beam_default_nu_min": g["beam"]["nu_min"][beam_default_idx],
            "beam_default_nu_max": g["beam"]["nu_max"][beam_default_idx],
            "beam_default_classification": g["beam"]["classification"][beam_default_idx],
        }
        summary.append(s)

    # Build atlas
    atlas = {
        "metadata": {
            "n_grounds": len(grounds),
            "version": "0.1.0",
            "spring_k_ang_grid": SPRING_K_ANG_GRID,
            "spring_default_idx": spring_default_idx,
            "beam_AR_grid": BEAM_AR_GRID,
            "beam_default_idx": beam_default_idx,
            "theta_grid_deg": THETA_GRID_DEG,
            "build_date": datetime.now().isoformat(),
            "build_elapsed_seconds": elapsed,
            "attribution": (
                "TesseLace ground patterns by Veronika Irvine, "
                "https://d-bl.github.io/tesselace-to-gf/, CC-BY 4.0. "
                "PhD thesis: https://dspace.library.uvic.ca/items/"
                "867c403c-4f45-4c54-89d1-1c8d138dfe92"
            ),
            "n_failures": len(failed),
        },
        "summary": summary,
        "grounds": grounds,
        "failures": failed,
    }

    print(f"Writing {args.output}...")
    with open(args.output, "w") as f:
        json.dump(atlas, f, separators=(",", ":"))
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"  {size_mb:.2f} MB")

    # ---------------------------------------------------------------
    # Companion CSV: one row per ground, all scalar fields flattened.
    # Outsiders typically prefer this over nested JSON for pandas/R use.
    # ---------------------------------------------------------------
    csv_path = args.output.replace(".json", "_features.csv")
    if csv_path == args.output:
        csv_path = args.output + ".csv"
    print(f"Writing {csv_path}...")
    _write_features_csv(grounds, csv_path,
                          SPRING_K_ANG_GRID, BEAM_AR_GRID,
                          spring_default_idx, beam_default_idx)
    size_kb = os.path.getsize(csv_path) / 1e3
    print(f"  {size_kb:.1f} KB")

    # Summary stats
    print("\nQuick auxetic stats (spring, k_ang=0.01):")
    k_ang_idx = SPRING_K_ANG_GRID.index(0.01) if 0.01 in SPRING_K_ANG_GRID else 1
    classifications: Dict[str, int] = {}
    for g in grounds:
        cls = g["spring"]["classification"][k_ang_idx]
        classifications[cls] = classifications.get(cls, 0) + 1
    for cls, n in sorted(classifications.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n} ({100 * n / len(grounds):.1f}%)")


if __name__ == "__main__":
    main()
