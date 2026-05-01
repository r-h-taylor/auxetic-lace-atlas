"""
rank_submission.py
==================

Rank a computed submission record against an existing atlas to give the
user a sense of where their contribution fits in the design space.

Reports:
  1. PERCENTILES: where does this submission's nu_min and nu_max fall in
     the distribution of the existing catalog (per-model)?
  2. NEAREST NEIGHBORS: which existing entries have the most similar
     mechanical response, by Frobenius distance on the C tensor?
  3. PARETO FRONT: is this submission strictly better than any existing
     entry on the auxetic axes (nu_min lowest, |C11+C22| highest)?
  4. DUPLICATE CHECK: is this submission isomorphic to an existing entry
     (using canonical-label comparison)?

USAGE:
    python3 rank_submission.py record.json --atlas atlas.json
    python3 rank_submission.py record.json --atlas atlas.json --json
        # output ranking as JSON instead of human-readable text
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_default_C(record: Dict[str, Any], model: str) -> Optional[np.ndarray]:
    """Return the C tensor at the default parameter setting for the
    given model ('spring' or 'beam'). None if not available."""
    if model not in record:
        return None
    m = record[model]
    if model == "spring":
        # Default = first non-zero k_ang (k_ang=0 is fully floppy)
        k_ang_grid = m.get("k_ang", [])
        try:
            idx = k_ang_grid.index(0.01)
        except ValueError:
            idx = 1 if len(k_ang_grid) > 1 else 0
    else:
        AR_grid = m.get("AR", [])
        try:
            idx = AR_grid.index(10.0)
        except ValueError:
            idx = 1 if len(AR_grid) > 1 else 0
    C = m.get("C_voigt")
    if not C or idx >= len(C):
        return None
    arr = np.array(C[idx])
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def get_default_nu(record: Dict[str, Any], model: str
                    ) -> Tuple[Optional[float], Optional[float]]:
    if model not in record:
        return (None, None)
    m = record[model]
    if model == "spring":
        k_ang_grid = m.get("k_ang", [])
        try:
            idx = k_ang_grid.index(0.01)
        except ValueError:
            idx = 1 if len(k_ang_grid) > 1 else 0
    else:
        AR_grid = m.get("AR", [])
        try:
            idx = AR_grid.index(10.0)
        except ValueError:
            idx = 1 if len(AR_grid) > 1 else 0
    nu_min_list = m.get("nu_min", [])
    nu_max_list = m.get("nu_max", [])
    nu_min = nu_min_list[idx] if idx < len(nu_min_list) else None
    nu_max = nu_max_list[idx] if idx < len(nu_max_list) else None
    return (nu_min, nu_max)


def percentile_rank(value: float, distribution: List[float]) -> float:
    """What percentile is `value` in `distribution`? 0.0 = lowest in
    population, 100.0 = highest. Linear interpolation between adjacent
    ranks. Skips None/NaN values."""
    valid = [x for x in distribution if x is not None and np.isfinite(x)]
    if not valid:
        return float('nan')
    sorted_vals = sorted(valid)
    n = len(sorted_vals)
    # Find position
    rank = 0
    for v in sorted_vals:
        if v < value:
            rank += 1
        else:
            break
    return 100.0 * rank / n


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def percentile_report(submission: Dict[str, Any],
                       atlas: Dict[str, Any]) -> Dict[str, Any]:
    """For each model, report the percentile of nu_min and nu_max
    against the atlas distribution."""
    out: Dict[str, Any] = {}
    for model in ["spring", "beam"]:
        nu_min, nu_max = get_default_nu(submission, model)
        if nu_min is None or nu_max is None:
            out[model] = None
            continue
        # Collect the distribution from atlas
        nu_min_dist: List[float] = []
        nu_max_dist: List[float] = []
        for g in atlas.get("grounds", []):
            mn, mx = get_default_nu(g, model)
            if mn is not None:
                nu_min_dist.append(mn)
            if mx is not None:
                nu_max_dist.append(mx)
        out[model] = {
            "submission_nu_min": nu_min,
            "submission_nu_max": nu_max,
            "nu_min_percentile": percentile_rank(nu_min, nu_min_dist),
            "nu_max_percentile": percentile_rank(nu_max, nu_max_dist),
            "n_in_distribution": len(nu_min_dist),
        }
    return out


def nearest_neighbors(submission: Dict[str, Any],
                       atlas: Dict[str, Any],
                       k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """For each model, find the K existing entries with the closest
    Frobenius distance on the C tensor."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for model in ["spring", "beam"]:
        C_sub = get_default_C(submission, model)
        if C_sub is None:
            out[model] = []
            continue
        distances: List[Tuple[float, Dict[str, Any]]] = []
        for g in atlas.get("grounds", []):
            C_g = get_default_C(g, model)
            if C_g is None:
                continue
            d = float(np.linalg.norm(C_sub - C_g, ord='fro'))
            mn, mx = get_default_nu(g, model)
            distances.append((d, {
                "name": g.get("name"),
                "family": g.get("family"),
                "distance": d,
                "nu_min": mn,
                "nu_max": mx,
            }))
        distances.sort(key=lambda x: x[0])
        out[model] = [d[1] for d in distances[:k]]
    return out


def pareto_check(submission: Dict[str, Any],
                  atlas: Dict[str, Any]) -> Dict[str, Any]:
    """Is the submission on the Pareto front for (nu_min lowest =
    most-auxetic) vs (cell_area smallest = most-compact)?

    Reports:
      - dominators: existing entries that strictly dominate the submission
        (lower nu_min AND smaller cell_area)
      - dominated: existing entries that the submission strictly
        dominates
      - is_pareto: True iff dominators is empty
    """
    out: Dict[str, Any] = {}
    for model in ["spring", "beam"]:
        nu_min, _ = get_default_nu(submission, model)
        cell_area = submission.get("cell_area")
        if nu_min is None or cell_area is None:
            out[model] = None
            continue
        dominators = []
        dominated = []
        for g in atlas.get("grounds", []):
            mn, _ = get_default_nu(g, model)
            ca = g.get("cell_area")
            if mn is None or ca is None:
                continue
            # Strict domination on (nu_min, cell_area) — lower is better in both
            if mn < nu_min and ca <= cell_area:
                dominators.append({
                    "name": g.get("name"), "family": g.get("family"),
                    "nu_min": mn, "cell_area": ca,
                })
            elif mn < nu_min and ca < cell_area:
                dominators.append({
                    "name": g.get("name"), "family": g.get("family"),
                    "nu_min": mn, "cell_area": ca,
                })
            elif nu_min < mn and cell_area <= ca:
                dominated.append({
                    "name": g.get("name"), "family": g.get("family"),
                    "nu_min": mn, "cell_area": ca,
                })
        out[model] = {
            "is_pareto": len(dominators) == 0,
            "n_dominators": len(dominators),
            "dominators_top": dominators[:5],
            "n_dominated": len(dominated),
            "dominated_top": dominated[:5],
        }
    return out


def duplicate_check(submission: Dict[str, Any],
                     atlas: Dict[str, Any]) -> Dict[str, Any]:
    """Is the submission isomorphic to any existing entry (by canonical
    label)? Uses enum_v2.brute_force canonical-label machinery.

    NOTE: This only catches isomorphism among graphs with the SAME unit
    cell size. Submissions whose period parallelogram differs from any
    catalog entry are guaranteed unique by this metric (but might still
    be related by lattice rescaling — out of scope for this MVP)."""
    try:
        from auxetic_lace.canonical.brute_force import Ground, _canonical_label
    except ImportError as exc:
        return {"error": f"could not import enum_v2.brute_force: {exc}"}

    def submission_to_ground(rec: Dict[str, Any]):
        verts = rec["vertices"]
        n_rows = rec["n_rows"]
        n_cols = rec["n_cols"]
        edges = set()
        for e in rec["edges"]:
            sc, sr = verts[e["src"]]
            dc, dr = verts[e["dst"]]
            wrap = e["wrap"]
            dx = dc - sc + wrap[0] * n_cols
            dy = dr - sr + wrap[1] * n_rows
            edges.add((sc, sr, dx, dy))
        return Ground(n_cols=n_cols, n_rows=n_rows,
                       edges=frozenset(edges))

    g_sub = submission_to_ground(submission)
    label_sub = _canonical_label(g_sub)

    matches = []
    for g_atlas in atlas.get("grounds", []):
        if g_atlas.get("n_rows") != submission["n_rows"]:
            continue
        if g_atlas.get("n_cols") != submission["n_cols"]:
            continue
        try:
            g_other = submission_to_ground(g_atlas)
            label_other = _canonical_label(g_other)
            if label_other == label_sub:
                matches.append({
                    "name": g_atlas.get("name"),
                    "family": g_atlas.get("family"),
                })
        except Exception:
            continue

    return {
        "is_unique": len(matches) == 0,
        "n_matches": len(matches),
        "matches": matches,
    }


# ---------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------

def rank_submission(submission: Dict[str, Any],
                     atlas: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the full ranking report."""
    return {
        "name": submission.get("name"),
        "family": submission.get("family"),
        "atlas_size": len(atlas.get("grounds", [])),
        "percentiles": percentile_report(submission, atlas),
        "nearest_neighbors": nearest_neighbors(submission, atlas, k=5),
        "pareto": pareto_check(submission, atlas),
        "duplicates": duplicate_check(submission, atlas),
    }


def format_report(report: Dict[str, Any]) -> str:
    """Pretty-print the ranking report."""
    lines = []
    lines.append(f"Ranking: {report['family']}/{report['name']}")
    lines.append(f"  vs atlas of {report['atlas_size']} entries")
    lines.append("")

    # Percentiles
    pct = report["percentiles"]
    for model in ["spring", "beam"]:
        m = pct.get(model)
        if not m:
            lines.append(f"[{model}] no data")
            continue
        lines.append(f"[{model}] @ default params:")
        lines.append(f"  nu_min = {m['submission_nu_min']:+.4f} "
                      f"(percentile {m['nu_min_percentile']:.1f}% "
                      f"in distribution of {m['n_in_distribution']})")
        lines.append(f"  nu_max = {m['submission_nu_max']:+.4f} "
                      f"(percentile {m['nu_max_percentile']:.1f}%)")
        lines.append("  Lower nu_min percentile = more auxetic than the "
                      "catalog distribution.")
        lines.append("")

    # Nearest neighbors
    lines.append("Nearest neighbors by C-tensor Frobenius distance:")
    nn = report["nearest_neighbors"]
    for model in ["spring", "beam"]:
        neighbors = nn.get(model, [])
        if not neighbors:
            continue
        lines.append(f"  [{model}]")
        for n in neighbors:
            mn_str = f"{n['nu_min']:+.3f}" if n['nu_min'] is not None else "N/A"
            mx_str = f"{n['nu_max']:+.3f}" if n['nu_max'] is not None else "N/A"
            lines.append(f"    {n['family']:>20s}/{n['name']:<25s} "
                          f"d={n['distance']:.4f}  "
                          f"nu_min={mn_str}, nu_max={mx_str}")
    lines.append("")

    # Pareto
    pareto = report["pareto"]
    lines.append("Pareto status (most-auxetic vs most-compact):")
    for model in ["spring", "beam"]:
        p = pareto.get(model)
        if not p:
            continue
        if p["is_pareto"]:
            lines.append(f"  [{model}] ON Pareto front "
                          f"(no entry strictly dominates this submission). "
                          f"{p['n_dominated']} entries are dominated.")
        else:
            lines.append(f"  [{model}] NOT on Pareto front ("
                          f"{p['n_dominators']} entries strictly dominate)")
            for d in p["dominators_top"]:
                lines.append(f"    dominated by {d['family']}/{d['name']}: "
                              f"nu_min={d['nu_min']:+.3f}, "
                              f"cell_area={d['cell_area']}")
    lines.append("")

    # Duplicates
    dup = report["duplicates"]
    if "error" in dup:
        lines.append(f"Duplicate check: {dup['error']}")
    elif dup["is_unique"]:
        lines.append("Duplicate check: UNIQUE — not isomorphic to any "
                      "existing catalog entry at this unit cell size.")
    else:
        lines.append(f"Duplicate check: ISOMORPHIC to {dup['n_matches']} "
                      f"existing entries:")
        for m in dup["matches"]:
            lines.append(f"  {m['family']}/{m['name']}")
        lines.append("  Consider using an existing entry instead, or merging "
                      "with a clearer name.")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Rank a computed submission against an atlas.")
    ap.add_argument("record", help="Path to submission record JSON "
                                    "(output of compute_submission.py)")
    ap.add_argument("--atlas", required=True,
                    help="Path to atlas.json to compare against")
    ap.add_argument("--json", action="store_true",
                    help="Output ranking as JSON instead of human-readable text")
    args = ap.parse_args()

    with open(args.record) as f:
        submission = json.load(f)
    with open(args.atlas) as f:
        atlas = json.load(f)

    report = rank_submission(submission, atlas)

    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        print()
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
