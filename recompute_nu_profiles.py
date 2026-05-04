"""
recompute_nu_profiles.py

Recompute nu_profile / nu_min / nu_max / classification for every ground
in atlas.json, using the now-fixed mechanics.poisson_ratio_at_angle.

C_voigt itself is unchanged (homogenization didn't use the buggy rotation),
so this is just a derived-quantity refresh — milliseconds per ground.

Both spring and beam blocks are refreshed across all parameter settings
(5 spring k_ang values, 5 beam AR values).

After the data is fixed, summary[] is rebuilt to reflect the new
default-parameter scalars.

Idempotent.

Usage:
    python3 recompute_nu_profiles.py            # dry run
    python3 recompute_nu_profiles.py --apply
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.mechanics import poisson_ratio_at_angle  # noqa: E402


ATLAS_PATH = Path("docs/atlas.json")


def classify(nu_min, nu_max):
    if nu_min is None or nu_max is None:
        return "fully_floppy"
    if not (np.isfinite(nu_min) and np.isfinite(nu_max)):
        return "fully_floppy"
    if nu_max < 0:
        return "homogeneously_auxetic"
    if nu_min < 0:
        return "directionally_auxetic"
    return "non_auxetic"


def nu_profile_from_C(C, theta_grid_deg):
    """Compute nu profile at a fixed angle grid. Returns list of floats
    or None for non-finite values, matching the convention used in
    build_atlas.py (capped to [-100, 100])."""
    out = []
    for td in theta_grid_deg:
        t = np.radians(td)
        nu = poisson_ratio_at_angle(C, t)
        if not np.isfinite(nu):
            out.append(None)
        else:
            out.append(max(-100.0, min(100.0, float(nu))))
    return out


def nu_extrema(nu_profile):
    finite = [v for v in nu_profile if v is not None]
    if not finite:
        return None, None, "fully_floppy"
    nu_min = min(finite)
    nu_max = max(finite)
    return nu_min, nu_max, classify(nu_min, nu_max)


def recompute_block(block, theta_grid_deg):
    """Refresh nu_profile / nu_min / nu_max / classification arrays in a
    spring or beam block. The C_voigt arrays must be present and unchanged.

    Returns (deltas_count, total_count) — how many entries changed.
    """
    C_arrays = block.get("C_voigt")
    if not C_arrays:
        return 0, 0
    n_params = len(C_arrays)

    new_profiles = []
    new_nu_min = []
    new_nu_max = []
    new_classification = []
    for k in range(n_params):
        C = np.asarray(C_arrays[k], dtype=float)
        prof = nu_profile_from_C(C, theta_grid_deg)
        nu_min, nu_max, cls = nu_extrema(prof)
        new_profiles.append(prof)
        new_nu_min.append(nu_min)
        new_nu_max.append(nu_max)
        new_classification.append(cls)

    # Count changes for reporting
    changes = 0
    for k in range(n_params):
        if (block.get("nu_min") and block["nu_min"][k] != new_nu_min[k]):
            changes += 1
        if (block.get("nu_max") and block["nu_max"][k] != new_nu_max[k]):
            changes += 1
        if (block.get("classification")
            and block["classification"][k] != new_classification[k]):
            changes += 1

    block["nu_profile"] = new_profiles
    block["nu_min"] = new_nu_min
    block["nu_max"] = new_nu_max
    block["classification"] = new_classification
    return changes, n_params * 3


def rebuild_summary(atlas):
    spring_default_idx = atlas["metadata"].get("spring_default_idx", 2)
    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)

    def safe_index(arr, i):
        if not arr:
            return None
        try:
            return arr[i]
        except (IndexError, TypeError):
            return None

    new_summary = []
    for idx, g in enumerate(atlas["grounds"]):
        spring = g.get("spring", {}) or {}
        beam = g.get("beam", {}) or {}
        new_summary.append({
            "idx": idx,
            "name": g["name"],
            "family": g["family"],
            "source": g.get("source", "irvine"),
            "traditional_name": g.get("traditional_name"),
            "n_rows": g["n_rows"],
            "n_cols": g["n_cols"],
            "n_vertices": g["n_vertices"],
            "spring_default_nu_min": safe_index(spring.get("nu_min"), spring_default_idx),
            "spring_default_nu_max": safe_index(spring.get("nu_max"), spring_default_idx),
            "spring_default_classification":
                safe_index(spring.get("classification"), spring_default_idx),
            "beam_default_nu_min": safe_index(beam.get("nu_min"), beam_default_idx),
            "beam_default_nu_max": safe_index(beam.get("nu_max"), beam_default_idx),
            "beam_default_classification":
                safe_index(beam.get("classification"), beam_default_idx),
        })
    return new_summary


def atomic_write_json(path, data):
    parent = path.parent
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--atlas", default=str(ATLAS_PATH))
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    with atlas_path.open() as f:
        atlas = json.load(f)

    theta_grid = atlas["metadata"].get("theta_grid_deg",
                                        list(range(0, 180, 2)))

    print(f"Atlas: {atlas_path}")
    print(f"Grounds: {len(atlas['grounds'])}")
    print(f"theta grid: {len(theta_grid)} angles")
    print()

    # Track classification changes for impact summary
    from collections import Counter
    spring_class_before = Counter()
    spring_class_after = Counter()
    beam_class_before = Counter()
    beam_class_after = Counter()

    spring_default_idx = atlas["metadata"].get("spring_default_idx", 2)
    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)

    spring_top10_before = []
    spring_top10_after = []
    beam_top10_before = []
    beam_top10_after = []

    # First pass: snapshot old values
    for g in atlas["grounds"]:
        sp = g.get("spring", {}) or {}
        bm = g.get("beam", {}) or {}
        if sp.get("classification"):
            spring_class_before[sp["classification"][spring_default_idx]] += 1
        if bm.get("classification"):
            beam_class_before[bm["classification"][beam_default_idx]] += 1

    # Recompute
    total_changes = 0
    total_entries = 0
    for g in atlas["grounds"]:
        if "spring" in g and g["spring"].get("C_voigt"):
            ch, tot = recompute_block(g["spring"], theta_grid)
            total_changes += ch
            total_entries += tot
        if "beam" in g and g["beam"].get("C_voigt"):
            ch, tot = recompute_block(g["beam"], theta_grid)
            total_changes += ch
            total_entries += tot

    # Snapshot new
    for g in atlas["grounds"]:
        sp = g.get("spring", {}) or {}
        bm = g.get("beam", {}) or {}
        if sp.get("classification"):
            spring_class_after[sp["classification"][spring_default_idx]] += 1
        if bm.get("classification"):
            beam_class_after[bm["classification"][beam_default_idx]] += 1

    # Rebuild summary
    atlas["summary"] = rebuild_summary(atlas)

    # Top-10 by beam_default nu_min, before vs after
    s = atlas["summary"]
    s_after = sorted([
        x for x in s
        if x.get("beam_default_nu_min") is not None
        and x.get("beam_default_nu_max") is not None
        and x["beam_default_nu_max"] < 0
    ], key=lambda x: x["beam_default_nu_min"])[:10]

    print("=== Recompute summary ===")
    print(f"  Entries changed: {total_changes} / {total_entries}")
    print()
    print(f"=== Spring classification (default k_ang) ===")
    for cls in sorted(set(spring_class_before) | set(spring_class_after)):
        b = spring_class_before.get(cls, 0)
        a = spring_class_after.get(cls, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        print(f"  {cls:35s}  before={b:4d}  after={a:4d}  ({sign}{delta:+d})")
    print()
    print(f"=== Beam classification (default AR) ===")
    for cls in sorted(set(beam_class_before) | set(beam_class_after)):
        b = beam_class_before.get(cls, 0)
        a = beam_class_after.get(cls, 0)
        delta = a - b
        print(f"  {cls:35s}  before={b:4d}  after={a:4d}  ({delta:+d})")

    print(f"\n=== New top-10 homogeneously auxetic (beam, default AR) ===")
    for i, x in enumerate(s_after, 1):
        print(f"  {i:2d}. {x['family']:10s} / {x['name']:14s}  "
              f"nu_min={x['beam_default_nu_min']:+.3f}  "
              f"nu_max={x['beam_default_nu_max']:+.3f}  "
              f"({x['source']})")

    if not args.apply:
        print(f"\n(dry run -- atlas not modified)")
        return

    print(f"\nWriting {atlas_path} (atomic)...")
    atomic_write_json(atlas_path, atlas)
    print("Done.")


if __name__ == "__main__":
    main()
