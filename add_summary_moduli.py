"""
add_summary_moduli.py

Extend each entry of atlas.summary[] with beam moduli scalars at the
default AR setting:

    beam_default_E_min
    beam_default_E_max
    beam_default_E_mean
    beam_default_G_min
    beam_default_G_max
    beam_default_K

These are read from the existing per-ground beam block (E_min, E_max,
E_profile, G_min, G_max, K). Idempotent.

Usage:
    python3 add_summary_moduli.py            # dry run
    python3 add_summary_moduli.py --apply
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np


ATLAS_PATH = Path("docs/atlas.json")


def safe_index(arr, i):
    if not arr:
        return None
    try:
        return arr[i]
    except (IndexError, TypeError):
        return None


def safe_mean(profile):
    """Mean of a profile (list of floats with possible None entries)."""
    if not profile:
        return None
    finite = [v for v in profile if v is not None
               and isinstance(v, (int, float)) and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


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

    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)
    print(f"Atlas: {atlas_path}")
    print(f"Grounds: {len(atlas['grounds'])}")
    print(f"Summary entries: {len(atlas['summary'])}")
    print(f"Beam default idx: {beam_default_idx}")
    print()

    # Build a name -> ground lookup
    by_name = {g["name"]: g for g in atlas["grounds"]}

    n_changed = 0
    n_missing_beam = 0
    for s in atlas["summary"]:
        g = by_name.get(s["name"])
        if g is None or "beam" not in g:
            n_missing_beam += 1
            continue
        bm = g["beam"]

        E_min = safe_index(bm.get("E_min"), beam_default_idx)
        E_max = safe_index(bm.get("E_max"), beam_default_idx)
        G_min = safe_index(bm.get("G_min"), beam_default_idx)
        G_max = safe_index(bm.get("G_max"), beam_default_idx)
        K_val = safe_index(bm.get("K"), beam_default_idx)
        E_profile_default = safe_index(bm.get("E_profile"), beam_default_idx)
        E_mean = safe_mean(E_profile_default)

        new_fields = {
            "beam_default_E_min": E_min,
            "beam_default_E_max": E_max,
            "beam_default_E_mean": E_mean,
            "beam_default_G_min": G_min,
            "beam_default_G_max": G_max,
            "beam_default_K": K_val,
        }
        # Idempotency check: skip if all fields already match
        already = all(s.get(k) == v for k, v in new_fields.items())
        if not already:
            s.update(new_fields)
            n_changed += 1

    print(f"Summary entries updated: {n_changed}")
    if n_missing_beam:
        print(f"Summary entries missing beam block: {n_missing_beam}")

    # Sanity: peek at one
    if atlas["summary"]:
        s = atlas["summary"][0]
        print(f"\nSample summary entry ({s['name']}):")
        for k in ("beam_default_E_min", "beam_default_E_max",
                  "beam_default_E_mean", "beam_default_K"):
            v = s.get(k)
            v_fmt = f"{v:.4g}" if v is not None else "None"
            print(f"  {k}: {v_fmt}")

    # Range distribution check
    E_means = [s.get("beam_default_E_mean") for s in atlas["summary"]
                if s.get("beam_default_E_mean") is not None]
    K_vals = [s.get("beam_default_K") for s in atlas["summary"]
               if s.get("beam_default_K") is not None]
    if E_means:
        arr = np.asarray(E_means)
        print(f"\nE_mean distribution: count={len(arr)}, "
              f"min={arr.min():.4g}, median={np.median(arr):.4g}, "
              f"max={arr.max():.4g}")
    if K_vals:
        arr = np.asarray(K_vals)
        print(f"K     distribution: count={len(arr)}, "
              f"min={arr.min():.4g}, median={np.median(arr):.4g}, "
              f"max={arr.max():.4g}")

    if not args.apply:
        print("\n(dry run -- atlas not modified)")
        return

    print(f"\nWriting {atlas_path} (atomic)...")
    atomic_write_json(atlas_path, atlas)
    print("Done.")


if __name__ == "__main__":
    main()
