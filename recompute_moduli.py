"""
recompute_moduli.py

Compute E_profile, E_min, E_max, G_profile, G_min, G_max, K for every
ground in atlas.json, in both spring and beam blocks. Reads from the
existing C_voigt arrays — no homogenization needed.

Idempotent.

Usage:
    python3 recompute_moduli.py            # dry run
    python3 recompute_moduli.py --apply
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.mechanics import (                # noqa: E402
    youngs_modulus_at_angle,
    shear_modulus_at_angle,
    area_bulk_modulus,
)


ATLAS_PATH = Path("docs/atlas.json")


def profile_from_C(C, theta_grid_deg, modulus_fn):
    """Compute a profile (one float per angle) using modulus_fn(C, theta)."""
    out = []
    for td in theta_grid_deg:
        t = np.radians(td)
        v = modulus_fn(C, t)
        if not np.isfinite(v):
            out.append(None)
        else:
            # Cap extreme values for JSON serializability — same convention
            # as nu_profile.
            out.append(max(-1e6, min(1e6, float(v))))
    return out


def extrema(profile):
    finite = [v for v in profile if v is not None]
    if not finite:
        return None, None
    return float(min(finite)), float(max(finite))


def recompute_block(block, theta_grid_deg):
    """Add E_profile / E_min / E_max / G_profile / G_min / G_max / K to a
    spring or beam block. C_voigt arrays must be present.

    Returns the number of params processed (== number of K values added)."""
    C_arrays = block.get("C_voigt")
    if not C_arrays:
        return 0
    n_params = len(C_arrays)

    E_profiles = []
    E_mins = []
    E_maxs = []
    G_profiles = []
    G_mins = []
    G_maxs = []
    Ks = []

    for k in range(n_params):
        C = np.asarray(C_arrays[k], dtype=float)

        E_prof = profile_from_C(C, theta_grid_deg, youngs_modulus_at_angle)
        E_min, E_max = extrema(E_prof)
        G_prof = profile_from_C(C, theta_grid_deg, shear_modulus_at_angle)
        G_min, G_max = extrema(G_prof)
        K_val = area_bulk_modulus(C)
        if not np.isfinite(K_val):
            K_val = None
        else:
            K_val = float(K_val)

        E_profiles.append(E_prof)
        E_mins.append(E_min)
        E_maxs.append(E_max)
        G_profiles.append(G_prof)
        G_mins.append(G_min)
        G_maxs.append(G_max)
        Ks.append(K_val)

    block["E_profile"] = E_profiles
    block["E_min"] = E_mins
    block["E_max"] = E_maxs
    block["G_profile"] = G_profiles
    block["G_min"] = G_mins
    block["G_max"] = G_maxs
    block["K"] = Ks
    return n_params


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

    n_done = 0
    for g in atlas["grounds"]:
        if "spring" in g and g["spring"].get("C_voigt"):
            recompute_block(g["spring"], theta_grid)
        if "beam" in g and g["beam"].get("C_voigt"):
            recompute_block(g["beam"], theta_grid)
        n_done += 1

    print(f"Recomputed moduli for {n_done} grounds.")

    # Quick sanity: distribution of E_min, G_min, K at default beam AR
    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)
    E_mins_beam = [g["beam"]["E_min"][beam_default_idx]
                   for g in atlas["grounds"]
                   if "beam" in g and g["beam"].get("E_min")
                   and g["beam"]["E_min"][beam_default_idx] is not None]
    K_beam = [g["beam"]["K"][beam_default_idx]
              for g in atlas["grounds"]
              if "beam" in g and g["beam"].get("K")
              and g["beam"]["K"][beam_default_idx] is not None]

    if E_mins_beam:
        print(f"\nBeam @ AR=10 (default), E_min distribution:")
        arr = np.asarray(E_mins_beam)
        print(f"  count: {len(arr)}, min: {arr.min():.4g}, "
              f"median: {np.median(arr):.4g}, max: {arr.max():.4g}")
    if K_beam:
        print(f"\nBeam @ AR=10 (default), K distribution:")
        arr = np.asarray(K_beam)
        print(f"  count: {len(arr)}, min: {arr.min():.4g}, "
              f"median: {np.median(arr):.4g}, max: {arr.max():.4g}")

    # Show a few sample grounds to eyeball
    print(f"\n=== Sample grounds (beam @ AR=10) ===")
    samples = [
        ("4", "2x1_1"),                          # square tiling — should be near-isotropic
        ("3_6", "R3M3_6x6_1"),                  # top auxetic
        ("3_4_5_7", "V9_3x3_028"),              # top Taylor auxetic
    ]
    for fam, nm in samples:
        g = next((g for g in atlas["grounds"]
                   if g["family"] == fam and g["name"] == nm), None)
        if g is None:
            print(f"  {fam}/{nm}: not found")
            continue
        bm = g["beam"]
        idx = beam_default_idx
        print(f"  {fam}/{nm}:")
        print(f"    E: min={bm['E_min'][idx]:.4g}  max={bm['E_max'][idx]:.4g}  "
              f"anisotropy={bm['E_max'][idx]/max(bm['E_min'][idx], 1e-12):.2f}x")
        print(f"    G: min={bm['G_min'][idx]:.4g}  max={bm['G_max'][idx]:.4g}")
        print(f"    K: {bm['K'][idx]:.4g}")

    if not args.apply:
        print(f"\n(dry run -- atlas not modified)")
        return

    print(f"\nWriting {atlas_path} (atomic)...")
    atomic_write_json(atlas_path, atlas)
    print("Done.")


if __name__ == "__main__":
    main()
