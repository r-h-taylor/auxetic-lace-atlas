"""
analyze_humidity_class.py
=========================

Characterize the "humidity-active but weakly auxetic" class identified
in the (nu_min, eta_pore) scatter of the atlas.

Self-contained: derives geometric features directly from atlas.json so it
doesn't depend on a separate features.csv file.

Run from the repo root:
    python3 analysis/analyze_humidity_class.py [atlas.json]

Outputs:
  - Cohen's d for each feature, comparing the humidity-only class
    against the rest of the atlas.
  - Top distinguishing features ranked by |d|.
  - Feature values for the top grounds in the class.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter


def derive_features(g: dict) -> dict:
    """Derive a small but informative set of geometric / structural features
    from a ground record."""
    feats = {}
    feats["n_vertices"] = g["n_vertices"]
    feats["n_edges"] = g["n_edges"]
    feats["n_rows"] = g["n_rows"]
    feats["n_cols"] = g["n_cols"]
    feats["cell_area"] = g.get("cell_area", 0.0)
    feats["aspect_ratio"] = g["n_cols"] / max(g["n_rows"], 1)

    L = [[g["lattice"][0][0], g["lattice"][0][1]],
         [g["lattice"][1][0], g["lattice"][1][1]]]
    edge_lens = []
    edge_angles = []
    for e in g["edges"]:
        src = g["vertices"][e["src"]]
        dst = g["vertices"][e["dst"]]
        src_xy = [
            src[0] * L[0][0] / g["n_cols"] + src[1] * L[1][0] / g["n_rows"],
            src[0] * L[0][1] / g["n_cols"] + src[1] * L[1][1] / g["n_rows"],
        ]
        dst_xy = [
            dst[0] * L[0][0] / g["n_cols"] + dst[1] * L[1][0] / g["n_rows"]
              + e["wrap"][0] * L[0][0] + e["wrap"][1] * L[1][0],
            dst[0] * L[0][1] / g["n_cols"] + dst[1] * L[1][1] / g["n_rows"]
              + e["wrap"][0] * L[0][1] + e["wrap"][1] * L[1][1],
        ]
        dx = dst_xy[0] - src_xy[0]
        dy = dst_xy[1] - src_xy[1]
        length = math.hypot(dx, dy)
        if length > 1e-9:
            edge_lens.append(length)
            ang = math.atan2(dy, dx)
            if ang < 0:
                ang += math.pi
            if ang >= math.pi:
                ang -= math.pi
            edge_angles.append(ang)

    if edge_lens:
        feats["mean_edge_length"] = statistics.mean(edge_lens)
        feats["std_edge_length"] = (statistics.stdev(edge_lens)
                                       if len(edge_lens) > 1 else 0.0)
        feats["min_edge_length"] = min(edge_lens)
        feats["max_edge_length"] = max(edge_lens)
        feats["edge_length_ratio"] = max(edge_lens) / min(edge_lens)
        feats["edge_density"] = len(g["edges"]) / max(feats["cell_area"], 1e-9)

    if edge_angles:
        cos2 = [math.cos(2 * a) for a in edge_angles]
        feats["axial_alignment"] = statistics.mean(cos2)
        feats["axial_alignment_abs"] = abs(statistics.mean(cos2))
        bins = Counter(round(math.degrees(a) / 15) * 15 for a in edge_angles)
        feats["n_orientation_bins"] = len(bins)
        feats["orientation_concentration"] = max(bins.values()) / len(edge_angles)

    if g.get("beam"):
        feats["nu_anisotropy_beam"] = (g["beam"]["nu_max"][1] -
                                         g["beam"]["nu_min"][1])
        feats["nu_min_beam_abs"] = abs(g["beam"]["nu_min"][1] or 0)
    if g.get("phonon"):
        feats["acoustic_min"] = g["phonon"].get("acoustic_min", 0)
        feats["v_anisotropy"] = g["phonon"].get("v_anisotropy", 0)
        feats["flat_acoustic_score"] = g["phonon"].get("flat_acoustic_score", 0)

    return feats


def cohens_d(group_a, group_b):
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mu_a = statistics.mean(group_a)
    mu_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)
    pooled_sd = math.sqrt(0.5 * (var_a + var_b))
    if pooled_sd < 1e-12:
        return 0.0
    return (mu_a - mu_b) / pooled_sd


def main():
    atlas_path = sys.argv[1] if len(sys.argv) > 1 else "docs/atlas.json"
    with open(atlas_path) as f:
        atlas = json.load(f)
    grounds = atlas["grounds"]
    print(f"Loaded atlas with {len(grounds)} grounds")

    rows = []
    for g in grounds:
        nu = g["beam"]["nu_min"][1] if g.get("beam") else None
        eta = g["humidity"]["eta_pore"] if g.get("humidity") else None
        if nu is None or eta is None or eta != eta:
            continue
        key = f"{g['family']}/{g['name']}"
        feats = derive_features(g)
        rows.append({"key": key, "nu_min": nu, "eta_pore": eta, "features": feats})

    print(f"  {len(rows)} grounds with both nu_min and eta_pore data")

    nus = sorted(r["nu_min"] for r in rows)
    etas = sorted(r["eta_pore"] for r in rows)
    nu_med = nus[len(nus) // 2]
    eta_med = etas[len(etas) // 2]
    print(f"\nMedians: nu_min = {nu_med:+.3f}, eta_pore = {eta_med:+.3f}")

    class_rows = [r for r in rows
                    if r["nu_min"] >= nu_med and r["eta_pore"] < eta_med]
    rest_rows = [r for r in rows if r not in class_rows]
    print(f"\nHumidity-active but weakly auxetic class: {len(class_rows)} grounds")
    print(f"Rest: {len(rest_rows)} grounds")

    sample_feats = (list(class_rows[0]["features"].keys())
                     if class_rows else [])
    effects = []
    for col in sample_feats:
        cls = [r["features"].get(col) for r in class_rows
                if r["features"].get(col) is not None]
        rst = [r["features"].get(col) for r in rest_rows
                if r["features"].get(col) is not None]
        if len(cls) < 2 or len(rst) < 2:
            continue
        d = cohens_d(cls, rst)
        effects.append((col, d, statistics.mean(cls), statistics.mean(rst)))

    effects.sort(key=lambda e: -abs(e[1]))

    print("\n" + "=" * 78)
    print("Top distinguishing features (by Cohen's d)")
    print("(|d|>1.5 = large; |d|>0.8 = medium; |d|>0.3 = small)")
    print("=" * 78)
    print(f"{'Feature':<32} {'Cohen d':>9}   {'class':>10}  {'rest':>10}  effect")
    print("-" * 78)
    for col, d, mc, mr in effects[:18]:
        if abs(d) > 1.5:
            marker = "LARGE"
        elif abs(d) > 0.8:
            marker = "medium"
        elif abs(d) > 0.3:
            marker = "small"
        else:
            marker = ""
        print(f"{col:<32} {d:>+9.3f}   {mc:>+10.3f}  {mr:>+10.3f}  {marker}")

    if effects:
        top_features = [e[0] for e in effects[:5]]
        top_class = sorted(class_rows, key=lambda r: r["eta_pore"])[:10]
        print("\n" + "=" * 78)
        print("Top-10 grounds in the class (most-contracting eta_pore first)")
        print("Showing the 5 most distinguishing features:")
        print("=" * 78)
        header = (f"{'Ground':<22} {'nu_min':>7} {'eta_pore':>9}  "
                  + " ".join(f"{f[:11]:>11}" for f in top_features))
        print(header)
        print("-" * len(header))
        for r in top_class:
            row = (f"{r['key']:<22} {r['nu_min']:>+7.2f} {r['eta_pore']:>+9.3f}  "
                   + " ".join(f"{(r['features'].get(f) or 0.0):>11.3f}"
                                for f in top_features))
            print(row)


if __name__ == "__main__":
    main()
