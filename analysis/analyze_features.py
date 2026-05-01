"""
analyze_features.py
===================

Read features.csv produced by extract_features.py, run a regression of
nu_min_beam on the structural / family / mechanical descriptors, and print:

  - Held-out R^2 from a cross-validated random forest (predicting how well
    structure predicts nu_min)
  - Feature importance (which descriptors carry the predictive signal)
  - A separate classification analysis for the homogeneous-auxetic case
    (rare class, deserves a dedicated treatment)

We keep two regressions:
  (A) Predict nu_min_beam from structural features alone (no mechanical
      derived features). This is the test that decides what kind of
      generator we need to build later: high R^2 here means topology
      determines auxeticity and we can train a GNN; low R^2 means we'd
      need to include geometry/mechanics in any surrogate.
  (B) Predict nu_min_beam including mechanical features (K, G, K/G).
      This is a sanity check / upper bound — if (B) is much higher than
      (A), then nothing in the topology alone tells us much, and we have
      to compute spring/beam ν to learn anything.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import r2_score, classification_report


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "features.csv"
    df = pd.read_csv(in_path)

    print(f"Loaded {len(df)} rows from {in_path}")
    print(f"Columns: {len(df.columns)}\n")

    # Define feature groups
    face_count_cols = [c for c in df.columns if c.startswith("n_faces_")]
    family_token_cols = [c for c in df.columns if c.startswith("fam_token_")]
    geom_cols = [
        "mean_edge_length", "std_edge_length", "min_edge_length", "max_edge_length",
        "n_reentrant_angles", "n_reentrant_vertices", "frac_reentrant_vertices",
        "mean_vertex_min_angle", "mean_vertex_max_angle",
        "cell_area", "edge_density", "n_distinct_face_sizes",
        "mean_face_size", "std_face_size",
        "n_vertices", "n_edges", "n_faces", "n_rows", "n_cols",
    ]
    mech_cols = ["K_beam", "G_beam", "K_over_G_beam", "nu_anisotropy_beam"]

    structural_features = face_count_cols + family_token_cols + geom_cols
    full_features = structural_features + mech_cols

    label = "nu_min_beam"
    df = df.dropna(subset=[label] + structural_features)
    print(f"After dropping rows with missing label or features: {len(df)}\n")

    y = df[label].values

    print("=" * 70)
    print("(A) Predict nu_min_beam from STRUCTURAL features alone")
    print("=" * 70)
    print(f"  features: {len(structural_features)} "
          f"({len(face_count_cols)} face counts + {len(family_token_cols)} family "
          f"tokens + {len(geom_cols)} geometry)")
    Xa = df[structural_features].values
    rfa = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1,
                                 oob_score=True)
    cv_a = cross_val_score(rfa, Xa, y, cv=KFold(5, shuffle=True, random_state=0),
                            scoring="r2", n_jobs=-1)
    rfa.fit(Xa, y)
    print(f"  5-fold CV R^2 = {cv_a.mean():.3f}  ±  {cv_a.std():.3f}")
    print(f"  OOB R^2       = {rfa.oob_score_:.3f}")

    # Feature importances (top 15)
    imp_a = sorted(zip(structural_features, rfa.feature_importances_),
                    key=lambda kv: -kv[1])
    print(f"\n  top 15 features by Gini importance:")
    for name, val in imp_a[:15]:
        print(f"    {name:30s} {val:.4f}")

    print()
    print("=" * 70)
    print("(B) Predict nu_min_beam INCLUDING mechanical features (sanity check)")
    print("=" * 70)
    df2 = df.dropna(subset=mech_cols)
    print(f"  rows after dropping NaN mech: {len(df2)}")
    Xb = df2[full_features].values
    yb = df2[label].values
    rfb = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1,
                                 oob_score=True)
    cv_b = cross_val_score(rfb, Xb, yb,
                            cv=KFold(5, shuffle=True, random_state=0),
                            scoring="r2", n_jobs=-1)
    rfb.fit(Xb, yb)
    print(f"  5-fold CV R^2 = {cv_b.mean():.3f}  ±  {cv_b.std():.3f}")
    print(f"  OOB R^2       = {rfb.oob_score_:.3f}")
    imp_b = sorted(zip(full_features, rfb.feature_importances_),
                    key=lambda kv: -kv[1])
    print(f"\n  top 10 features by Gini importance:")
    for name, val in imp_b[:10]:
        print(f"    {name:30s} {val:.4f}")

    print()
    print("=" * 70)
    print("(C) CLASSIFICATION: predict homogeneous-auxetic from structure alone")
    print("=" * 70)
    yc = df["is_homogeneous_auxetic"].values
    print(f"  n_homogeneous = {int(yc.sum())} / {len(yc)}")
    Xc = df[structural_features].values
    rfc = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1,
                                  class_weight="balanced", oob_score=True)
    # Use stratified CV because positive class is rare
    if yc.sum() >= 5:
        cv_c = cross_val_score(
            rfc, Xc, yc,
            cv=StratifiedKFold(5, shuffle=True, random_state=0),
            scoring="balanced_accuracy", n_jobs=-1)
        rfc.fit(Xc, yc)
        print(f"  5-fold stratified CV balanced accuracy = "
              f"{cv_c.mean():.3f}  ±  {cv_c.std():.3f}")
        print(f"  OOB score = {rfc.oob_score_:.3f}")
        imp_c = sorted(zip(structural_features, rfc.feature_importances_),
                        key=lambda kv: -kv[1])
        print(f"\n  top 10 features for homogeneous-auxetic identification:")
        for name, val in imp_c[:10]:
            print(f"    {name:30s} {val:.4f}")
    else:
        print("  too few positives for cross-validation")

    print()
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
  If (A) R^2 > 0.6:  Topology alone strongly predicts nu_min. A GNN
                     surrogate trained on (graph -> nu_min) should work,
                     and a generator that produces topologies similar to
                     the 321 (with or without strict workability) gives
                     useful training data.

  If (A) R^2 in [0.3, 0.6]: Structure carries real signal but geometry /
                     mechanics matter too. The generator needs to emit
                     equilibrium positions and the surrogate needs to see
                     them. Workability may matter more.

  If (A) R^2 < 0.3:  Coarse structural features alone are not enough.
                     Either we need much richer descriptors (symmetry
                     group, explicit motif detection) or the prediction
                     genuinely requires running spring / beam mechanics.
                     In that case the surrogate has to be trained on
                     mechanical labels, not structural ones — and the
                     generator must produce data we can mechanically
                     evaluate cheaply (spring model on tens of thousands
                     of grounds).
""")


if __name__ == "__main__":
    main()
