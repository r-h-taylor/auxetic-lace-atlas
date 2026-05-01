"""
validate_submission.py
======================

Validate a graph-submission JSON file against the schema and structural
invariants required by the Auxetic Lace Atlas pipeline.

Two levels of check:

  Level 1 (basic, always run):
    - JSON schema (vertex coords in range, edges reference valid vertex
      indices, all required fields present)
    - 2-regular invariant: every vertex has in-degree = out-degree = 2
    - All edges have nonzero length in the embedded geometry
    - Underlying graph is connected on the torus

  Level 2 (tesselace properties, only run if --check-tesselace):
    - Partial-order property (no contractible directed cycles):
      checked via rotational consistency at every vertex
    - Thread-conservation property: osculating partition exists and
      every circuit has wrap (1, 0)

USAGE:
    python3 validate_submission.py path/to/submission.json
    python3 validate_submission.py --check-tesselace submission.json
    python3 validate_submission.py --strict submission.json
        # treat tesselace failures as errors (else: warnings if
        # `tesselace_compliant: true` is asserted)

EXIT CODES:
    0 = passed all checks
    1 = level 1 errors
    2 = level 1 ok, level 2 errors (only with --strict or self-asserted
        tesselace_compliant: true)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from .submission_schema import validate_against_schema


# =====================================================================
# Level 1: structural invariants
# =====================================================================

def check_2regular(submission: Dict[str, Any]) -> List[str]:
    """Verify every vertex has in-degree 2 and out-degree 2."""
    errors = []
    n_verts = len(submission["vertices"])
    in_deg = [0] * n_verts
    out_deg = [0] * n_verts
    for e in submission["edges"]:
        out_deg[e["src"]] += 1
        in_deg[e["dst"]] += 1
    for i in range(n_verts):
        if out_deg[i] != 2:
            errors.append(
                f"vertex {i} ({submission['vertices'][i]}): "
                f"out-degree = {out_deg[i]}, must be 2")
        if in_deg[i] != 2:
            errors.append(
                f"vertex {i} ({submission['vertices'][i]}): "
                f"in-degree = {in_deg[i]}, must be 2")
    return errors


def check_nonzero_lengths(submission: Dict[str, Any]) -> List[str]:
    """Verify every edge has nonzero length in the embedded geometry."""
    errors = []
    n_rows = submission["n_rows"]
    n_cols = submission["n_cols"]
    L = submission.get("lattice_vectors", [[1.0, 0.0], [0.0, 1.0]])
    a1 = L[0]
    a2 = L[1]
    verts = submission["vertices"]
    for i, e in enumerate(submission["edges"]):
        sc, sr = verts[e["src"]]
        dc, dr = verts[e["dst"]]
        wc, wr = e["wrap"]
        # Cartesian displacement
        # dst_world = a1 * (dc + wc * n_cols) + a2 * (dr + wr * n_rows)
        # src_world = a1 * sc + a2 * sr
        delta_uc = (dc + wc * n_cols - sc, dr + wr * n_rows - sr)
        dx = a1[0] * delta_uc[0] + a2[0] * delta_uc[1]
        dy = a1[1] * delta_uc[0] + a2[1] * delta_uc[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            errors.append(
                f"edge {i} ({e['src']} -> {e['dst']} wrap={e['wrap']}): "
                f"zero length (probably a self-loop with wrap = (0, 0))")
    return errors


def check_connected(submission: Dict[str, Any]) -> List[str]:
    """Underlying undirected graph must be connected on the torus."""
    errors = []
    n_verts = len(submission["vertices"])
    if n_verts == 0:
        return ["no vertices"]
    # Union-find
    parent = list(range(n_verts))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b
    for e in submission["edges"]:
        union(e["src"], e["dst"])
    roots = {find(i) for i in range(n_verts)}
    if len(roots) > 1:
        components = {}
        for i in range(n_verts):
            r = find(i)
            components.setdefault(r, []).append(i)
        errors.append(
            f"graph is not connected: {len(roots)} components: "
            f"{[sorted(comp) for comp in components.values()]}")
    return errors


# =====================================================================
# Level 2: tesselace properties
# =====================================================================

def check_rotational_consistency(submission: Dict[str, Any]) -> List[str]:
    """Property 3.2.4 (no contractible directed cycles): edges at each
    vertex must be rotationally 'consecutive' (sorted by angle, the
    cyclic in/out type sequence has at most 2 transitions)."""
    errors = []
    verts = submission["vertices"]
    n_rows = submission["n_rows"]
    n_cols = submission["n_cols"]
    L = submission.get("lattice_vectors", [[1.0, 0.0], [0.0, 1.0]])
    a1 = L[0]
    a2 = L[1]

    # For each vertex, list outgoing direction unit vectors and incoming
    # direction unit vectors (pointing AWAY from the vertex)
    by_vertex_out: Dict[int, List[Tuple[float, float]]] = {}
    by_vertex_in_pointing_away: Dict[int, List[Tuple[float, float]]] = {}
    for e in submission["edges"]:
        src = e["src"]; dst = e["dst"]; w = e["wrap"]
        sc, sr = verts[src]
        dc, dr = verts[dst]
        delta_uc = (dc + w[0] * n_cols - sc, dr + w[1] * n_rows - sr)
        dx = a1[0] * delta_uc[0] + a2[0] * delta_uc[1]
        dy = a1[1] * delta_uc[0] + a2[1] * delta_uc[1]
        L_e = math.sqrt(dx * dx + dy * dy)
        if L_e < 1e-12:
            continue
        # Outgoing at src in direction (dx, dy)
        by_vertex_out.setdefault(src, []).append((dx / L_e, dy / L_e))
        # Incoming at dst, pointing away (back toward src) = (-dx, -dy)
        by_vertex_in_pointing_away.setdefault(
            dst, []).append((-dx / L_e, -dy / L_e))

    for i in range(len(verts)):
        out_dirs = by_vertex_out.get(i, [])
        in_dirs = by_vertex_in_pointing_away.get(i, [])
        n = len(out_dirs) + len(in_dirs)
        if n != 4:
            errors.append(
                f"vertex {i}: total incident edges = {n}, must be 4 for 2-regular")
            continue
        items = ([(math.atan2(d[1], d[0]), 'out') for d in out_dirs] +
                 [(math.atan2(d[1], d[0]), 'in') for d in in_dirs])
        items.sort()
        types = [t for (_, t) in items]
        transitions = sum(
            1 for k in range(len(types))
            if types[k] != types[(k + 1) % len(types)])
        if transitions > 2:
            errors.append(
                f"vertex {i}: rotationally alternating "
                f"(types around vertex = {types}, {transitions} transitions). "
                f"This violates the partial-order property and would create "
                f"contractible directed cycles.")
    return errors


def check_thread_conservation(submission: Dict[str, Any]) -> List[str]:
    """Property 3.2.5: the embedding can be partitioned into osculating
    circuits, each of which wraps once around the meridian and zero
    times around the longitude.

    We rely on the canonical algorithm in enum_v2/brute_force.py."""
    errors = []
    try:
        # Build a Ground from the submission and use brute_force checks
        from auxetic_lace.canonical.brute_force import (
            Ground, _osculating_partition, _check_thread_conserving)
    except ImportError as exc:
        errors.append(f"could not import enum_v2.brute_force: {exc}. "
                       "Skipping thread-conservation check.")
        return errors

    # Build ground edges in the (sc, sr, dx_step, dy_step) format expected
    # by Ground. The step (dx, dy) here is the per-edge displacement in
    # unit-cell coords (NOT the position-space delta).
    verts = submission["vertices"]
    n_rows = submission["n_rows"]
    n_cols = submission["n_cols"]
    edges = set()
    for e in submission["edges"]:
        sc, sr = verts[e["src"]]
        dc, dr = verts[e["dst"]]
        # Step vector in unit-cell coords from src to dst (not wrapped):
        # we want the "displacement step" the lace path takes.
        dx_step = dc - sc + e["wrap"][0] * n_cols
        dy_step = dr - sr + e["wrap"][1] * n_rows
        edges.add((sc, sr, dx_step, dy_step))
    g = Ground(n_cols=n_cols, n_rows=n_rows, edges=frozenset(edges))

    if not _check_thread_conserving(g):
        errors.append(
            "thread-conservation property fails: at least one osculating "
            "circuit does not have wrap (1, 0). The embedding is not a "
            "tesselace ground.")
    return errors


# =====================================================================
# Top-level
# =====================================================================

def validate_submission(submission: Dict[str, Any],
                         check_tesselace: bool = False,
                         strict: bool = False
                         ) -> Tuple[bool, Dict[str, List[str]]]:
    """Run all checks on a submission. Returns (passed, results) where
    results is a dict with 'level1' and 'level2' lists of error messages.
    """
    results: Dict[str, List[str]] = {"level1": [], "level2": []}

    # Schema
    schema_errs = validate_against_schema(submission)
    results["level1"].extend(schema_errs)
    if schema_errs:
        return (False, results)

    # Level 1
    results["level1"].extend(check_2regular(submission))
    results["level1"].extend(check_nonzero_lengths(submission))
    results["level1"].extend(check_connected(submission))
    if results["level1"]:
        return (False, results)

    # Level 2 (only if requested or self-asserted)
    self_asserted = submission.get("tesselace_compliant", False)
    if check_tesselace or self_asserted:
        results["level2"].extend(check_rotational_consistency(submission))
        results["level2"].extend(check_thread_conservation(submission))

    if strict and results["level2"]:
        return (False, results)
    if self_asserted and results["level2"]:
        results["level1"].append(
            "tesselace_compliant: true was asserted but tesselace checks "
            "failed. Either fix the graph or set tesselace_compliant: false.")
        return (False, results)

    return (True, results)


def main():
    ap = argparse.ArgumentParser(
        description="Validate a graph-submission JSON file.")
    ap.add_argument("path", help="Path to submission JSON")
    ap.add_argument("--check-tesselace", action="store_true",
                    help="Also run the tesselace-property checks")
    ap.add_argument("--strict", action="store_true",
                    help="Treat tesselace failures as errors (exit 2)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        with open(args.path) as f:
            submission = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"FAIL: could not read JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    passed, results = validate_submission(
        submission, check_tesselace=args.check_tesselace, strict=args.strict)

    if not args.quiet:
        if results["level1"]:
            print("Level 1 errors (structural):")
            for err in results["level1"]:
                print(f"  - {err}")
        if results["level2"]:
            label = "errors" if (args.strict or
                                 submission.get("tesselace_compliant", False)
                                 ) else "warnings"
            print(f"Level 2 {label} (tesselace properties):")
            for err in results["level2"]:
                print(f"  - {err}")
        if passed:
            tag = "tesselace-compliant" if (args.check_tesselace and
                                              not results["level2"]
                                              ) else "valid"
            print(f"PASS: {args.path} is {tag}")
        else:
            print(f"FAIL: {args.path}", file=sys.stderr)

    if not passed:
        sys.exit(1 if results["level1"] else 2)


if __name__ == "__main__":
    main()
