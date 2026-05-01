"""
compute_submission.py
=====================

Compute the full mechanical analysis (spring + beam, parameter sweep)
for a user-submitted graph and output an atlas-compatible record.

This is a thin wrapper around `build_atlas.build_ground_record` that
takes a submission JSON, builds a `LaceGraph`, runs the mechanics
pipeline, and writes the result as a JSON record ready to merge into
atlas.json.

USAGE:
    python3 compute_submission.py submission.json --output record.json
    python3 compute_submission.py submission.json --output record.json \\
        --thumbnails-out thumbnails/user/my_pattern/

OUTPUT RECORD STRUCTURE (matches atlas.json grounds[i] schema):
    {
      "name": "...", "family": "...",
      "n_rows": ..., "n_cols": ...,
      "vertices": [...], "edges": [...],
      "lattice": [...], "cell_area": ...,
      "thumbnails": {"lace": "...", "deformed": "..."},  # if --thumbnails-out
      "spring": { ... },  # parameter-grid sweep
      "beam":   { ... },
      "submission_metadata": {            # added for submissions
        "submitted_via": "validate_submission.py",
        "validation": {...},
        "tesselace_compliant_asserted": ...,
        "tesselace_compliant_verified": ...,
        "provenance": {...}
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

from ..parse_to_graph import LaceGraph, Edge
from ..build_atlas import build_ground_record, SPRING_K_ANG_GRID, BEAM_AR_GRID
from .validate_submission import validate_submission


def submission_to_lacegraph(submission: Dict[str, Any]) -> LaceGraph:
    """Convert a validated submission dict to a LaceGraph."""
    vertices = [tuple(v) for v in submission["vertices"]]
    edges = []
    for e in submission["edges"]:
        edges.append(Edge(
            src_idx=e["src"],
            dst_idx=e["dst"],
            wrap=tuple(e["wrap"]),
            polyline=(),  # not used by mechanics
        ))
    return LaceGraph(
        name=submission["name"],
        family=submission.get("family", "user/contributed"),
        keyword="submission",
        n_rows=submission["n_rows"],
        n_cols=submission["n_cols"],
        vertices=vertices,
        edges=edges,
    )


def maybe_render_thumbnails(graph: LaceGraph, out_dir: str
                              ) -> Optional[Dict[str, str]]:
    """If render scripts and matplotlib are available, render the lace
    pair-diagram and deformed-overlay PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from auxetic_lace.render_lace_view import render_ground_lace_views
        from auxetic_lace.render_lace_deformed import render_ground_deformed
    except ImportError as exc:
        print(f"  [thumbnails] not rendered: {exc}", file=sys.stderr)
        return None

    os.makedirs(out_dir, exist_ok=True)
    lace_path = os.path.join(out_dir, "lace.png")
    deformed_path = os.path.join(out_dir, "deformed.png")
    try:
        render_ground_lace_views(graph, lace_path, n_tiles=3)
    except Exception as exc:
        print(f"  [thumbnails] lace render failed: {exc}", file=sys.stderr)
        lace_path = None
    try:
        render_ground_deformed(graph, deformed_path)
    except Exception as exc:
        print(f"  [thumbnails] deformed render failed: {exc}", file=sys.stderr)
        deformed_path = None

    if not lace_path and not deformed_path:
        return None
    return {
        "lace": lace_path or "",
        "deformed": deformed_path or "",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute mechanics for a user-submitted graph and "
                    "output an atlas-compatible record JSON.")
    ap.add_argument("submission", help="Path to submission JSON")
    ap.add_argument("--output", "-o", default=None,
                    help="Output record JSON path (default: stdout)")
    ap.add_argument("--thumbnails-out", default=None,
                    help="Directory to render lace.png + deformed.png into")
    ap.add_argument("--thumbnail-url-prefix", default=None,
                    help="URL prefix for the thumbnail field in the output "
                         "record (default: same as --thumbnails-out)")
    ap.add_argument("--skip-validation", action="store_true",
                    help="Skip pre-computation validation (NOT RECOMMENDED)")
    ap.add_argument("--check-tesselace", action="store_true",
                    help="Run tesselace-property checks during validation")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Load
    try:
        with open(args.submission) as f:
            submission = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"FAIL: could not read submission JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    # Validate
    validation_results = None
    if not args.skip_validation:
        passed, results = validate_submission(
            submission, check_tesselace=args.check_tesselace, strict=False)
        validation_results = results
        if not passed:
            print("FAIL: submission did not pass validation:", file=sys.stderr)
            for err in results.get("level1", []):
                print(f"  - {err}", file=sys.stderr)
            for err in results.get("level2", []):
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

    # Compute
    graph = submission_to_lacegraph(submission)
    if args.verbose:
        print(f"  Computing mechanics for "
              f"{graph.family}/{graph.name} "
              f"({graph.n_rows}x{graph.n_cols}, "
              f"{len(graph.vertices)} verts, {len(graph.edges)} edges)...",
              file=sys.stderr)
    record = build_ground_record(
        graph,
        name=submission["name"],
        family=submission.get("family", "user/contributed"),
        thumbnail_dir=None)  # we'll handle thumbnails specially

    # Thumbnails
    if args.thumbnails_out:
        thumbs = maybe_render_thumbnails(graph, args.thumbnails_out)
        if thumbs:
            url_prefix = args.thumbnail_url_prefix or args.thumbnails_out
            record["thumbnails"] = {
                "lace": os.path.join(url_prefix, "lace.png"),
                "deformed": os.path.join(url_prefix, "deformed.png"),
            }

    # Submission metadata
    tesselace_verified = (
        not validation_results.get("level2", [])
        if validation_results and args.check_tesselace
        else None
    )
    record["submission_metadata"] = {
        "submitted_via": "compute_submission.py",
        "tesselace_compliant_asserted": submission.get(
            "tesselace_compliant", False),
        "tesselace_compliant_verified": tesselace_verified,
        "provenance": submission.get("provenance", {}),
    }

    # Write
    output_text = json.dumps(record, separators=(",", ":"))
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        if args.verbose:
            size_kb = os.path.getsize(args.output) / 1024
            print(f"  Wrote {args.output} ({size_kb:.1f} KB)", file=sys.stderr)
    else:
        sys.stdout.write(output_text)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
