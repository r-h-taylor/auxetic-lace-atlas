"""
rerender_all_deformed.py

Re-render deformed.png for every ground in atlas.json.

Why: the Voigt-rotation bug fixed today affected render_lace_deformed.py
too. Every existing deformed.png was generated with the buggy compliance
rotation, so the deformed overlays show the wrong deformation. We need
to regenerate all of them.

Idempotent and resumable via --skip-existing (default off — we want to
overwrite). Failures are logged and don't kill the run.

Usage:
    python3 rerender_all_deformed.py            # re-render all
    python3 rerender_all_deformed.py --dry-run  # plan only
    python3 rerender_all_deformed.py --limit 5  # smoke test
    python3 rerender_all_deformed.py --skip-existing  # only fill missing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Headless matplotlib (must be set before importing the renderer)
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.parse_to_graph import LaceGraph, Edge          # noqa: E402
from auxetic_lace.render_lace_deformed import render_ground_deformed  # noqa: E402


ATLAS_PATH = Path("docs/atlas.json")
THUMBS_DIR = Path("docs/thumbnails")


def reconstruct_lacegraph(g: dict) -> LaceGraph:
    vertices = [tuple(v) for v in g["vertices"]]
    edges = []
    for e in g["edges"]:
        polyline = tuple(tuple(p) for p in e.get("polyline", []) or [])
        edges.append(Edge(
            src_idx=e["src"],
            dst_idx=e["dst"],
            wrap=tuple(e["wrap"]),
            polyline=polyline,
        ))
    return LaceGraph(
        name=g["name"],
        family=g.get("family", ""),
        keyword=g.get("keyword", ""),
        n_rows=g["n_rows"],
        n_cols=g["n_cols"],
        vertices=vertices,
        edges=edges,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(ATLAS_PATH))
    ap.add_argument("--thumbs-dir", default=str(THUMBS_DIR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Only render where deformed.png doesn't already "
                         "exist. Default OFF (overwrite all) since the bug "
                         "fix invalidates all existing renders.")
    ap.add_argument("--filter-source", choices=["irvine", "taylor"],
                    default=None,
                    help="Only render grounds with this source")
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    thumbs_dir = Path(args.thumbs_dir)

    with atlas_path.open() as f:
        atlas = json.load(f)

    grounds = atlas["grounds"]
    if args.filter_source:
        grounds = [g for g in grounds
                    if g.get("source") == args.filter_source]

    print(f"Atlas: {atlas_path}")
    print(f"Total grounds to consider: {len(grounds)}")

    plan = []
    skipped_exist = 0
    for g in grounds:
        out_path = thumbs_dir / g["family"] / g["name"] / "deformed.png"
        if args.skip_existing and out_path.exists():
            skipped_exist += 1
            continue
        plan.append((g, out_path))

    print(f"Skipped (exists, skip-existing on): {skipped_exist}")
    print(f"To render:                          {len(plan)}")

    if args.limit and args.limit < len(plan):
        plan = plan[:args.limit]
        print(f"Limited to first {len(plan)} (smoke test)")

    if args.dry_run:
        print("\n(dry run)")
        for g, out_path in plan[:5]:
            print(f"  {g['family']}/{g['name']} -> {out_path}")
        return

    print()
    failures = []
    t0 = time.time()
    for i, (g, out_path) in enumerate(plan):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(plan) - i) if i > 0 else 0
        if i % 25 == 0 or i == len(plan) - 1:
            print(f"  [{i+1}/{len(plan)}] {g['family']}/{g['name']} "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        try:
            graph = reconstruct_lacegraph(g)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            render_ground_deformed(graph, str(out_path))
        except Exception as e:
            failures.append((g["name"], f"{type(e).__name__}: {e}"))

    elapsed = time.time() - t0
    print(f"\nRendered {len(plan) - len(failures)} / {len(plan)} "
          f"in {elapsed:.1f}s.")
    if failures:
        print(f"\n{len(failures)} failures:")
        for nm, msg in failures[:10]:
            print(f"  {nm}: {msg}")


if __name__ == "__main__":
    main()
