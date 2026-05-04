"""
render_taylor_thumbnails.py

Render deformed.png and dispersion.png for every Taylor ground in
atlas.json. The existing build_thumbnails.py reads from the TesseLace
manifest (so only knows about Irvine grounds); this driver walks the
atlas instead and renders only what's missing.

Idempotent: skips files that already exist (override with --force).

Usage:
    python3 render_taylor_thumbnails.py            # render all missing
    python3 render_taylor_thumbnails.py --dry-run  # show what would render
    python3 render_taylor_thumbnails.py --limit 3  # smoke test
    python3 render_taylor_thumbnails.py --force    # re-render even if exists
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Headless matplotlib (must be set before importing the renderers)
import matplotlib
matplotlib.use("Agg")

import json

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.parse_to_graph import LaceGraph, Edge       # noqa: E402
from auxetic_lace.render_lace_deformed import render_ground_deformed  # noqa: E402
from auxetic_lace.render_dispersion import render_ground_dispersion  # noqa: E402


ATLAS_PATH = Path("docs/atlas.json")
THUMBS_DIR = Path("docs/thumbnails")


def reconstruct_lacegraph(g_record: dict) -> LaceGraph:
    """Rebuild a LaceGraph from a per-ground atlas record."""
    vertices = [tuple(v) for v in g_record["vertices"]]
    edges = []
    for e in g_record["edges"]:
        polyline = tuple(tuple(p) for p in e.get("polyline", []) or [])
        edges.append(Edge(
            src_idx=e["src"],
            dst_idx=e["dst"],
            wrap=tuple(e["wrap"]),
            polyline=polyline,
        ))
    return LaceGraph(
        name=g_record["name"],
        family=g_record.get("family", ""),
        keyword=g_record.get("keyword", ""),
        n_rows=g_record["n_rows"],
        n_cols=g_record["n_cols"],
        vertices=vertices,
        edges=edges,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(ATLAS_PATH))
    ap.add_argument("--thumbs-dir", default=str(THUMBS_DIR))
    ap.add_argument("--limit", type=int, default=None,
                    help="Render at most N Taylor grounds (smoke test).")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if PNG already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without rendering.")
    ap.add_argument("--skip-deformed", action="store_true")
    ap.add_argument("--skip-dispersion", action="store_true")
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    thumbs_dir = Path(args.thumbs_dir)

    if not atlas_path.exists():
        print(f"ERROR: atlas not found at {atlas_path}", file=sys.stderr)
        sys.exit(2)

    with atlas_path.open() as f:
        atlas = json.load(f)

    # Find Taylor grounds
    taylor = [g for g in atlas["grounds"] if g.get("source") == "taylor"]
    print(f"Atlas: {atlas_path}")
    print(f"Taylor grounds: {len(taylor)}")

    # Build a render plan: skip files that exist unless --force
    plan = []   # list of (g, deformed_path, dispersion_path, what)
    skipped_exist = 0
    for g in taylor:
        family = g["family"]
        name = g["name"]
        ground_dir = thumbs_dir / family / name
        deformed_path = ground_dir / "deformed.png"
        dispersion_path = ground_dir / "dispersion.png"

        do_def = (not args.skip_deformed
                   and (args.force or not deformed_path.exists()))
        do_disp = (not args.skip_dispersion
                    and (args.force or not dispersion_path.exists()))

        if not do_def and not do_disp:
            skipped_exist += 1
            continue
        plan.append((g, deformed_path, dispersion_path, do_def, do_disp))

    print(f"Already complete:  {skipped_exist}")
    print(f"To render:         {len(plan)}")

    # Stats by family
    if plan:
        from collections import Counter
        fam_counts = Counter(g["family"] for g, *_ in plan)
        print("\nBy family:")
        for fam, c in sorted(fam_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {fam:14s} {c}")

    if args.limit is not None and args.limit < len(plan):
        plan = plan[:args.limit]
        print(f"\nLimiting to first {len(plan)} (smoke test).")

    if args.dry_run:
        print(f"\n(dry run — nothing rendered)")
        print(f"\nFirst 5:")
        for g, dp, dpp, do_def, do_disp in plan[:5]:
            todo = []
            if do_def: todo.append("deformed")
            if do_disp: todo.append("dispersion")
            print(f"  {g['family']}/{g['name']}: {', '.join(todo)}")
        return

    # Render
    print()
    failures = []
    t0 = time.time()
    for i, (g, deformed_path, dispersion_path, do_def, do_disp) in enumerate(plan):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(plan) - i) if i > 0 else 0
        print(f"  [{i+1}/{len(plan)}] {g['family']}/{g['name']} "
              f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        try:
            graph = reconstruct_lacegraph(g)
        except Exception as e:
            failures.append((g["name"], "reconstruct",
                              f"{type(e).__name__}: {e}"))
            continue

        deformed_path.parent.mkdir(parents=True, exist_ok=True)

        if do_def:
            try:
                render_ground_deformed(graph, str(deformed_path))
            except Exception as e:
                failures.append((g["name"], "deformed",
                                  f"{type(e).__name__}: {e}"))

        if do_disp:
            try:
                render_ground_dispersion(graph, str(dispersion_path))
            except Exception as e:
                failures.append((g["name"], "dispersion",
                                  f"{type(e).__name__}: {e}"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    if failures:
        print(f"\n{len(failures)} failures:")
        for nm, kind, msg in failures[:10]:
            print(f"  {nm} ({kind}): {msg}")


if __name__ == "__main__":
    main()
