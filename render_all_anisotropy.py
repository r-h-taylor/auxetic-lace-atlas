"""
render_all_anisotropy.py

Render anisotropy.png (E and G polar plot at AR=10) for every ground
in atlas.json. Idempotent — overwrites by default since this is the
first time the plot is generated.

Usage:
    python3 render_all_anisotropy.py            # render all
    python3 render_all_anisotropy.py --dry-run
    python3 render_all_anisotropy.py --limit 5  # smoke
    python3 render_all_anisotropy.py --skip-existing  # only fill missing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.parse_to_graph import LaceGraph, Edge          # noqa: E402
from auxetic_lace.render_anisotropy import render_ground_anisotropy  # noqa: E402


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
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--ar", type=float, default=10.0,
                    help="Beam aspect ratio (default 10.0)")
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    thumbs_dir = Path(args.thumbs_dir)

    with atlas_path.open() as f:
        atlas = json.load(f)

    grounds = atlas["grounds"]
    print(f"Atlas: {atlas_path}")
    print(f"Total grounds: {len(grounds)}")

    plan = []
    skipped = 0
    for g in grounds:
        out_path = thumbs_dir / g["family"] / g["name"] / "anisotropy.png"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        plan.append((g, out_path))

    print(f"Skipped (exists): {skipped}")
    print(f"To render:        {len(plan)}")

    if args.limit and args.limit < len(plan):
        plan = plan[:args.limit]
        print(f"Limited to first {len(plan)}")

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
            render_ground_anisotropy(graph, str(out_path), ar=args.ar)
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
