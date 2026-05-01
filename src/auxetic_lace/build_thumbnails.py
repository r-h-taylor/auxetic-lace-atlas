"""
build_thumbnails.py
===================

Pre-renders PNG thumbnails for every ground in the TesseLace catalog,
producing two artifacts per ground:

  thumbnails/{family}/{name}/lace.png       — pair diagram + thread sketch
                                                (rest geometry, on cream paper)
  thumbnails/{family}/{name}/deformed.png   — rest + deformed overlay
                                                under tension along most-auxetic axis

These are referenced from atlas.json so the visualizer can show them
inline as part of each ground's record.

USAGE:
    python3 build_thumbnails.py
    python3 build_thumbnails.py --catalog tesselace_catalog
    python3 build_thumbnails.py --output-dir thumbnails
    python3 build_thumbnails.py --limit 10
    python3 build_thumbnails.py --filter-family 3_6
    python3 build_thumbnails.py --skip-deformed   # only lace views (faster)
    python3 build_thumbnails.py --skip-lace       # only deformed overlays

Note: the deformed overlay requires running mechanics on each ground to
identify the most-auxetic loading direction. This adds ~0.5-1 sec per
ground compared to the lace-view-only render. Total runtime for the
full 321-ground catalog is ~5-10 min depending on hardware.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")

from .parse_to_graph import parse_file, parse_manifest
from .render_lace_view import render_ground_lace_views, render_ground_lace_views_split
from .render_lace_deformed import render_ground_deformed
from .render_dispersion import render_ground_dispersion


def thumbnail_paths(out_dir: str, family: str, name: str
                    ) -> Dict[str, str]:
    """Standard layout for thumbnails."""
    base = os.path.join(out_dir, family, name)
    return {
        "dir": base,
        "lace": os.path.join(base, "lace.png"),
        "pair": os.path.join(base, "pair.png"),
        "thread": os.path.join(base, "thread.png"),
        "deformed": os.path.join(base, "deformed.png"),
        "dispersion": os.path.join(base, "dispersion.png"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Pre-render PNG thumbnails for every ground in the catalog."
    )
    ap.add_argument("--catalog", default="tesselace_catalog",
                    help="Path to scraped catalog directory containing manifest.csv")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write thumbnails. Default: <catalog>/thumbnails")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter-family", type=str, default=None)
    ap.add_argument("--skip-lace", action="store_true",
                    help="Don't render the lace pair-diagram view")
    ap.add_argument("--skip-deformed", action="store_true",
                    help="Don't render the deformed-overlay view")
    ap.add_argument("--skip-dispersion", action="store_true",
                    help="Don't render the phonon-dispersion view")
    ap.add_argument("--n-tiles", type=int, default=3,
                    help="Tile count for the lace pair diagram (default 3 -> 3x3 swatch)")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if PNG already exists")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    output_dir = args.output_dir or os.path.join(args.catalog, "thumbnails")
    manifest_path = os.path.join(args.catalog, "manifest.csv")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("       Run scrape_tesselace.py first.", file=sys.stderr)
        sys.exit(1)

    manifest = parse_manifest(manifest_path)
    if args.filter_family:
        manifest = [m for m in manifest if m['family'] == args.filter_family]
    if args.limit is not None:
        manifest = manifest[:args.limit]

    print(f"Rendering thumbnails for {len(manifest)} grounds -> {output_dir}")
    if args.skip_lace:
        print("  (skipping lace views)")
    if args.skip_deformed:
        print("  (skipping deformed views)")
    if args.skip_dispersion:
        print("  (skipping dispersion views)")

    os.makedirs(output_dir, exist_ok=True)

    rendered_lace = 0
    rendered_deformed = 0
    rendered_dispersion = 0
    skipped = 0
    failed: List[Dict[str, str]] = []
    t0 = time.time()

    for i, graph in enumerate(manifest):
        family = graph.family
        name = graph.name
        thumbs = thumbnail_paths(output_dir, family, name)
        os.makedirs(thumbs["dir"], exist_ok=True)

        # Lace view
        if not args.skip_lace:
            if os.path.isfile(thumbs["pair"]) and os.path.isfile(thumbs["thread"]) and not args.force:
                skipped += 1
            else:
                try:
                    render_ground_lace_views_split(
                        graph, thumbs["pair"], thumbs["thread"],
                        n_tiles=args.n_tiles)
                    rendered_lace += 1
                except Exception as exc:
                    failed.append({"family": family, "name": name,
                                    "stage": "lace", "error": str(exc)})
                    if args.verbose:
                        print(f"  [{i+1}] LACE FAIL {family}/{name}: {exc}")

        # Deformed view
        if not args.skip_deformed:
            if os.path.isfile(thumbs["deformed"]) and not args.force:
                skipped += 1
            else:
                try:
                    render_ground_deformed(graph, thumbs["deformed"])
                    rendered_deformed += 1
                except Exception as exc:
                    failed.append({"family": family, "name": name,
                                    "stage": "deformed", "error": str(exc)})
                    if args.verbose:
                        print(f"  [{i+1}] DEFORMED FAIL {family}/{name}: {exc}")

        # Phonon dispersion view
        if not args.skip_dispersion:
            if os.path.isfile(thumbs["dispersion"]) and not args.force:
                skipped += 1
            else:
                try:
                    render_ground_dispersion(graph, thumbs["dispersion"])
                    rendered_dispersion += 1
                except Exception as exc:
                    failed.append({"family": family, "name": name,
                                    "stage": "dispersion", "error": str(exc)})
                    if args.verbose:
                        print(f"  [{i+1}] DISPERSION FAIL {family}/{name}: {exc}")

        if args.verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(manifest) - i - 1)
            print(f"  [{i+1}/{len(manifest)}] {family}/{name}  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Rendered {rendered_lace} lace views, "
          f"{rendered_deformed} deformed views, "
          f"{rendered_dispersion} dispersion views")
    print(f"  Skipped (already existed) {skipped}")
    print(f"  Failed {len(failed)}")
    if failed and args.verbose:
        print("\nFailures:")
        for f in failed[:20]:
            print(f"  {f['family']}/{f['name']} ({f['stage']}): {f['error']}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")


if __name__ == "__main__":
    main()
