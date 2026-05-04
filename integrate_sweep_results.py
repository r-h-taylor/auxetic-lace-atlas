"""
integrate_sweep_results.py
==========================

Integrate bobbin lace sweep results into atlas.json.

Pipeline:
  1. Walk sweep output (each seed dir with graph.json is a success).
  2. Reconstruct each as a LaceGraph and compute its graph_canonical.
  3. Multi-stage dedup against the existing atlas:
       - vs Irvine grounds (321)
       - vs existing Taylor grounds (39, from the original sweep)
       - vs each other (within this batch)
     Report every category.
  4. For survivors: compute face-set family, generate a fresh name
     using the V<n_v>_<rows>x<cols>_<serial> convention (continuing
     existing serial numbering per bucket), and run the full physics
     pipeline via build_ground_record (with source="taylor" overrides
     for manufacturability and provenance).
  5. Append survivors to atlas.grounds[], rebuild summary[], update
     metadata.n_grounds, atomic-write atlas.json.
  6. Copy lace.png from each sweep dir to thumbnails/<face_set>/<name>/thread.png
     so the new grounds have a thread thumbnail in the catalog grid.

Per-survivor physics is the slow step: assemble_stiffness +
homogenize on multiple parameter settings, plus phonon dispersion
and humidity. Expect roughly 30-90s per survivor on a laptop.

USAGE (from repo root):
    python3 integrate_sweep_results.py --sweep <path> --dry-run
    python3 integrate_sweep_results.py --sweep <path> --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Repo imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.parse_to_graph import LaceGraph, Edge   # noqa: E402
from auxetic_lace.canonicalize import both_canonical, graph_canonical  # noqa: E402
from auxetic_lace.compute_family import family_label             # noqa: E402
from auxetic_lace.build_atlas import build_ground_record         # noqa: E402
from auxetic_lace.manufacturability import (                      # noqa: E402
    manufacturability_block, provenance_block,
)


ATLAS_PATH = Path("docs/atlas.json")
THUMBS_DIR = Path("docs/thumbnails")
NAME_RE = re.compile(r"^V(\d+)_(\d+)x(\d+)_(\d+)$")


# -----------------------------------------------------------------------
# Sweep ingestion
# -----------------------------------------------------------------------

def reconstruct_lacegraph_from_sweep(graph_dict: dict) -> LaceGraph:
    """Build a LaceGraph from a sweep's graph.json. The sweep stores
    edges as {src, dst, wrap, polyline} (same keys the atlas uses)."""
    vertices = [tuple(v) for v in graph_dict["vertices"]]
    edges = []
    for e in graph_dict["edges"]:
        polyline = tuple(tuple(p) for p in e.get("polyline", []) or [])
        edges.append(Edge(
            src_idx=e["src"],
            dst_idx=e["dst"],
            wrap=tuple(e["wrap"]),
            polyline=polyline,
        ))
    return LaceGraph(
        name=graph_dict.get("name", "?"),
        family=graph_dict.get("family", ""),    # ignored; we'll compute fresh
        keyword=graph_dict.get("keyword", ""),
        n_rows=graph_dict["n_rows"],
        n_cols=graph_dict["n_cols"],
        vertices=vertices,
        edges=edges,
    )


def lacegraph_to_canonical_dict(graph: LaceGraph) -> dict:
    """graph_canonical() and lace_canonical() take a record-dict, not a
    LaceGraph. Build the minimal dict they need.
    """
    return {
        "vertices": [list(v) for v in graph.vertices],
        "edges": [
            {"src": e.src_idx,
             "dst": e.dst_idx,
             "wrap": list(e.wrap),
             "polyline": [list(p) for p in e.polyline]}
            for e in graph.edges
        ],
        "n_rows": graph.n_rows,
        "n_cols": graph.n_cols,
    }


def walk_sweep(sweep_root: Path) -> List[Tuple[Path, dict]]:
    """Return list of (seed_dir, graph_dict) for every successful seed."""
    out = []
    if not sweep_root.exists():
        return out
    for combo_dir in sorted(sweep_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        for seed_dir in sorted(combo_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            graph_path = seed_dir / "graph.json"
            if not graph_path.exists():
                continue
            try:
                with graph_path.open() as f:
                    graph_dict = json.load(f)
                out.append((seed_dir, graph_dict))
            except Exception as e:
                print(f"  WARN: couldn't load {graph_path}: {e}",
                      file=sys.stderr)
    return out


# -----------------------------------------------------------------------
# Atlas dedup
# -----------------------------------------------------------------------

def reconstruct_lacegraph_from_atlas(g_record: dict) -> LaceGraph:
    """Reconstruct LaceGraph from an atlas grounds[] record."""
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


def build_existing_canonicals(atlas: dict) -> Tuple[Dict[str, dict], int, int]:
    """Map graph_canonical -> ground record for every ground in atlas.

    Returns (canon_to_ground, n_from_stored, n_recomputed).

    We use the stored `graph_canonical` field if present (computed by
    build_atlas at build time). If absent, we compute it on the fly.
    """
    canon_to_ground = {}
    n_stored = 0
    n_recomputed = 0
    n_errors = 0
    for g in atlas["grounds"]:
        canon = g.get("graph_canonical")
        if canon:
            n_stored += 1
        else:
            try:
                graph = reconstruct_lacegraph_from_atlas(g)
                canon = graph_canonical(lacegraph_to_canonical_dict(graph))
                n_recomputed += 1
            except Exception as e:
                print(f"  WARN: couldn't compute canonical for "
                      f"{g.get('name', '?')}: {e}", file=sys.stderr)
                n_errors += 1
                continue
        canon_to_ground[canon] = g
    if n_errors:
        print(f"  {n_errors} atlas grounds errored on canonical computation",
              file=sys.stderr)
    return canon_to_ground, n_stored, n_recomputed


# -----------------------------------------------------------------------
# Naming
# -----------------------------------------------------------------------

def discover_serial_buckets(atlas: dict) -> Dict[Tuple[int, int, int], int]:
    """For each existing (n_v, n_rows, n_cols) bucket of Taylor grounds,
    return the next available serial number (max + 1)."""
    max_serial: Dict[Tuple[int, int, int], int] = {}
    for g in atlas["grounds"]:
        if g.get("source") != "taylor":
            continue
        m = NAME_RE.match(g.get("name", ""))
        if not m:
            continue
        n_v, n_r, n_c, serial = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        key = (n_v, n_r, n_c)
        max_serial[key] = max(max_serial.get(key, 0), serial)
    return {k: v + 1 for k, v in max_serial.items()}


def assign_name(graph: LaceGraph, serial_state: Dict[Tuple[int, int, int], int]) -> str:
    """Generate a V<n_v>_<r>x<c>_<serial> name and bump the bucket counter."""
    n_v = len(graph.vertices)
    key = (n_v, graph.n_rows, graph.n_cols)
    serial = serial_state.get(key, 1)
    serial_state[key] = serial + 1
    return f"V{n_v}_{graph.n_rows}x{graph.n_cols}_{serial:03d}"


# -----------------------------------------------------------------------
# Build per-survivor record (with source=taylor overrides)
# -----------------------------------------------------------------------

def build_taylor_record(graph: LaceGraph, name: str, family: str,
                          thumbnail_dir: str = "thumbnails") -> dict:
    """Wrap build_ground_record and override source-dependent blocks
    so they reflect source='taylor' instead of the hardcoded 'irvine'."""
    # Set graph.name and graph.family so build_ground_record doesn't get
    # confused by the placeholder values from sweep graph.json
    graph.name = name
    graph.family = family

    record = build_ground_record(graph, name=name, family=family,
                                   thumbnail_dir=thumbnail_dir)
    # Override the source-tagged blocks. manufacturability_block reads
    # the record's geometry to derive its truth values, so we want to
    # call it again with source='taylor' rather than mutate the prior
    # output.
    record["manufacturability"] = manufacturability_block(record, source="taylor")
    record["provenance"] = provenance_block(
        source="taylor",
        # No irvine_label for Taylor grounds. The provenance_block
        # signature accepts whatever extra kwargs it accepts; if it
        # rejects irvine_label=None, change to omit instead.
    )
    # Add the new fields the migration introduced
    record["source"] = "taylor"
    record["traditional_name"] = "taylor_bobbin"
    return record


# -----------------------------------------------------------------------
# Summary rebuild (matches the integration script from May 4 morning)
# -----------------------------------------------------------------------

def rebuild_summary(atlas: dict) -> List[dict]:
    spring_default_idx = atlas["metadata"].get("spring_default_idx", 2)
    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)

    def safe_index(arr, i):
        if not arr:
            return None
        try:
            return arr[i]
        except (IndexError, TypeError):
            return None

    new_summary = []
    for idx, g in enumerate(atlas["grounds"]):
        spring = g.get("spring", {}) or {}
        beam = g.get("beam", {}) or {}
        new_summary.append({
            "idx": idx,
            "name": g["name"],
            "family": g["family"],
            "source": g.get("source", "irvine"),
            "traditional_name": g.get("traditional_name"),
            "n_rows": g["n_rows"],
            "n_cols": g["n_cols"],
            "n_vertices": g["n_vertices"],
            "spring_default_nu_min": safe_index(spring.get("nu_min"), spring_default_idx),
            "spring_default_nu_max": safe_index(spring.get("nu_max"), spring_default_idx),
            "spring_default_classification":
                safe_index(spring.get("classification"), spring_default_idx),
            "beam_default_nu_min": safe_index(beam.get("nu_min"), beam_default_idx),
            "beam_default_nu_max": safe_index(beam.get("nu_max"), beam_default_idx),
            "beam_default_classification":
                safe_index(beam.get("classification"), beam_default_idx),
        })
    return new_summary


def atomic_write_json(path: Path, data: dict) -> None:
    parent = path.parent
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
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


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    help="Sweep root, e.g. ~/Dropbox/Research/Aux-Mat/"
                         "bobbin_sweep_2026_05_04_evening/")
    ap.add_argument("--apply", action="store_true",
                    help="actually update atlas.json and copy thumbnails. "
                         "Without this, prints the dedup report only.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N survivors (smoke test).")
    ap.add_argument("--atlas", default=str(ATLAS_PATH))
    ap.add_argument("--thumbnails-dir", default=str(THUMBS_DIR))
    args = ap.parse_args()

    sweep_root = Path(os.path.expanduser(args.sweep)).resolve()
    atlas_path = Path(args.atlas).resolve()
    thumbs_dir = Path(args.thumbnails_dir).resolve()

    if not atlas_path.exists():
        print(f"ERROR: atlas not found at {atlas_path}", file=sys.stderr)
        sys.exit(2)
    if not sweep_root.exists():
        print(f"ERROR: sweep root not found at {sweep_root}", file=sys.stderr)
        sys.exit(2)

    print(f"Atlas: {atlas_path}")
    print(f"Sweep: {sweep_root}")
    print()

    # ---- 1. Load sweep ----
    successes = walk_sweep(sweep_root)
    print(f"Sweep successes: {len(successes)}")

    # ---- 2. Compute canonical for each ----
    print(f"Computing canonicals for sweep results...")
    sweep_records = []   # list of dicts: seed_dir, graph (LaceGraph), canon
    canon_errors = []
    for seed_dir, graph_dict in successes:
        try:
            graph = reconstruct_lacegraph_from_sweep(graph_dict)
            canon = graph_canonical(lacegraph_to_canonical_dict(graph))
            sweep_records.append({
                "seed_dir": seed_dir, "graph": graph, "canon": canon,
            })
        except Exception as e:
            canon_errors.append((seed_dir, f"{type(e).__name__}: {e}"))
    if canon_errors:
        print(f"  WARN: {len(canon_errors)} sweep results errored on "
              f"canonical computation:")
        for sd, msg in canon_errors[:5]:
            print(f"    {sd.name}: {msg}")

    # ---- 3. Load atlas, build canonical lookup ----
    with atlas_path.open() as f:
        atlas = json.load(f)
    print(f"Atlas grounds before: {len(atlas['grounds'])}")

    print(f"Computing canonicals for existing atlas...")
    canon_to_ground, n_stored, n_recomputed = build_existing_canonicals(atlas)
    print(f"  {len(canon_to_ground)} unique canonicals "
          f"(from {n_stored} stored + {n_recomputed} recomputed = "
          f"{n_stored + n_recomputed} total grounds processed)")

    # ---- 4. Multi-stage dedup ----
    dup_irvine = []        # (seed_dir, name_in_atlas)
    dup_taylor = []        # (seed_dir, name_in_atlas)
    dup_within = []        # (seed_dir, first_seed_dir)  same canon, second seen
    survivors = []         # records that pass all dedup

    seen_in_batch: Dict[str, Path] = {}   # canon -> first seed_dir

    for rec in sweep_records:
        canon = rec["canon"]
        sd = rec["seed_dir"]

        existing = canon_to_ground.get(canon)
        if existing is not None:
            existing_source = existing.get("source", "irvine")
            if existing_source == "irvine":
                dup_irvine.append((sd, existing.get("name", "?")))
            else:
                dup_taylor.append((sd, existing.get("name", "?")))
            continue

        if canon in seen_in_batch:
            dup_within.append((sd, seen_in_batch[canon]))
            continue
        seen_in_batch[canon] = sd
        survivors.append(rec)

    print()
    print("=== Dedup report ===")
    print(f"  Sweep successes:                    {len(sweep_records)}")
    print(f"  Duplicates of Irvine grounds:       {len(dup_irvine)}")
    print(f"  Duplicates of existing Taylor:      {len(dup_taylor)}")
    print(f"  Duplicates within sweep:            {len(dup_within)}")
    print(f"  Genuinely new survivors:            {len(survivors)}")
    overlap_pct = (100 * len(dup_taylor) /
                    max(1, len(sweep_records) - len(dup_irvine) - len(dup_within)))
    print(f"  Overlap with original 39 Taylor:    {overlap_pct:.1f}% "
          f"of non-Irvine, non-self-dup successes")

    if dup_irvine[:3]:
        print(f"\n  Irvine duplicate examples:")
        for sd, nm in dup_irvine[:3]:
            print(f"    {sd.name} -> already exists as {nm!r}")
    if dup_taylor[:3]:
        print(f"\n  Existing-Taylor duplicate examples:")
        for sd, nm in dup_taylor[:3]:
            print(f"    {sd.name} -> already exists as {nm!r}")

    if not survivors:
        print("\nNo new grounds. Exiting.")
        return

    if args.limit is not None and args.limit < len(survivors):
        print(f"\nLimiting to first {args.limit} survivors (smoke test).")
        survivors = survivors[:args.limit]

    # ---- 5. Family + naming for survivors ----
    serial_state = discover_serial_buckets(atlas)
    print(f"\nNext-available serials by bucket:")
    for k, v in sorted(serial_state.items()):
        print(f"  V{k[0]}_{k[1]}x{k[2]}: starts at {v}")

    print(f"\n=== Family + naming for {len(survivors)} survivors ===")
    plan = []   # list of (seed_dir, graph, family, name)
    family_dist = defaultdict(int)
    for rec in survivors:
        graph = rec["graph"]
        family = family_label(graph)
        family_dist[family] += 1
        # Need a fresh copy of serial_state so dry-run doesn't bump it
        # for real. Easier: do the bump now since serial_state is local.
        name = assign_name(graph, serial_state)
        plan.append((rec["seed_dir"], graph, family, name))
    for fam, n in sorted(family_dist.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {fam:14s} +{n}")

    if not args.apply:
        print(f"\n(dry run — atlas not modified, no thumbnails copied)")
        print(f"\nFirst 5 planned additions:")
        for sd, g, fam, nm in plan[:5]:
            print(f"  {sd.name}  ->  family={fam!r}  name={nm!r}")
        print(f"\nRe-run with --apply to commit.")
        return

    # ---- 6. Apply ----
    print(f"\n=== Computing physics for {len(plan)} survivors ===")
    print(f"This is the slow step. Expect 30-90s per ground.\n")
    new_records = []
    physics_errors = []
    t0 = time.time()
    for i, (sd, graph, family, name) in enumerate(plan):
        elapsed = time.time() - t0
        eta = (elapsed / max(i, 1)) * (len(plan) - i) if i > 0 else 0
        print(f"  [{i+1}/{len(plan)}] {family}/{name} "
              f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
        try:
            record = build_taylor_record(
                graph, name=name, family=family,
                thumbnail_dir="thumbnails")
            new_records.append((sd, record))
        except Exception as e:
            physics_errors.append((sd, name, f"{type(e).__name__}: {e}"))
            print(f"     FAIL: {type(e).__name__}: {e}")

    if physics_errors:
        print(f"\n  Physics errors ({len(physics_errors)}):")
        for sd, nm, msg in physics_errors[:5]:
            print(f"    {sd.name} ({nm}): {msg}")

    print(f"\nSuccessfully built {len(new_records)} new records.")

    # ---- 7. Append to atlas ----
    for _sd, record in new_records:
        atlas["grounds"].append(record)
    atlas["summary"] = rebuild_summary(atlas)
    atlas["metadata"]["n_grounds"] = len(atlas["grounds"])

    print(f"\nAtlas grounds after: {len(atlas['grounds'])}")
    print(f"Writing {atlas_path} (atomic)...")
    atomic_write_json(atlas_path, atlas)

    # ---- 8. Copy thread thumbnails ----
    print(f"\nCopying lace.png -> thread.png for new grounds...")
    copied = 0
    copy_errors = []
    for sd, record in new_records:
        family = record["family"]
        name = record["name"]
        src = sd / "lace.png"
        dst_dir = thumbs_dir / family / name
        dst = dst_dir / "thread.png"
        if not src.exists():
            copy_errors.append((sd, "lace.png missing in sweep dir"))
            continue
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            copy_errors.append((sd, f"{type(e).__name__}: {e}"))
    print(f"  Copied: {copied} / {len(new_records)}")
    if copy_errors:
        print(f"  Errors:")
        for sd, msg in copy_errors[:5]:
            print(f"    {sd.name}: {msg}")

    # ---- Done ----
    print(f"\nDone.")
    print(f"\nNext steps:")
    print(f"  1. cd docs && python3 -m http.server 8000  # verify visualizer")
    print(f"  2. git status                                # review changes")
    print(f"  3. git add docs/atlas.json docs/thumbnails/")
    print(f"  4. git commit -m 'Add N new Taylor grounds from "
          f"{sweep_root.name}'")


if __name__ == "__main__":
    main()
