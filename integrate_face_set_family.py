"""
integrate_face_set_family.py

Overwrite the `family` field of every ground in docs/atlas.json with
the face-set label computed from the graph's planar embedding. Also
add three new metadata fields per ground:

    family            (overwritten)  face-set label, e.g. "3_6"
    traditional_name  (new)          old non-numeric family if any,
                                     else null. e.g. "cloth", "taylor_bobbin"
    source            (new)          "irvine" or "taylor". Drives the
                                     visualizer's marker shape choice.

Then rebuild summary[] from grounds[] (now including `source` in each
summary entry so the top-N table can render a provenance column
without looking up grounds[]) and update metadata.n_grounds.

Writes atomically (temp file + rename) so you never have a half-
written atlas.json on disk.

Usage:
    cd /Users/richardtaylor/Dropbox/Research/Aux-Mat/Aux_mat_repo/
    python3 integrate_face_set_family.py            # dry run, prints diff
    python3 integrate_face_set_family.py --apply    # actually writes

Always run the dry run first. The --apply flag is required to commit.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Repo-relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from auxetic_lace.parse_to_graph import LaceGraph, Edge   # noqa: E402
from auxetic_lace.compute_family import (                  # noqa: E402
    family_label,
    assign_traditional_name,
)


ATLAS_PATH = Path("docs/atlas.json")


def reconstruct_graph(g_record: dict) -> LaceGraph:
    """Rebuild a LaceGraph from a per-ground record."""
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


def derive_source(old_family: str) -> str:
    """Provenance label: 'taylor' for taylor_bobbin grounds, 'irvine' otherwise."""
    return "taylor" if old_family == "taylor_bobbin" else "irvine"


def rebuild_summary(atlas: dict) -> list:
    """Rebuild summary[] from grounds[] using the schema documented in
    docs/AUX_MAT_PROJECT_STATUS.md, plus the new `source` field for
    provenance-aware UI rendering."""
    spring_default_idx = atlas["metadata"].get("spring_default_idx", 2)
    beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)

    new_summary = []
    for idx, g in enumerate(atlas["grounds"]):
        spring = g.get("spring", {}) or {}
        beam = g.get("beam", {}) or {}

        def safe_index(arr, i):
            if not arr:
                return None
            try:
                return arr[i]
            except (IndexError, TypeError):
                return None

        spring_nu_min = safe_index(spring.get("nu_min"), spring_default_idx)
        spring_nu_max = safe_index(spring.get("nu_max"), spring_default_idx)
        spring_class = safe_index(spring.get("classification"), spring_default_idx)
        beam_nu_min = safe_index(beam.get("nu_min"), beam_default_idx)
        beam_nu_max = safe_index(beam.get("nu_max"), beam_default_idx)
        beam_class = safe_index(beam.get("classification"), beam_default_idx)

        new_summary.append({
            "idx": idx,
            "name": g["name"],
            "family": g["family"],
            "source": g.get("source", "irvine"),
            "traditional_name": g.get("traditional_name"),
            "n_rows": g["n_rows"],
            "n_cols": g["n_cols"],
            "n_vertices": g["n_vertices"],
            "spring_default_nu_min": spring_nu_min,
            "spring_default_nu_max": spring_nu_max,
            "spring_default_classification": spring_class,
            "beam_default_nu_min": beam_nu_min,
            "beam_default_nu_max": beam_nu_max,
            "beam_default_classification": beam_class,
        })
    return new_summary


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: write to temp file in same dir, fsync, rename."""
    path = Path(path)
    parent = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually write atlas.json. Without this, dry-run.")
    ap.add_argument("--atlas", default=str(ATLAS_PATH),
                    help=f"path to atlas.json (default: {ATLAS_PATH})")
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    if not atlas_path.exists():
        print(f"ERROR: {atlas_path} not found.", file=sys.stderr)
        sys.exit(2)

    with atlas_path.open() as f:
        atlas = json.load(f)

    grounds = atlas["grounds"]
    n_total = len(grounds)
    print(f"Loaded atlas: {n_total} grounds.")

    # ---------------- compute new fields ----------------
    family_changes = Counter()
    traditional_name_assignments = Counter()
    source_counts = Counter()
    family_dist_old = Counter()
    family_dist_new = Counter()

    errors = []

    for idx, g in enumerate(grounds):
        old_family = g.get("family", "")
        family_dist_old[old_family] += 1

        try:
            graph = reconstruct_graph(g)
            new_family = family_label(graph)
        except Exception as e:
            errors.append((idx, g.get("name", "?"), f"{type(e).__name__}: {e}"))
            continue

        traditional = assign_traditional_name(g.get("name", ""), old_family)
        source = derive_source(old_family)

        g["family"] = new_family
        g["traditional_name"] = traditional
        g["source"] = source

        family_changes[(old_family, new_family)] += 1
        if traditional is not None:
            traditional_name_assignments[traditional] += 1
        source_counts[source] += 1
        family_dist_new[new_family] += 1

    if errors:
        print(f"\nERROR: {len(errors)} grounds failed reconstruction:", file=sys.stderr)
        for idx, name, msg in errors[:10]:
            print(f"  [{idx}] {name}: {msg}", file=sys.stderr)
        print("Aborting.", file=sys.stderr)
        sys.exit(1)

    # ---------------- rebuild summary + metadata ----------------
    new_summary = rebuild_summary(atlas)
    atlas["summary"] = new_summary
    atlas["metadata"]["n_grounds"] = len(atlas["grounds"])

    # ---------------- report ----------------
    print(f"\n=== Family changes ===")
    changed = sum(c for (o, n), c in family_changes.items() if o != n)
    unchanged = sum(c for (o, n), c in family_changes.items() if o == n)
    print(f"  Unchanged: {unchanged}")
    print(f"  Changed:   {changed}")
    for (old, new), c in sorted(family_changes.items(),
                                key=lambda x: (-x[1], x[0])):
        if old != new:
            print(f"    {old:20s} -> {new:20s}  {c:4d}")

    print(f"\n=== Source distribution ===")
    for src, c in sorted(source_counts.items()):
        print(f"  {src:10s} {c}")

    print(f"\n=== traditional_name assignments ===")
    if traditional_name_assignments:
        for nm, c in sorted(traditional_name_assignments.items(),
                            key=lambda x: (-x[1], x[0])):
            print(f"  {nm:20s} {c}")
    else:
        print("  (none)")

    print(f"\n=== Family distribution (after integration) ===")
    for fam, c in sorted(family_dist_new.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {fam:20s} {c}")

    print(f"\n=== Sanity checks ===")
    print(f"  metadata.n_grounds: {atlas['metadata']['n_grounds']}")
    print(f"  len(grounds):       {len(atlas['grounds'])}")
    print(f"  len(summary):       {len(atlas['summary'])}")
    assert (atlas["metadata"]["n_grounds"]
            == len(atlas["grounds"])
            == len(atlas["summary"])), "count mismatch"
    print("  counts agree: OK")

    sample_idxs = [0, len(grounds) // 2, len(grounds) - 1]
    print(f"\n=== Spot-check sample records ===")
    for idx in sample_idxs:
        g = atlas["grounds"][idx]
        s = atlas["summary"][idx]
        print(f"  [{idx}] grounds: name={g['name']!r}  family={g['family']!r}  "
              f"traditional_name={g.get('traditional_name')!r}  source={g.get('source')!r}")
        print(f"        summary: name={s['name']!r}  family={s['family']!r}  source={s.get('source')!r}")
        assert g["name"] == s["name"], f"name mismatch at idx {idx}"
        assert g["family"] == s["family"], f"family mismatch at idx {idx}"
        assert g.get("source") == s.get("source"), f"source mismatch at idx {idx}"

    if args.apply:
        print(f"\nWriting {atlas_path} (atomic)...")
        atomic_write_json(atlas_path, atlas)
        print("Done.")
        print("\nNext steps:")
        print(f"  1. cd docs/ && python3 -m http.server 8000   # verify visualizer")
        print(f"  2. git add {atlas_path} src/auxetic_lace/compute_family.py")
        print(f"  3. git commit -m 'face-set family + traditional_name + source fields'")
    else:
        print(f"\n(dry run — atlas.json NOT modified)")
        print(f"Re-run with --apply to commit changes.")


if __name__ == "__main__":
    main()
