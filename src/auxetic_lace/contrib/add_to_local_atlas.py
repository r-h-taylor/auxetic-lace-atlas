"""
add_to_local_atlas.py
=====================

Merge a computed submission record into a local atlas.json file.

This is the "local mode" of submission contribution. After running:
    python3 validate_submission.py my.json
    python3 compute_submission.py my.json --output my_record.json
    python3 rank_submission.py my_record.json --atlas atlas.json
the user can then run:
    python3 add_to_local_atlas.py my_record.json --atlas atlas.json
to merge their submission into their local atlas. The visualizer (which
loads atlas.json) will then show their submission alongside catalog
entries.

By default this script REFUSES TO ADD if duplicate detection finds an
isomorphic existing entry. Use --allow-duplicates to override (e.g.,
when the user explicitly wants to record a re-derivation under a new
name).

USAGE:
    python3 add_to_local_atlas.py record.json --atlas atlas.json
    python3 add_to_local_atlas.py record.json --atlas atlas.json --output atlas_new.json
        # write to a different output instead of in-place
    python3 add_to_local_atlas.py record.json --atlas atlas.json --allow-duplicates
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict

from .rank_submission import duplicate_check


def main():
    ap = argparse.ArgumentParser(
        description="Merge a computed submission record into a local atlas.")
    ap.add_argument("record", help="Computed record JSON path")
    ap.add_argument("--atlas", required=True,
                    help="Local atlas.json path (will be modified in place "
                         "unless --output is given)")
    ap.add_argument("--output", default=None,
                    help="Write to this path instead of modifying --atlas")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="Skip the isomorphism duplicate check")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip making a .bak copy of the atlas before "
                         "modifying")
    args = ap.parse_args()

    # Load
    with open(args.record) as f:
        record = json.load(f)
    with open(args.atlas) as f:
        atlas = json.load(f)

    # Sanity: required fields on the record
    for f in ("name", "family", "n_rows", "n_cols", "vertices", "edges"):
        if f not in record:
            print(f"FAIL: record missing required field '{f}'", file=sys.stderr)
            sys.exit(1)

    # Duplicate check
    if not args.allow_duplicates:
        dup = duplicate_check(record, atlas)
        if "error" not in dup and not dup.get("is_unique", True):
            print(f"FAIL: submission is isomorphic to "
                   f"{dup['n_matches']} existing entries:",
                   file=sys.stderr)
            for m in dup["matches"]:
                print(f"  - {m['family']}/{m['name']}", file=sys.stderr)
            print("Use --allow-duplicates to add anyway, or remove the "
                   "submission.", file=sys.stderr)
            sys.exit(1)

    # Name collision check (different topology but same family/name)
    fam_name = (record["family"], record["name"])
    for g in atlas.get("grounds", []):
        if (g.get("family"), g.get("name")) == fam_name:
            print(f"FAIL: family/name collision with existing "
                   f"entry {fam_name[0]}/{fam_name[1]}. "
                   "Choose a different name.", file=sys.stderr)
            sys.exit(1)

    # Append
    atlas.setdefault("grounds", []).append(record)
    atlas.setdefault("metadata", {})
    atlas["metadata"]["n_grounds"] = len(atlas["grounds"])

    # Update summary if present
    if "summary" in atlas:
        spring_default_idx = atlas["metadata"].get("spring_default_idx", 2)
        beam_default_idx = atlas["metadata"].get("beam_default_idx", 1)

        def safe_get(arr, idx):
            return arr[idx] if (arr is not None and idx < len(arr)) else None

        new_summary_entry = {
            "idx": len(atlas["grounds"]) - 1,
            "name": record["name"],
            "family": record["family"],
            "n_rows": record["n_rows"],
            "n_cols": record["n_cols"],
            "n_vertices": record["n_vertices"],
            "spring_default_nu_min": safe_get(
                record.get("spring", {}).get("nu_min"), spring_default_idx),
            "spring_default_nu_max": safe_get(
                record.get("spring", {}).get("nu_max"), spring_default_idx),
            "spring_default_classification": safe_get(
                record.get("spring", {}).get("classification"),
                spring_default_idx),
            "beam_default_nu_min": safe_get(
                record.get("beam", {}).get("nu_min"), beam_default_idx),
            "beam_default_nu_max": safe_get(
                record.get("beam", {}).get("nu_max"), beam_default_idx),
            "beam_default_classification": safe_get(
                record.get("beam", {}).get("classification"),
                beam_default_idx),
        }
        atlas["summary"].append(new_summary_entry)

    # Write
    target = args.output or args.atlas
    if not args.no_backup and target == args.atlas and os.path.exists(target):
        backup = target + ".bak"
        shutil.copyfile(target, backup)
        print(f"  Backed up {target} -> {backup}")

    # Write atomically: tmpfile + rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(target),
        dir=os.path.dirname(os.path.abspath(target)) or ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(atlas, f, separators=(",", ":"))
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Added {record['family']}/{record['name']} to {target}")
    print(f"  Atlas now contains {len(atlas['grounds'])} grounds.")


if __name__ == "__main__":
    main()
