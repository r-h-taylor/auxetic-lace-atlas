"""
fix_source_field.py

Re-derives the `source` field on every ground in atlas.json based on
the name pattern, since `derive_source()` in the integration script
keyed off the (now-overwritten) family field. Taylor-bobbin grounds
follow the naming convention `V<n_vertices>_<rows>x<cols>_<serial>`,
e.g. `V9_3x3_007`. Everything else is irvine.

Also fixes the related issue that summary[] inherited the bad source
values, by rebuilding summary from grounds.

This is idempotent — safe to run multiple times.

Usage:
    python3 fix_source_field.py            # dry run
    python3 fix_source_field.py --apply    # write
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ATLAS_PATH = Path("docs/atlas.json")

# Taylor names: V<digits>_<digits>x<digits>_<digits>
TAYLOR_NAME_RE = re.compile(r"^V\d+_\d+x\d+_\d+$")


def derive_source_from_name(name: str) -> str:
    return "taylor" if TAYLOR_NAME_RE.match(name or "") else "irvine"


def atomic_write_json(path: Path, data: dict) -> None:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--atlas", default=str(ATLAS_PATH))
    args = ap.parse_args()

    atlas_path = Path(args.atlas)
    with atlas_path.open() as f:
        atlas = json.load(f)

    grounds = atlas["grounds"]
    summary = atlas.get("summary", [])

    before = Counter(g.get("source") for g in grounds)
    print(f"Before: {dict(before)}")

    changed_grounds = 0
    for g in grounds:
        new_source = derive_source_from_name(g["name"])
        if g.get("source") != new_source:
            g["source"] = new_source
            changed_grounds += 1

    # Also fix traditional_name: taylor grounds should have it set to
    # "taylor_bobbin" (provenance), since the heuristic relied on the
    # original family value which was overwritten on first integration.
    # Irvine grounds with traditional_name from cloth are already correct
    # (cloth -> traditional_name="cloth" was preserved on first pass).
    changed_traditional = 0
    for g in grounds:
        if g.get("source") == "taylor":
            if g.get("traditional_name") != "taylor_bobbin":
                g["traditional_name"] = "taylor_bobbin"
                changed_traditional += 1

    after = Counter(g.get("source") for g in grounds)
    print(f"After:  {dict(after)}")
    print(f"Grounds with corrected source:           {changed_grounds}")
    print(f"Grounds with corrected traditional_name: {changed_traditional}")

    # Rebuild summary entries that have source/traditional_name fields.
    # Keep all existing fields, only overwrite source and traditional_name.
    name_to_ground = {g["name"]: g for g in grounds}
    fixed_summary = 0
    for s in summary:
        g = name_to_ground.get(s["name"])
        if g is None:
            continue
        if s.get("source") != g["source"]:
            s["source"] = g["source"]
            fixed_summary += 1
        if s.get("traditional_name") != g.get("traditional_name"):
            s["traditional_name"] = g.get("traditional_name")
    print(f"Summary entries with corrected source:   {fixed_summary}")

    # Sample
    print("\n=== Sample V-named grounds ===")
    sampled = 0
    for g in grounds:
        if TAYLOR_NAME_RE.match(g["name"]):
            print(f"  {g['name']}: family={g['family']!r}  "
                  f"source={g['source']!r}  "
                  f"traditional_name={g.get('traditional_name')!r}")
            sampled += 1
            if sampled >= 5:
                break

    if args.apply:
        atomic_write_json(atlas_path, atlas)
        print(f"\nWrote {atlas_path}.")
    else:
        print("\n(dry run — atlas.json NOT modified)")


if __name__ == "__main__":
    main()
