"""
verify_family_against_atlas.py

Compute the face-set family for every ground in docs/atlas.json and
report disagreements with the stored `family` field. Use BEFORE
overwriting any data — this sanity-checks that our face-tracing
produces labels matching Irvine's existing catalog labels.

Usage:
    cd /Users/richardtaylor/Dropbox/Research/Aux-Mat/Aux_mat_repo/
    python3 verify_family_against_atlas.py

Reports:
- Counts: total grounds, agreement, disagreement, errors
- Per-source breakdown (irvine vs taylor_bobbin)
- Disagreement transition matrix (old family -> computed family)
- Sample of individual disagreements
- Full transition matrix for all (old, new) combinations

Exits 0 if all grounds match (modulo expected traditional-name
overrides like cloth -> face-set code), non-zero otherwise.

NOTE: this script does NOT modify atlas.json. It only reports.

Atlas edge schema (from observation):
    {"src": int, "dst": int, "wrap": [int, int], "polyline": [[x,y],[x,y]]}

The repo's Edge dataclass uses src_idx/dst_idx; the atlas serialization
uses the shorter src/dst. We translate during reconstruction.
"""

import json
import sys
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
    """
    Build a LaceGraph from a per-ground record in atlas.json.

    Schema (from terminal probe):
        record top-level keys include: name, family, n_rows, n_cols,
            n_vertices, n_edges, vertices, edges, ...
        each edge: {src, dst, wrap, polyline}
            wrap is [wrap_col, wrap_row]
            polyline is a list of [x, y] points (typically 2)
    """
    name = g_record["name"]
    family = g_record.get("family", "")
    n_rows = g_record["n_rows"]
    n_cols = g_record["n_cols"]
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
    # LaceGraph in the repo also requires `keyword`. Atlas may not store it;
    # default to empty string. (compute_family doesn't use keyword.)
    return LaceGraph(
        name=name,
        family=family,
        keyword=g_record.get("keyword", ""),
        n_rows=n_rows,
        n_cols=n_cols,
        vertices=vertices,
        edges=edges,
    )


def main():
    if not ATLAS_PATH.exists():
        print(f"ERROR: {ATLAS_PATH} not found. Run from repo root.")
        sys.exit(2)

    with ATLAS_PATH.open() as f:
        atlas = json.load(f)

    grounds = atlas["grounds"]
    n_total = len(grounds)
    print(f"Loaded atlas: {n_total} grounds.")
    print(f"Computing face-set family for each...\n")

    agree = 0
    agree_via_traditional_override = 0
    disagree = []     # (idx, name, old_family, new_family)
    errored = []      # (idx, name, old_family, error_msg)
    by_source = defaultdict(lambda: {"agree": 0, "disagree": 0, "error": 0})
    transition_matrix = Counter()  # (old, new) -> count

    for idx, g in enumerate(grounds):
        name = g.get("name", f"<idx {idx}>")
        old_family = g.get("family", "")
        source = "taylor" if old_family == "taylor_bobbin" else "irvine"

        try:
            graph = reconstruct_graph(g)
            new_family = family_label(graph)
        except KeyError as e:
            errored.append((idx, name, old_family,
                            f"missing field {e} in record"))
            by_source[source]["error"] += 1
            continue
        except Exception as e:
            errored.append((idx, name, old_family,
                            f"{type(e).__name__}: {e}"))
            by_source[source]["error"] += 1
            continue

        transition_matrix[(old_family, new_family)] += 1

        # Agreement logic:
        # 1. Direct match: old == new -> agree
        # 2. Old is a traditional name (cloth, etc.) AND it's an irvine
        #    ground -> expected override, count as agree
        # 3. Otherwise -> disagree
        if old_family == new_family:
            agree += 1
            by_source[source]["agree"] += 1
        else:
            traditional = assign_traditional_name(name, old_family)
            is_traditional_irvine = (traditional is not None and source == "irvine"
                                     and traditional != "taylor_bobbin")
            if is_traditional_irvine:
                agree += 1
                agree_via_traditional_override += 1
                by_source[source]["agree"] += 1
            else:
                disagree.append((idx, name, old_family, new_family))
                by_source[source]["disagree"] += 1

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print(f"=== Summary ===")
    print(f"  Total grounds:                    {n_total}")
    print(f"  Agree:                            {agree}")
    print(f"    (of which traditional override:  {agree_via_traditional_override})")
    print(f"  Disagree:                         {len(disagree)}")
    print(f"  Errored:                          {len(errored)}")

    print(f"\n=== By source ===")
    for src, counts in by_source.items():
        print(f"  {src:15s} agree={counts['agree']:4d}  "
              f"disagree={counts['disagree']:4d}  error={counts['error']:4d}")

    if errored:
        print(f"\n=== Errors (first 10) ===")
        for idx, name, old_family, msg in errored[:10]:
            print(f"  [{idx}] {name} (family={old_family}): {msg}")

    if disagree:
        print(f"\n=== Disagreement transition matrix (old -> new : count) ===")
        dis_matrix = Counter((d[2], d[3]) for d in disagree)
        for (old, new), count in sorted(dis_matrix.items(), key=lambda x: -x[1]):
            print(f"  {old:20s} -> {new:20s}  {count:4d}")

        print(f"\n=== Disagreement examples (up to 15) ===")
        for idx, name, old_family, new_family in disagree[:15]:
            print(f"  [{idx}] name={name!r}  old={old_family!r}  new={new_family!r}")

    print(f"\n=== Full transition matrix (all entries) ===")
    for (old, new), count in sorted(transition_matrix.items(),
                                    key=lambda x: (-x[1], x[0])):
        marker = "  " if old == new else "* "
        print(f"  {marker}{old:20s} -> {new:20s}  {count:4d}")

    # exit code
    if errored:
        print("\nFAIL: some grounds errored during reconstruction.")
        sys.exit(1)
    if disagree:
        print(f"\nWARN: {len(disagree)} grounds disagree (after traditional-name allowance).")
        print("Inspect the transition matrix and examples above before integrating.")
        sys.exit(1)
    print("\nOK: all grounds match (within traditional-name allowance).")
    sys.exit(0)


if __name__ == "__main__":
    main()
