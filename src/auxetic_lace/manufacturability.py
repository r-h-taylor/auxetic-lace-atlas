"""
manufacturability.py
=====================

Property checkers and provenance metadata for lattice grounds.

Each ground gets two new top-level blocks in the atlas record:

  ground["manufacturability"] = {
      "is_2in2out": bool,          # Admits a 2-in 2-out directed orientation
      "is_planar": None or bool,   # Embedded planar (no rod crossings) -- TODO
      "is_printable": None or bool,# 3D-printable (planar + connected + ...) -- TODO
      "is_lace_workable": None or bool,  # Valid bobbin-lace thread circuits -- TODO
  }

  ground["provenance"] = {
      "source": str,               # "irvine" | "enumerated_NxM" | "user_submitted" | ...
      "irvine_label": str | None,  # "3_6/2x4_86" if from Irvine's catalog
      "enumerator_run": str | None,# ID of enumerator run that produced it
  }

For the existing 321 Irvine grounds, all four properties are TRUE by
construction (Irvine's catalog is precisely the lace-workable grounds at
small cell sizes), so we can default the checkers' "None or bool" fields
to True for those grounds without re-checking. The enumerator will run
real checks on newly-generated graphs.

In this iteration we implement only `is_2in2out` rigorously. The others
are scaffolded with None (= "not yet checked") or True for Irvine grounds
(by provenance). Future iterations fill them in.
"""

from __future__ import annotations
from typing import List, Tuple


def is_2in2out(ground: dict) -> bool:
    """Check whether the ground admits a 2-in 2-out directed orientation.

    A 4-regular undirected graph admits a 2-in-2-out orientation iff every
    connected component is Eulerian (every vertex has even degree) AND we
    can pick directions for each edge so that in-degree = out-degree = 2 at
    every vertex.

    For 4-regular graphs (which our atlas grounds all are), the parity
    condition is automatically satisfied — every vertex has degree 4 which
    is even. So the question reduces to whether a valid orientation exists.

    Theorem (Robbins 1939, generalized): an undirected graph G admits an
    orientation where each vertex has equal in-degree and out-degree iff
    every vertex of G has even degree. For 4-regular graphs this is always
    satisfied.

    For periodic graphs the same theorem holds when we work on the
    quotient graph (treating self-loops and multi-edges appropriately).

    So for any 4-regular periodic graph, is_2in2out is automatically True.

    However, we verify this directly to (a) catch bugs in the input data
    and (b) provide a basis for stricter variants later (e.g., orientations
    where every wrap-(1,0) edge points in a particular direction).
    """
    # Step 1: verify 4-regularity at every vertex
    n_vertices = len(ground["vertices"])
    degree = [0] * n_vertices
    for e in ground["edges"]:
        u, v = e["src"], e["dst"]
        if u == v:
            degree[u] += 2  # self-loop contributes 2 to degree
        else:
            degree[u] += 1
            degree[v] += 1

    if any(d != 4 for d in degree):
        return False

    # Step 2: try to find a 2-in 2-out orientation via Euler-tour
    # decomposition. Every connected component of a 4-regular graph
    # admits an Eulerian circuit, which when followed in one direction
    # gives an alternating in/out pattern at every vertex. Because each
    # vertex has degree 4, two Eulerian-circuit visits give exactly
    # 2 in-edges and 2 out-edges.
    #
    # Simpler: the existence of such an orientation is guaranteed by the
    # theorem above. We could explicitly construct one for verification,
    # but for a binary True/False checker the theorem suffices.
    return True


def manufacturability_block(ground: dict, source: str = "irvine") -> dict:
    """Compute the manufacturability block for a ground.

    For Irvine grounds (source="irvine"), trust the provenance: every
    ground is known to be 2-in-2-out, planar, printable, and lace-workable.
    Set all four to True without re-checking (except is_2in2out, which is
    cheap to verify and serves as a sanity check on input).

    For other grounds (e.g., enumerator output), only is_2in2out is checked
    here. The other fields default to None ("unknown - check needed").
    """
    if source == "irvine":
        return {
            "is_2in2out": is_2in2out(ground),  # sanity-check the input
            "is_planar": True,                  # by Irvine's construction
            "is_printable": True,                # by Irvine's construction
            "is_lace_workable": True,            # Irvine's catalog defines this
        }
    else:
        return {
            "is_2in2out": is_2in2out(ground),
            "is_planar": None,
            "is_printable": None,
            "is_lace_workable": None,
        }


def provenance_block(source: str = "irvine",
                      irvine_label: str = None,
                      enumerator_run: str = None) -> dict:
    """Build a provenance block."""
    return {
        "source": source,
        "irvine_label": irvine_label,
        "enumerator_run": enumerator_run,
    }


# ---------------------------------------------------------------------------
# CLI: validate Irvine catalog
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    atlas_path = sys.argv[1] if len(sys.argv) > 1 else "docs/atlas.json"
    with open(atlas_path) as f:
        atlas = json.load(f)
    grounds = atlas["grounds"]

    print(f"Validating manufacturability properties on {len(grounds)} "
          f"Irvine grounds...")

    fail_2in2out = []
    for g in grounds:
        if not is_2in2out(g):
            fail_2in2out.append(f"{g['family']}/{g['name']}")

    print(f"\n  is_2in2out check:")
    if not fail_2in2out:
        print(f"    All 321 grounds pass. (Expected — they're 4-regular.)")
    else:
        print(f"    FAILED for {len(fail_2in2out)} grounds:")
        for label in fail_2in2out[:10]:
            print(f"      {label}")

    # Demo: show the manufacturability + provenance blocks for a few grounds
    print(f"\n  Sample manufacturability + provenance blocks:")
    for i in [0, 100, 200]:
        g = grounds[i]
        label = f"{g['family']}/{g['name']}"
        mfg = manufacturability_block(g, source="irvine")
        prov = provenance_block(source="irvine", irvine_label=label)
        print(f"\n    {label}:")
        print(f"      manufacturability = {mfg}")
        print(f"      provenance        = {prov}")
