"""
manufacturability.py
=====================

Property checkers and provenance metadata for lattice grounds.

Each ground gets two new top-level blocks in the atlas record:

  ground["manufacturability"] = {
      "is_2in2out": bool,            # 2-in 2-out directed structure
      "is_planar": None or bool,     # Embedded planar (no rod crossings) -- TODO
      "is_printable": None or bool,  # 3D-printable (planar + connected + ...) -- TODO
      "is_lace_workable": bool,      # Valid bobbin-lace thread circuits
  }

  ground["provenance"] = {
      "source": str,
      "irvine_label": str | None,
      "enumerator_run": str | None,
  }

For Irvine grounds, all properties are TRUE by construction. We verify
is_2in2out and is_lace_workable directly as a sanity check; the others
default to True for Irvine grounds pending real checker implementations.

For enumerator output, is_2in2out and is_lace_workable are computed
directly. is_planar and is_printable default to None.
"""

from __future__ import annotations
from typing import Tuple

from .lace_workability import is_lace_workable


def is_2in2out(ground: dict) -> bool:
    """Check whether the ground has 2-in 2-out directed structure."""
    n_vertices = len(ground["vertices"])
    in_count = [0] * n_vertices
    out_count = [0] * n_vertices
    for e in ground["edges"]:
        u, v = e["src"], e["dst"]
        out_count[u] += 1
        in_count[v] += 1
    return all(in_count[v] == 2 and out_count[v] == 2 for v in range(n_vertices))


def manufacturability_block(ground: dict, source: str = "irvine",
                              verify: bool = True) -> dict:
    """Compute the manufacturability block for a ground."""
    if source == "irvine":
        if verify:
            mfg_2in2out = is_2in2out(ground)
            mfg_lace = is_lace_workable(ground)
        else:
            mfg_2in2out = True
            mfg_lace = True
        return {
            "is_2in2out": mfg_2in2out,
            "is_planar": True,
            "is_printable": True,
            "is_lace_workable": mfg_lace,
        }
    else:
        return {
            "is_2in2out": is_2in2out(ground),
            "is_planar": None,
            "is_printable": None,
            "is_lace_workable": is_lace_workable(ground),
        }


def provenance_block(source: str = "irvine",
                      irvine_label: str = None,
                      enumerator_run: str = None) -> dict:
    return {
        "source": source,
        "irvine_label": irvine_label,
        "enumerator_run": enumerator_run,
    }


if __name__ == "__main__":
    import json
    import sys

    atlas_path = sys.argv[1] if len(sys.argv) > 1 else "docs/atlas.json"
    with open(atlas_path) as f:
        atlas = json.load(f)
    grounds = atlas["grounds"]

    print(f"Validating manufacturability properties on {len(grounds)} "
          f"Irvine grounds...")
    print()

    fail_2in2out = []
    fail_lace = []
    for g in grounds:
        label = f"{g['family']}/{g['name']}"
        mfg = manufacturability_block(g, source="irvine", verify=True)
        if not mfg["is_2in2out"]:
            fail_2in2out.append(label)
        if not mfg["is_lace_workable"]:
            fail_lace.append(label)

    print(f"  is_2in2out:        {len(grounds) - len(fail_2in2out)} / {len(grounds)} pass")
    print(f"  is_lace_workable:  {len(grounds) - len(fail_lace)} / {len(grounds)} pass")
    print(f"  is_planar:         all True by provenance (real check pending)")
    print(f"  is_printable:      all True by provenance (real check pending)")

    if fail_2in2out:
        print(f"\n  is_2in2out failures (first 10):")
        for f in fail_2in2out[:10]:
            print(f"    {f}")
    if fail_lace:
        print(f"\n  is_lace_workable failures (first 10):")
        for f in fail_lace[:10]:
            print(f"    {f}")

    if not fail_2in2out and not fail_lace:
        print(f"\n  All Irvine grounds pass all real checks.")
