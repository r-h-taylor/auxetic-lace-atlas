"""
submission_schema.py
====================

JSON schema for user-submitted graph topologies, the core data structure
for community contributions to the Auxetic Lace Atlas.

A submission is a periodic 2D directed graph on a torus, optionally
satisfying the tesselace properties. Each submission lives as a single
JSON file with the structure shown in SUBMISSION_SCHEMA below.

The submission lifecycle:
  1. User authors submission.json
  2. validate_submission.py checks structural invariants
  3. compute_submission.py runs mechanics, produces atlas-compatible record
  4. rank_submission.py compares to existing catalog
  5. (a) local: add_to_local_atlas.py merges into user's atlas.json copy
     (b) PR: open GitHub PR with the submission file; CI runs (1)-(4)
        automatically and posts the result as a comment.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple


SUBMISSION_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Auxetic Lace Atlas — graph submission",
    "type": "object",
    "required": ["name", "n_rows", "n_cols", "vertices", "edges"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Short identifier, kebab- or snake-case. "
                           "Must be unique within `family`.",
            "pattern": "^[A-Za-z0-9_\\-]+$",
            "examples": ["my_kagome_v2", "rotating_squares_4x4"],
        },
        "family": {
            "type": "string",
            "description": "Family name. Use 'tesselace/{hole_shapes}' for "
                           "tesselace-compliant submissions, "
                           "'user/{anything}' for general submissions.",
            "default": "user/contributed",
            "examples": ["tesselace/3_6", "user/rotating_squares",
                          "user/chiral_honeycomb"],
        },
        "tesselace_compliant": {
            "type": "boolean",
            "description": "Self-asserted: does this graph satisfy the 5 "
                           "tesselace properties (2-regular, periodic, "
                           "connected, partial-order, thread-conserving)? "
                           "validate_submission.py will check.",
            "default": False,
        },
        "n_rows": {
            "type": "integer", "minimum": 1,
            "description": "Number of rows in the period parallelogram",
        },
        "n_cols": {
            "type": "integer", "minimum": 1,
            "description": "Number of columns in the period parallelogram",
        },
        "vertices": {
            "type": "array",
            "description": "List of [col, row] integer coordinates within "
                           "[0, n_cols) x [0, n_rows). The position of "
                           "vertex i in Cartesian space is "
                           "lattice_vectors @ vertices[i].",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
            },
            "minItems": 1,
        },
        "edges": {
            "type": "array",
            "description": "List of directed edges. Each edge has a source "
                           "vertex (index into `vertices`), destination "
                           "vertex (index), and integer wrap = (wc, wr) "
                           "indicating which periodic image of the "
                           "destination the edge actually reaches.",
            "items": {
                "type": "object",
                "required": ["src", "dst", "wrap"],
                "properties": {
                    "src": {"type": "integer", "minimum": 0},
                    "dst": {"type": "integer", "minimum": 0},
                    "wrap": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2, "maxItems": 2,
                    },
                },
            },
            "minItems": 1,
        },
        "lattice_vectors": {
            "type": "array",
            "description": "2x2 matrix of lattice vectors [[a1x, a2x], "
                           "[a1y, a2y]]. Default identity (square lattice).",
            "default": [[1.0, 0.0], [0.0, 1.0]],
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2, "maxItems": 2,
            },
        },
        "provenance": {
            "type": "object",
            "description": "Author and origin metadata.",
            "properties": {
                "author": {"type": "string"},
                "author_orcid": {"type": "string"},
                "description": {"type": "string"},
                "derived_from": {
                    "type": "string",
                    "description": "Optional reference to existing entry "
                                   "if this is a variant",
                },
                "doi": {"type": "string"},
                "license": {"type": "string", "default": "CC-BY 4.0"},
            },
        },
    },
}


def validate_against_schema(submission: Dict[str, Any]) -> List[str]:
    """Lightweight schema validation. Returns list of error messages
    (empty if valid).

    Performs the structural checks from SUBMISSION_SCHEMA without
    requiring jsonschema as a dependency."""
    errors = []
    required = SUBMISSION_SCHEMA["required"]
    for f in required:
        if f not in submission:
            errors.append(f"missing required field: '{f}'")
    if errors:
        return errors

    # Type checks
    if not isinstance(submission["name"], str):
        errors.append("name must be a string")
    elif not all(c.isalnum() or c in "_-" for c in submission["name"]):
        errors.append(f"name '{submission['name']}' contains invalid characters "
                       "(only alphanumeric, underscore, hyphen)")

    n_rows = submission.get("n_rows")
    n_cols = submission.get("n_cols")
    if not isinstance(n_rows, int) or n_rows < 1:
        errors.append("n_rows must be a positive integer")
    if not isinstance(n_cols, int) or n_cols < 1:
        errors.append("n_cols must be a positive integer")

    verts = submission.get("vertices", [])
    if not isinstance(verts, list) or not verts:
        errors.append("vertices must be a non-empty list")
    else:
        for i, v in enumerate(verts):
            if not (isinstance(v, list) and len(v) == 2 and
                    all(isinstance(x, int) for x in v)):
                errors.append(f"vertex {i} must be [int, int]")
                continue
            c, r = v
            if not (0 <= c < n_cols):
                errors.append(f"vertex {i} col {c} out of range [0, {n_cols})")
            if not (0 <= r < n_rows):
                errors.append(f"vertex {i} row {r} out of range [0, {n_rows})")

    edges = submission.get("edges", [])
    if not isinstance(edges, list) or not edges:
        errors.append("edges must be a non-empty list")
    else:
        n_verts = len(verts)
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                errors.append(f"edge {i} must be an object")
                continue
            for f in ("src", "dst", "wrap"):
                if f not in e:
                    errors.append(f"edge {i} missing field '{f}'")
            if "src" in e and not (0 <= e["src"] < n_verts):
                errors.append(f"edge {i} src {e['src']} out of range [0, {n_verts})")
            if "dst" in e and not (0 <= e["dst"] < n_verts):
                errors.append(f"edge {i} dst {e['dst']} out of range [0, {n_verts})")
            if "wrap" in e:
                w = e["wrap"]
                if not (isinstance(w, list) and len(w) == 2 and
                        all(isinstance(x, int) for x in w)):
                    errors.append(f"edge {i} wrap must be [int, int]")

    return errors


def main():
    """When run as a script, print the schema as formatted JSON to stdout."""
    json.dump(SUBMISSION_SCHEMA, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
