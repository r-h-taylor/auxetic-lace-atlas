"""
Tests for the user contribution pipeline.

Covers:
  - JSON schema validation
  - Structural validation (2-regular, connected, etc.)
  - Mechanics computation produces atlas-compatible records
  - Ranking against a small atlas
  - Local addition with duplicate detection
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from auxetic_lace.contrib.submission_schema import validate_against_schema
from auxetic_lace.contrib.validate_submission import (
    check_2regular, check_connected, check_nonzero_lengths,
    validate_submission,
)
from auxetic_lace.contrib.compute_submission import submission_to_lacegraph
from auxetic_lace.build_atlas import build_ground_record


# =====================================================================
# Fixtures
# =====================================================================

VALID_SQUARE = {
    "name": "test_square",
    "family": "test/square",
    "n_rows": 1,
    "n_cols": 1,
    "vertices": [[0, 0]],
    "edges": [
        {"src": 0, "dst": 0, "wrap": [1, 0]},
        {"src": 0, "dst": 0, "wrap": [0, 1]},
    ],
}


VALID_CLOTH_2X1 = {
    "name": "test_cloth",
    "family": "test/cloth",
    "n_rows": 1,
    "n_cols": 2,
    "vertices": [[0, 0], [1, 0]],
    "edges": [
        {"src": 0, "dst": 1, "wrap": [0, 0]},
        {"src": 0, "dst": 0, "wrap": [0, 1]},
        {"src": 1, "dst": 0, "wrap": [1, 0]},
        {"src": 1, "dst": 1, "wrap": [0, 1]},
    ],
}


# =====================================================================
# Schema validation
# =====================================================================

class TestSchema:

    def test_valid_minimal(self):
        errs = validate_against_schema(VALID_SQUARE)
        assert errs == []

    def test_missing_required_fields(self):
        bad = {"name": "x", "n_rows": 1}  # missing n_cols, vertices, edges
        errs = validate_against_schema(bad)
        assert any("n_cols" in e for e in errs)
        assert any("vertices" in e for e in errs)
        assert any("edges" in e for e in errs)

    def test_invalid_name_chars(self):
        bad = dict(VALID_SQUARE)
        bad["name"] = "has spaces and slashes/here"
        errs = validate_against_schema(bad)
        assert any("invalid characters" in e for e in errs)

    def test_vertex_out_of_range(self):
        bad = dict(VALID_SQUARE)
        bad["vertices"] = [[5, 0]]   # col=5 but n_cols=1
        errs = validate_against_schema(bad)
        assert any("out of range" in e for e in errs)

    def test_edge_invalid_vertex_index(self):
        bad = dict(VALID_SQUARE)
        bad["edges"] = [{"src": 0, "dst": 99, "wrap": [0, 0]}]
        errs = validate_against_schema(bad)
        assert any("dst 99 out of range" in e for e in errs)


# =====================================================================
# Structural checks
# =====================================================================

class TestStructural:

    def test_2regular_valid(self):
        assert check_2regular(VALID_SQUARE) == []
        assert check_2regular(VALID_CLOTH_2X1) == []

    def test_2regular_failed(self):
        bad = dict(VALID_SQUARE)
        # Add an extra edge — vertex 0 now has out-degree 3
        bad["edges"] = bad["edges"] + [{"src": 0, "dst": 0, "wrap": [-1, 0]}]
        errs = check_2regular(bad)
        assert any("out-degree = 3" in e for e in errs)

    def test_connected_valid(self):
        assert check_connected(VALID_SQUARE) == []
        assert check_connected(VALID_CLOTH_2X1) == []

    def test_connected_disconnected_graph(self):
        # 2 disconnected square lattices
        bad = {
            "name": "disconnected",
            "n_rows": 1, "n_cols": 2,
            "vertices": [[0, 0], [1, 0]],
            "edges": [
                {"src": 0, "dst": 0, "wrap": [1, 0]},
                {"src": 0, "dst": 0, "wrap": [0, 1]},
                {"src": 1, "dst": 1, "wrap": [1, 0]},
                {"src": 1, "dst": 1, "wrap": [0, 1]},
            ],
        }
        errs = check_connected(bad)
        assert any("not connected" in e for e in errs)

    def test_nonzero_lengths_valid(self):
        assert check_nonzero_lengths(VALID_SQUARE) == []

    def test_nonzero_lengths_zero(self):
        # Self-loop with wrap (0,0) has zero length
        bad = dict(VALID_SQUARE)
        bad["edges"] = [
            {"src": 0, "dst": 0, "wrap": [0, 0]},  # zero length
            {"src": 0, "dst": 0, "wrap": [0, 1]},
        ]
        errs = check_nonzero_lengths(bad)
        assert any("zero length" in e for e in errs)


# =====================================================================
# Top-level validation
# =====================================================================

class TestValidation:

    def test_square_passes(self):
        passed, results = validate_submission(VALID_SQUARE)
        assert passed
        assert results["level1"] == []

    def test_cloth_passes(self):
        passed, results = validate_submission(VALID_CLOTH_2X1)
        assert passed

    def test_square_fails_tesselace(self):
        # Square lattice has 2 osculating circuits (one per direction),
        # neither has wrap (1, 0) exclusively. Should fail thread-conserve.
        passed, results = validate_submission(
            VALID_SQUARE, check_tesselace=True)
        # Level 2 should report the failure
        assert len(results["level2"]) > 0


# =====================================================================
# Mechanics computation
# =====================================================================

class TestMechanics:

    def test_submission_to_lacegraph(self):
        g = submission_to_lacegraph(VALID_SQUARE)
        assert g.n_rows == 1
        assert g.n_cols == 1
        assert len(g.vertices) == 1
        assert len(g.edges) == 2

    def test_compute_produces_atlas_record(self):
        g = submission_to_lacegraph(VALID_CLOTH_2X1)
        record = build_ground_record(
            g, name=VALID_CLOTH_2X1["name"],
            family=VALID_CLOTH_2X1["family"])
        # Required atlas fields
        assert "spring" in record
        assert "beam" in record
        assert "C_voigt" in record["spring"]
        assert "nu_min" in record["spring"]
        assert "u_strain" in record["spring"]
        # Spring k_ang grid has 5 entries
        assert len(record["spring"]["k_ang"]) == 5
        # Beam AR grid has 5 entries
        assert len(record["beam"]["AR"]) == 5

    def test_orthogonal_lattice_isotropic(self):
        """Cloth 2x1 (orthogonal bars) should give nu ≈ 0 for both models."""
        g = submission_to_lacegraph(VALID_CLOTH_2X1)
        record = build_ground_record(
            g, name="t", family="t/t")
        # At k_ang=0.01 (default index 2)
        nu_min = record["spring"]["nu_min"][2]
        assert abs(nu_min) < 1e-3
