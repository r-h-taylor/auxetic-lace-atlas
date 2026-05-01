"""
Validation tests for beam mechanics on hand-built simple graphs.

These tests don't require the TesseLace catalog to be scraped — they
build the test graphs in code so the test suite runs in any environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from auxetic_lace.parse_to_graph import LaceGraph, Edge
from auxetic_lace.mechanics import analyze
from auxetic_lace.mechanics_beam import analyze_beam


# =====================================================================
# Test fixtures: hand-built canonical graphs
# =====================================================================

def make_square_lattice() -> LaceGraph:
    """1×1 square lattice: 1 vertex, horizontal + vertical wrapping edges."""
    return LaceGraph(
        name="square_1x1", family="test", keyword="Lattice Path",
        n_rows=1, n_cols=1,
        vertices=[(0, 0)],
        edges=[
            Edge(src_idx=0, dst_idx=0, wrap=(1, 0), polyline=()),
            Edge(src_idx=0, dst_idx=0, wrap=(0, 1), polyline=()),
        ],
    )


def make_cloth_2x1() -> LaceGraph:
    """2x1 cloth: 2 vertices, horizontals + vertical self-loops."""
    return LaceGraph(
        name="cloth_2x1", family="test_cloth", keyword="Lattice Path",
        n_rows=1, n_cols=2,
        vertices=[(0, 0), (1, 0)],
        edges=[
            Edge(src_idx=0, dst_idx=1, wrap=(0, 0), polyline=()),
            Edge(src_idx=0, dst_idx=0, wrap=(0, 1), polyline=()),
            Edge(src_idx=1, dst_idx=0, wrap=(1, 0), polyline=()),
            Edge(src_idx=1, dst_idx=1, wrap=(0, 1), polyline=()),
        ],
    )


# =====================================================================
# Tests
# =====================================================================

class TestSquareLattice:
    """The simplest 2-regular periodic graph: 1 vertex per cell, two
    orthogonal axial bars wrapping to itself."""

    def test_orthotropic_response(self):
        """Square lattice should be isotropic: C11 = C22, C12 ≈ 0,
        Poisson ratio ≈ 0."""
        g = make_square_lattice()
        result = analyze_beam(g, name="square_1x1", aspect_ratio=10.0)
        c = result.C_voigt
        assert abs(c[0, 0] - c[1, 1]) < 1e-6, "C11 != C22"
        assert abs(c[0, 1]) < 1e-6, "C12 should be ~0 for orthogonal bars"
        # nu_min should be very close to 0 (orthogonal lattice)
        assert abs(result.nu_min) < 1e-4

    @pytest.mark.parametrize("AR", [5.0, 10.0, 50.0, 200.0])
    def test_axial_stiffness_independent_of_AR(self, AR):
        """Axial stiffness C11=C22 should be 1.0 (one bar, k=1) for all
        aspect ratios — bending doesn't affect axial response of a
        purely-axially-loaded orthogonal lattice."""
        g = make_square_lattice()
        result = analyze_beam(g, name="square_1x1", aspect_ratio=AR)
        assert abs(result.C_voigt[0, 0] - 1.0) < 1e-4
        assert abs(result.C_voigt[1, 1] - 1.0) < 1e-4

    def test_C66_scales_with_AR_squared(self):
        """Shear stiffness C66 should scale as 1/AR^2, since shearing
        a square lattice requires bending the bars."""
        g = make_square_lattice()
        r5 = analyze_beam(g, name="square_1x1", aspect_ratio=5.0)
        r50 = analyze_beam(g, name="square_1x1", aspect_ratio=50.0)
        # Ratio should be ~ (50/5)^2 = 100
        ratio = r5.C_voigt[2, 2] / r50.C_voigt[2, 2]
        assert 80 < ratio < 130, f"C66 ratio {ratio} not ~100 (= AR^2)"


class TestClothPattern:
    """The simplest tesselace ground: 2 vertices per cell, horizontal +
    vertical self-loops."""

    def test_orthotropic_response(self):
        g = make_cloth_2x1()
        result = analyze_beam(g, name="cloth_2x1", aspect_ratio=10.0)
        c = result.C_voigt
        assert abs(c[0, 0] - c[1, 1]) < 1e-6
        assert abs(c[0, 1]) < 1e-6
        assert abs(result.nu_min) < 1e-4


class TestBeamConvergesToSpring:
    """At very high aspect ratio, beam mechanics should match the
    pin-jointed (axial-only) spring mechanics. This is the key
    verification that the beam implementation is correct."""

    @pytest.mark.parametrize("AR", [200.0, 1000.0, 5000.0])
    def test_cloth_2x1_convergence(self, AR):
        g = make_cloth_2x1()
        # Spring at k_ang=0 (pure pin-jointed)
        spring_result = analyze(g, k_angular=0.0)
        beam_result = analyze_beam(g, name="cloth_2x1", aspect_ratio=AR)

        # C11 and C22 should match to high precision
        c_spring = spring_result['C']
        c_beam = beam_result.C_voigt
        # Tolerance: the beam adds a small bending coupling at finite AR
        # that decays as 1/AR^2, so at AR=200 we should be within 1e-3
        tol = 10.0 / (AR ** 2)
        assert abs(c_beam[0, 0] - c_spring[0, 0]) < tol, \
            f"C11 mismatch at AR={AR}: beam={c_beam[0, 0]}, spring={c_spring[0, 0]}"
        assert abs(c_beam[1, 1] - c_spring[1, 1]) < tol, \
            f"C22 mismatch at AR={AR}: beam={c_beam[1, 1]}, spring={c_spring[1, 1]}"
