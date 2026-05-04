"""
render_anisotropy.py
====================

Render a polar plot of Young's modulus E(theta) and shear modulus
G(theta) for a single ground, in the same cream-paper style as the
other renderers. Default uses beam-model C_voigt at AR=10.

Public:

    render_ground_anisotropy(graph, output_path,
                              ar=10.0, n_angle_samples=180,
                              title=None, dpi=120) -> dict

Returns a dict of summary scalars: {E_min, E_max, G_min, G_max, K}.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parse_to_graph import LaceGraph
from .mechanics import (
    youngs_profile, shear_profile, area_bulk_modulus,
)
from .mechanics_beam import homogenize_beam


CREAM_BG = "#f5f0e3"
CREAM_PAPER = "#fffaf0"
DARK_TEXT = "#2a2520"
E_COLOR = "#a83232"        # dark red (primary)
G_COLOR = "#3a5a7a"        # slate blue (secondary)
GRID_COLOR = "#bbb6a8"


def render_ground_anisotropy(graph: LaceGraph,
                              output_path: str,
                              ar: float = 10.0,
                              n_angle_samples: int = 180,
                              title: Optional[str] = None,
                              dpi: int = 120,
                              ) -> dict:
    """Render and save the E(theta) and G(theta) polar plot for one ground.

    output_path: where to write the PNG.
    ar:          beam aspect ratio (default 10.0 — matches atlas default).
    title:       optional title; if None, uses '<family>/<name>'.
    Returns:     summary scalars {E_min, E_max, G_min, G_max, K}.
    """
    # Compute homogenized C at this AR using the beam model.
    # homogenize_beam returns (C_voigt, area, internal_disp) — we only
    # need C.
    C, *_ = homogenize_beam(graph, aspect_ratio=ar)

    # Sample E and G profiles
    thetas, Es = youngs_profile(C, n_samples=n_angle_samples)
    _,    Gs = shear_profile(C, n_samples=n_angle_samples)

    # Polar plots are doubly-periodic (E(theta+pi) = E(theta)). Mirror to
    # produce a closed curve over [0, 2pi).
    thetas_full = np.concatenate([thetas, thetas + np.pi])
    Es_full = np.concatenate([Es, Es])
    Gs_full = np.concatenate([Gs, Gs])

    # Replace any non-finite entries with 0 for plotting (keep extrema for label)
    Es_clean = np.where(np.isfinite(Es_full), Es_full, 0.0)
    Gs_clean = np.where(np.isfinite(Gs_full), Gs_full, 0.0)

    # Summary scalars
    finite_E = Es[np.isfinite(Es)]
    finite_G = Gs[np.isfinite(Gs)]
    E_min = float(finite_E.min()) if finite_E.size else float("nan")
    E_max = float(finite_E.max()) if finite_E.size else float("nan")
    G_min = float(finite_G.min()) if finite_G.size else float("nan")
    G_max = float(finite_G.max()) if finite_G.size else float("nan")
    K = float(area_bulk_modulus(C))

    # Plot
    fig = plt.figure(figsize=(5.5, 5.5), facecolor=CREAM_BG)
    ax = fig.add_subplot(111, projection="polar", facecolor=CREAM_PAPER)

    ax.plot(thetas_full, Es_clean, color=E_COLOR, linewidth=2.2,
             label="E(θ) — Young's modulus")
    ax.plot(thetas_full, Gs_clean, color=G_COLOR, linewidth=2.2,
             linestyle="--", label="G(θ) — shear modulus")

    # Style
    ax.tick_params(colors=DARK_TEXT, labelsize=9)
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.spines["polar"].set_color(DARK_TEXT)
    ax.spines["polar"].set_linewidth(1)

    # Theta tick labels — show 0/45/90/...
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # Radial limits — pad slightly above the larger of E_max and G_max
    rmax = max(np.nanmax(Es_clean), np.nanmax(Gs_clean), 1e-12) * 1.1
    ax.set_rmin(0)
    ax.set_rmax(rmax)
    ax.tick_params(axis="y", colors=DARK_TEXT, labelsize=8)

    if title is None:
        title = f"{graph.family}/{graph.name}"
    title_text = (
        f"{title}  —  stiffness anisotropy (beam, AR={ar:g})\n"
        f"E: min={E_min:.3g}  max={E_max:.3g}  ({E_max/max(E_min,1e-12):.1f}×);   "
        f"G: min={G_min:.3g}  max={G_max:.3g};   K = {K:.3g}\n"
        f"(values in units of EA, with EA=1 reference)"
    )
    fig.suptitle(title_text, color=DARK_TEXT, fontsize=10, y=0.99)

    # Legend in the lower-right corner outside the polar circle
    ax.legend(loc="lower right", bbox_to_anchor=(1.18, -0.05),
               framealpha=0.85, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=CREAM_BG,
                 bbox_inches="tight")
    plt.close(fig)

    return {
        "E_min": E_min, "E_max": E_max,
        "G_min": G_min, "G_max": G_max,
        "K": K,
    }
