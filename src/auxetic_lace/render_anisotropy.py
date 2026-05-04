"""
render_anisotropy.py
====================

Render a two-panel polar plot for a single ground:

  Left panel:  E(theta) and G(theta) — stiffness anisotropy
               E in dark red (solid), G in slate blue (dashed)

  Right panel: |nu(theta)| — Poisson anisotropy
               Red where nu<0 (auxetic), blue where nu>=0 (normal)
               Origin = nu=0 boundary

Both panels share the angular axis. Default uses the beam-model C_voigt
at AR=10. Cream-paper styling matches the other thumbnails.

Public:

    render_ground_anisotropy(graph, output_path,
                              ar=10.0, n_angle_samples=180,
                              title=None, dpi=120) -> dict

Returns {E_min, E_max, G_min, G_max, K, nu_min, nu_max, classification}.
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
    poisson_profile,
)
from .mechanics_beam import homogenize_beam


CREAM_BG = "#f5f0e3"
CREAM_PAPER = "#fffaf0"
DARK_TEXT = "#2a2520"
E_COLOR = "#a83232"        # dark red — Young's modulus
G_COLOR = "#3a5a7a"        # slate blue (dashed) — shear modulus
NU_AUX = "#a83232"         # dark red — auxetic ν<0
NU_NORM = "#3a5a7a"        # slate blue — normal ν≥0
GRID_COLOR = "#bbb6a8"

NU_PLOT_CAP = 5.0          # cap |ν| at this for radial display


def _classify(nu_min, nu_max):
    if not (np.isfinite(nu_min) and np.isfinite(nu_max)):
        return "fully_floppy"
    if nu_max < 0:
        return "homogeneously_auxetic"
    if nu_min < 0:
        return "directionally_auxetic"
    return "non_auxetic"


def _polar_axis(ax):
    """Apply consistent polar styling."""
    ax.tick_params(colors=DARK_TEXT, labelsize=8)
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.spines["polar"].set_color(DARK_TEXT)
    ax.spines["polar"].set_linewidth(1)
    ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)


def _mirror(thetas, values):
    """Mirror profile from [0, pi) to [0, 2pi) for closed polar curve."""
    return (np.concatenate([thetas, thetas + np.pi]),
             np.concatenate([values, values]))


def render_ground_anisotropy(graph: LaceGraph,
                              output_path: str,
                              ar: float = 10.0,
                              n_angle_samples: int = 180,
                              title: Optional[str] = None,
                              dpi: int = 120,
                              ) -> dict:
    """Two-panel polar plot of E/G (left) and |ν| color-by-sign (right)."""

    # Compute homogenized C
    C, *_ = homogenize_beam(graph, aspect_ratio=ar)

    # Profiles in [0, pi)
    thetas, Es = youngs_profile(C, n_samples=n_angle_samples)
    _, Gs = shear_profile(C, n_samples=n_angle_samples)
    _, nus = poisson_profile(C, n_samples=n_angle_samples)

    # Mirror to [0, 2pi)
    th_full, Es_full = _mirror(thetas, Es)
    _, Gs_full = _mirror(thetas, Gs)
    _, nus_full = _mirror(thetas, nus)

    # Replace non-finite for plotting
    Es_clean = np.where(np.isfinite(Es_full), Es_full, 0.0)
    Gs_clean = np.where(np.isfinite(Gs_full), Gs_full, 0.0)

    # Stats from original (unmirrored, uncapped) profiles
    finite_E = Es[np.isfinite(Es)]
    finite_G = Gs[np.isfinite(Gs)]
    finite_nu = nus[np.isfinite(nus)]
    E_min = float(finite_E.min()) if finite_E.size else float("nan")
    E_max = float(finite_E.max()) if finite_E.size else float("nan")
    G_min = float(finite_G.min()) if finite_G.size else float("nan")
    G_max = float(finite_G.max()) if finite_G.size else float("nan")
    nu_min = float(finite_nu.min()) if finite_nu.size else float("nan")
    nu_max = float(finite_nu.max()) if finite_nu.size else float("nan")
    K = float(area_bulk_modulus(C))
    classification = _classify(nu_min, nu_max)

    # Cap ν for radial display
    nus_capped = np.where(np.isfinite(nus_full),
                            np.clip(nus_full, -NU_PLOT_CAP, NU_PLOT_CAP),
                            np.nan)

    # ===== Plot ===== #
    fig = plt.figure(figsize=(11, 5.5), facecolor=CREAM_BG)
    ax_eg = fig.add_subplot(1, 2, 1, projection="polar", facecolor=CREAM_PAPER)
    ax_nu = fig.add_subplot(1, 2, 2, projection="polar", facecolor=CREAM_PAPER)

    # ----- LEFT: E and G ----- #
    ax_eg.plot(th_full, Es_clean, color=E_COLOR, linewidth=2.2,
                label="E(θ)")
    ax_eg.plot(th_full, Gs_clean, color=G_COLOR, linewidth=2.0,
                linestyle="--", label="G(θ)")
    _polar_axis(ax_eg)
    rmax_eg = max(np.nanmax(Es_clean), np.nanmax(Gs_clean), 1e-12) * 1.1
    ax_eg.set_rmin(0)
    ax_eg.set_rmax(rmax_eg)
    ax_eg.set_title(
        f"stiffness moduli\n"
        f"E: {E_min:.3g}–{E_max:.3g}  ({E_max/max(E_min,1e-12):.1f}×);   "
        f"G: {G_min:.3g}–{G_max:.3g};   K = {K:.3g}",
        color=DARK_TEXT, fontsize=9, pad=12,
    )
    ax_eg.legend(loc="lower right", bbox_to_anchor=(1.18, -0.05),
                  framealpha=0.85, fontsize=9)

    # ----- RIGHT: |ν|, sign-colored ----- #
    # Use NaN-masking so a single plot call shows two-color line
    neg_mask = nus_capped < 0
    pos_mask = (nus_capped >= 0) & np.isfinite(nus_capped)
    abs_nu = np.abs(nus_capped)

    neg_radius = np.where(neg_mask, abs_nu, np.nan)
    pos_radius = np.where(pos_mask, abs_nu, np.nan)

    ax_nu.plot(th_full, neg_radius, color=NU_AUX, linewidth=2.2,
                label="ν < 0  (auxetic)")
    ax_nu.plot(th_full, pos_radius, color=NU_NORM, linewidth=2.2,
                label="ν ≥ 0  (normal)")
    _polar_axis(ax_nu)
    rmax_nu = max(np.nanmax(np.abs(nus_capped)) if abs_nu.size else 0.0,
                   0.1) * 1.1
    ax_nu.set_rmin(0)
    ax_nu.set_rmax(rmax_nu)

    capped_note = (" (radius capped at 5)"
                    if (abs(nu_min) > NU_PLOT_CAP
                         or abs(nu_max) > NU_PLOT_CAP)
                    else "")
    ax_nu.set_title(
        f"Poisson anisotropy   |ν|\n"
        f"ν: {nu_min:.3g} → {nu_max:.3g};   "
        f"{classification.replace('_', ' ')}{capped_note}",
        color=DARK_TEXT, fontsize=9, pad=12,
    )
    ax_nu.legend(loc="lower right", bbox_to_anchor=(1.20, -0.05),
                  framealpha=0.85, fontsize=9)

    # ----- Figure-level title and footer ----- #
    if title is None:
        title = f"{graph.family}/{graph.name}"
    fig.suptitle(
        f"{title}  —  directional response (beam, AR={ar:g})",
        color=DARK_TEXT, fontsize=11, y=0.99,
    )
    fig.text(0.5, 0.02, "stiffness in units of EA, with EA=1 reference",
              ha="center", color=DARK_TEXT, fontsize=8, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(output_path, dpi=dpi, facecolor=CREAM_BG,
                 bbox_inches="tight")
    plt.close(fig)

    return {
        "E_min": E_min, "E_max": E_max,
        "G_min": G_min, "G_max": G_max,
        "K": K,
        "nu_min": nu_min, "nu_max": nu_max,
        "classification": classification,
    }
