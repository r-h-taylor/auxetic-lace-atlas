"""
render_dispersion.py
====================

Render a phonon-dispersion PNG for a single ground, in the same cream-paper
style as render_lace_view.py and render_lace_deformed.py. Acoustic branches
are highlighted in dark red; optical branches are dark gray. The path is
the standard rectangular Brillouin zone Gamma -> X -> M -> Gamma.

Public:

    render_ground_dispersion(graph, output_path, k_angular=0.01,
                              npts_per_seg=60, title=None, dpi=120) -> None
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .parse_to_graph import LaceGraph
from .phonons import omegas_along_path, high_symmetry_path


CREAM_BG = "#f5f0e3"
CREAM_PAPER = "#fffaf0"
DARK_TEXT = "#2a2520"
ACOUSTIC_COLOR = "#a83232"
OPTICAL_COLOR = "#2a2520"


def render_ground_dispersion(graph: LaceGraph,
                              output_path: str,
                              k_angular: float = 0.01,
                              k_per_len: float = 1.0,
                              npts_per_seg: int = 60,
                              title: Optional[str] = None,
                              dpi: int = 120,
                              ) -> None:
    """Render and save the phonon dispersion plot for one ground.

    output_path: where to write the PNG (e.g. thumbnails/.../dispersion.png).
    k_angular:   angular spring stiffness, default 0.01 (matches the spring
                 default used in build_atlas).
    title:       optional title; if None, uses '<family>/<name>'.
    """
    k_path, distances, markers = high_symmetry_path(graph, npts_per_seg)
    omegas = omegas_along_path(graph, k_path, k_angular=k_angular,
                                 k_per_len=k_per_len)
    n_bands = omegas.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(CREAM_PAPER)
    fig.patch.set_facecolor(CREAM_BG)

    # Optical bands first (so acoustic branches plot on top of them)
    for band in range(2, n_bands):
        ax.plot(distances, omegas[:, band], color=OPTICAL_COLOR, lw=0.9)
    # Acoustic branches highlighted in red
    for band in range(min(2, n_bands)):
        ax.plot(distances, omegas[:, band], color=ACOUSTIC_COLOR, lw=1.4)

    for m in markers:
        x = distances[min(m, len(distances) - 1)]
        ax.axvline(x, color="#888", lw=0.5)
    ax.set_xticks([distances[min(m, len(distances) - 1)] for m in markers])
    ax.set_xticklabels(["Γ", "X", "M", "Γ"])
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylim(0, None)
    ax.set_ylabel(r"$\omega$  (units $\sqrt{k/m}$)")
    ax.tick_params(colors=DARK_TEXT)
    for spine in ax.spines.values():
        spine.set_color(DARK_TEXT)

    if title is None:
        title = f"{graph.family}/{graph.name}    phonon dispersion"
    ax.set_title(title, color=DARK_TEXT)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=CREAM_BG)
    plt.close(fig)
