"""Visualización de bandas, campos y pérdidas."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .eigenmodes import ModeFields, reshape_field
from .grid import YeeGrid
from .perturbation import LossComparison


def _save_fig(path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_band_structure(
    k_norm: np.ndarray,
    bands: np.ndarray,
    output: Path,
    title: str = "TE band structure — square plasmonic rods",
    ylabel: str = r"Frequency $\omega a / 2\pi c$",
    convert_to_paper_units: bool = True,
) -> None:
    """Grafica estructura de bandas.

    El código interno usa ω en unidades c/a.
    Fig. 1 del paper grafica ωa/(2πc).
    """
    y = bands / (2.0 * np.pi) if convert_to_paper_units else bands

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for n in range(y.shape[1]):
        ax.plot(k_norm, y[:, n], "-", lw=0.8, alpha=0.9)

    ax.set_xlabel(r"Wave vector $k_x / (\pi/a)$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(np.nanmin(k_norm)), float(np.nanmax(k_norm)))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)

    _save_fig(output)


def plot_field_intensity(
    fields: ModeFields,
    grid: YeeGrid,
    output: Path,
    components: Optional[Sequence[str]] = None,
    vmin: float = -6.5,
    vmax: float = -3.0,
) -> None:
    """Mapas log10|campo|² para reproducir paneles tipo Fig. 1(b,c)."""
    n = grid.N
    comps = list(components or [])

    if not comps:
        preferred = ["Ex", "Vx"]
        comps = [c for c in preferred if c in fields.E or c in fields.V]
        if not comps:
            comps = list(fields.E.keys()) + list(fields.V.keys())

    fig, axes = plt.subplots(1, len(comps), figsize=(3.8 * len(comps), 3.6))

    if len(comps) == 1:
        axes = [axes]

    for ax, name in zip(axes, comps):
        if name in fields.E:
            data = fields.E[name]
        elif name in fields.V:
            data = fields.V[name]
        else:
            data = fields.H[name]

        img = np.log10(np.abs(reshape_field(data, n)) ** 2 + 1e-30)
        im = ax.imshow(
            img.T,
            origin="lower",
            cmap="hot",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(rf"$\log_{{10}} |{name}|^2$")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(rf"$\omega a/2\pi c = {fields.omega / (2 * np.pi):.4f}$")
    _save_fig(output)


def plot_loss_comparison(
    comparisons: List[LossComparison],
    output: Path,
    title: str = r"Modal loss, TE case: $\gamma = 0.01\omega_p$",
    convert_to_paper_units: bool = True,
) -> None:
    """Figura tipo Fig. 2: Im[ω] vs Re[ω].

    Se grafica -Im(ω) como cantidad positiva de pérdida/inverse lifetime.
    """
    omega = np.array([c.omega_real for c in comparisons], dtype=float)
    im_ex = np.array([-c.im_exact for c in comparisons], dtype=float)
    im_pt = np.array([-c.im_perturbative for c in comparisons], dtype=float)

    if convert_to_paper_units:
        omega = omega / (2.0 * np.pi)
        im_ex = im_ex / (2.0 * np.pi)
        im_pt = im_pt / (2.0 * np.pi)
        xlabel = r"Real frequency $\omega'a/2\pi c$"
        ylabel = r"Imaginary frequency $-\omega''a/2\pi c$"
    else:
        xlabel = r"Re[$\omega$]"
        ylabel = r"$-\mathrm{Im}[\omega]$"

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.plot(omega, im_ex, "o", ms=4, label="Exact non-Hermitian")
    ax.plot(omega, im_pt, "x", ms=5, label="Perturbation theory")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0.0)

    _save_fig(output)


def plot_convergence(
    resolutions: Sequence[int],
    frequencies: Sequence[float],
    reference: float,
    output: Path,
) -> None:
    errors = [abs(f - reference) / abs(reference) for f in frequencies]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(resolutions, errors, "o-")
    ax.set_xlabel("Resolution N")
    ax.set_ylabel("Relative error")
    ax.set_title("Spatial convergence")
    ax.grid(True, which="both", alpha=0.3)

    _save_fig(output)


def plot_energy_decomposition(
    e_frac: float,
    h_frac: float,
    v_frac: float,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ["E", "H", "V"]
    vals = [e_frac, h_frac, v_frac]

    ax.bar(labels, vals)
    ax.set_ylabel("Energy fraction")
    ax.set_ylim(0, 1)

    _save_fig(output)
