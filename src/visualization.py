"""Visualización de bandas, campos y comparaciones de pérdidas."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .constants import PI, LATTICE_A
from .eigenmodes import ModeFields, reshape_field
from .grid import YeeGrid
from .perturbation import LossComparison


def _save_fig(path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_band_structure(
    k_norm: np.ndarray,
    bands: np.ndarray,
    output: Path,
    title: str = "Estructura de bandas TE — barras plasmónicas",
    ylabel: str = r"$\omega c / a$",
) -> None:
    """Grafica estructura de bandas ω(k)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in range(bands.shape[1]):
        ax.plot(k_norm, bands[:, n], "b-", lw=0.8, alpha=0.85)
    ax.set_xlabel(r"$k / (\pi/a)$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(k_norm[0], k_norm[-1])
    ax.grid(True, alpha=0.3)
    _save_fig(output)


def plot_field_intensity(
    fields: ModeFields,
    grid: YeeGrid,
    output: Path,
    components: Optional[Sequence[str]] = None,
) -> None:
    """Mapas log₁₀|campo|² para componentes E y V."""
    n = grid.N
    comps = list(components or [])
    if not comps:
        comps = list(fields.E.keys()) + list(fields.V.keys())

    fig, axes = plt.subplots(1, len(comps), figsize=(4 * len(comps), 3.5))
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
        im = ax.imshow(img.T, origin="lower", cmap="hot", aspect="equal")
        ax.set_title(f"log₁₀|{name}|²")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Modo ω = {fields.omega:.4f}")
    _save_fig(output)


def plot_loss_comparison(
    comparisons: List[LossComparison],
    output: Path,
    title: str = "Pérdida modal: exacto vs perturbativo",
) -> None:
    """Figura 2 del paper: Im[ω] vs Re[ω]."""
    omega = np.array([c.omega_real for c in comparisons])
    im_ex = np.array([c.im_exact for c in comparisons])
    im_pt = np.array([c.im_perturbative for c in comparisons])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(omega, im_ex, "o", label="Exacto (NH)", ms=4)
    ax.plot(omega, im_pt, "x", label="Perturbativo (eq. 15)", ms=5)
    ax.set_xlabel(r"Re[$\omega$]")
    ax.set_ylabel(r"Im[$\omega$]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(output)


def plot_convergence(
    resolutions: Sequence[int],
    frequencies: Sequence[float],
    reference: float,
    output: Path,
) -> None:
    """Error relativo de frecuencia vs resolución."""
    errors = [abs(f - reference) / reference for f in frequencies]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(resolutions, errors, "o-")
    ax.set_xlabel("Resolución N")
    ax.set_ylabel("Error relativo")
    ax.set_title("Convergencia espacial")
    ax.grid(True, which="both", alpha=0.3)
    _save_fig(output)


def plot_energy_decomposition(
    e_frac: float, h_frac: float, v_frac: float, output: Path
) -> None:
    """Gráfico de barras de fracciones energéticas."""
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ["E", "H", "V"]
    vals = [e_frac, h_frac, v_frac]
    ax.bar(labels, vals, color=["#e74c3c", "#3498db", "#2ecc71"])
    ax.set_ylabel("Fracción de energía")
    ax.set_ylim(0, 1)
    _save_fig(output)
