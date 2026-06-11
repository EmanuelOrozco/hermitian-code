#!/usr/bin/env python3
"""Caso 4: comparación detallada perturbación vs solución exacta con pérdidas."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.constants import Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian, solve_nonhermitian
from src.operators import assemble_systems
from src.perturbation import compare_loss_methods, compute_all_perturbative
from src.utils import setup_logging
import matplotlib.pyplot as plt


def main() -> None:
    logger = setup_logging()
    gammas = [0.001, 0.005, 0.01, 0.02]
    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,
        nk=20,
        nbands=10,
        output_dir=ROOT / "results",
    )
    grid = YeeGrid.from_config(config)
    solver = SolverConfig(nev=config.nbands, sigma=config.default_sigma())
    kx = 0.5 * np.pi  # punto intermedio Γ→X

    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma in gammas:
        h0, h_lossy, _ = assemble_systems(grid, kx, 0.0, gamma=gamma)
        lossless = solve_hermitian(h0, grid, solver)
        exact = solve_nonhermitian(h_lossy, grid, solver)
        perturb = compute_all_perturbative(lossless, grid, gamma)
        cmp = compare_loss_methods(lossless, exact, perturb)
        omega = [c.omega_real for c in cmp]
        err = [c.relative_error for c in cmp]
        ax.plot(omega, err, "o-", label=f"Γ={gamma}ωp", ms=3)

    ax.set_xlabel(r"Re[$\omega$]")
    ax.set_ylabel("Error relativo |Im_exact − Im_pert| / |Im_exact|")
    ax.set_title("Convergencia perturbativa vs Γ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = config.output_dir / "figures" / "perturbation_vs_gamma.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Guardado en {out}")


if __name__ == "__main__":
    main()
