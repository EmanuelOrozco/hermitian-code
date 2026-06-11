#!/usr/bin/env python3
"""Caso 2: pérdidas modales — comparación exacta vs perturbativa (Fig. 2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian, solve_nonhermitian
from src.operators import assemble_systems
from src.perturbation import compare_loss_methods, compute_all_perturbative
from src.utils import kpath_gamma_x, setup_logging
from src.visualization import plot_loss_comparison


def main() -> None:
    logger = setup_logging()
    gamma = 0.01  # Γ = 0.01 ωp como en el paper

    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,
        gamma=gamma,
        nk=20,
        nbands=15,
        output_dir=ROOT / "results",
    )
    grid = YeeGrid.from_config(config)
    solver = SolverConfig(nev=config.nbands, sigma=config.default_sigma())

    kx = float(kpath_gamma_x(config.nk)[config.nk // 2])
    logger.info(f"k-point central: kx/π = {kx / 3.14159:.3f}")

    h0, h_lossy, _ = assemble_systems(grid, kx, 0.0, gamma=gamma)
    lossless = solve_hermitian(h0, grid, solver)
    exact = solve_nonhermitian(h_lossy, grid, solver)
    perturb = compute_all_perturbative(lossless, grid, gamma)
    cmp = compare_loss_methods(lossless, exact, perturb)

    out = config.output_dir / "figures" / "lossy_case_comparison.png"
    plot_loss_comparison(cmp, out, title=f"Pérdidas TE — Γ = {gamma} ωp")
    logger.info(f"Figura guardada en {out}")

    for c in cmp[:5]:
        logger.info(
            f"ω={c.omega_real:.4f}  Im_exact={c.im_exact:.6f}  "
            f"Im_pert={c.im_perturbative:.6f}  err={c.relative_error:.2%}"
        )


if __name__ == "__main__":
    main()
