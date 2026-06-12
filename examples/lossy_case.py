#!/usr/bin/env python3
"""Figura 2: pérdidas modales exactas vs perturbativas."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.constants import GeometryShape, Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian, solve_nonhermitian
from src.operators import assemble_systems
from src.perturbation import compare_loss_methods, compute_all_perturbative
from src.utils import kpath_gamma_x, setup_logging
from src.visualization import plot_loss_comparison


def main() -> None:
    logger = setup_logging()

    gamma = 0.01

    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,  # lado s/a
        shape=GeometryShape.SQUARE,
        eps_inf_metal=1.0,
        omega_p_metal=1.0,
        omega_0_metal=0.0,
        eps_inf_air=1.0,
        omega_p_air=0.0,
        omega_0_air=1.0e12,
        gamma=gamma,
        nk=40,
        nbands=35,
        sigma=0.35,
        output_dir=ROOT / "results",
        solver_tol=1e-9,
        solver_maxiter=2000,
    )

    grid = YeeGrid.from_config(config)
    solver = SolverConfig(
        nev=config.nbands,
        sigma=config.default_sigma(),
        tol=config.solver_tol,
        maxiter=config.solver_maxiter,
        omega_min=config.omega_min,
    )

    # Para Fig. 2 se comparan todos los modos en un k-point representativo.
    kx = float(kpath_gamma_x(config.nk)[config.nk // 2])
    logger.info(f"k-point central: kx/π = {kx / np.pi:.3f}")

    h0, h_lossy, _ = assemble_systems(grid, kx, 0.0, gamma=gamma)

    lossless = solve_hermitian(h0, grid, solver)
    exact = solve_nonhermitian(h_lossy, grid, solver)
    perturb = compute_all_perturbative(lossless, grid, gamma)

    cmp = compare_loss_methods(lossless, exact, perturb)

    out = config.output_dir / "figures" / "fig2_loss_comparison.png"
    plot_loss_comparison(
        cmp,
        out,
        title=r"TE modal loss — $\gamma = 0.01\omega_p$",
        convert_to_paper_units=True,
    )

    logger.info(f"Figura guardada en {out}")

    for c in cmp[:10]:
        logger.info(
            f"ωa/2πc={c.omega_real / (2 * np.pi):.5f} "
            f"-Im_exact/2π={-c.im_exact / (2 * np.pi):.6e} "
            f"-Im_pert/2π={-c.im_perturbative / (2 * np.pi):.6e} "
            f"err={c.relative_error:.2%}"
        )


if __name__ == "__main__":
    main()
