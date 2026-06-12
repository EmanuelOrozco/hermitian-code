#!/usr/bin/env python3
"""Perturbative-loss validity for several damping strengths."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.constants import GeometryShape, Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian, solve_nonhermitian
from src.operators import assemble_systems
from src.perturbation import compare_loss_methods, compute_all_perturbative
from src.utils import setup_logging


def main() -> None:
    logger = setup_logging()
    gammas = [0.001, 0.005, 0.01, 0.02]

    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,
        shape=GeometryShape.SQUARE,
        eps_inf_metal=1.0,
        omega_p_metal=1.0,
        omega_0_metal=0.0,
        eps_inf_air=1.0,
        omega_p_air=0.0,
        omega_0_air=1.0e12,
        nk=40,
        nbands=35,
        sigma=0.35,
        output_dir=ROOT / "results",
        solver_tol=1e-9,
        solver_maxiter=2500,
    )

    t0 = time.perf_counter()

    grid = YeeGrid.from_config(config)
    solver = SolverConfig(
        nev=config.nbands,
        sigma=config.default_sigma(),
        tol=config.solver_tol,
        maxiter=config.solver_maxiter,
        omega_min=config.omega_min,
    )

    kx = 0.5 * np.pi
    rows: list[list[float]] = []

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for gamma in gammas:
        h0, h_lossy, _ = assemble_systems(grid, kx, 0.0, gamma=gamma)

        lossless = solve_hermitian(h0, grid, solver)
        exact = solve_nonhermitian(h_lossy, grid, solver)
        perturb = compute_all_perturbative(lossless, grid, gamma)
        cmp = compare_loss_methods(lossless, exact, perturb)

        omega = np.array([c.omega_real / (2.0 * np.pi) for c in cmp])
        err = np.array([c.relative_error for c in cmp])

        for c in cmp:
            rows.append(
                [
                    gamma,
                    c.omega_real,
                    c.omega_real / (2.0 * np.pi),
                    c.im_exact,
                    c.im_perturbative,
                    c.relative_error,
                ]
            )

        ax.plot(omega, err, "o-", label=rf"$\gamma={gamma}\omega_p$", ms=3)

        logger.info(
            f"gamma={gamma}: mean error={np.nanmean(err):.3e}, "
            f"max error={np.nanmax(err):.3e}"
        )

    ax.set_xlabel(r"Real frequency $\omega'a/2\pi c$")
    ax.set_ylabel("Relative error")
    ax.set_title("Perturbative loss convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    figures_dir = config.output_dir / "figures"
    data_dir = config.output_dir / "band_structures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    out = figures_dir / "perturbation_vs_gamma.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    np.savetxt(
        data_dir / "perturbation_vs_gamma.dat",
        np.array(rows, dtype=float),
        header=(
            "gamma omega_real omega_real_paper_units "
            "im_exact im_perturbative relative_error"
        ),
    )

    meta = {
        "case": "perturbation_vs_gamma",
        "gammas": gammas,
        "mode": config.mode.value,
        "resolution": config.resolution,
        "square_side_over_a": config.fill_fraction,
        "kx_over_pi": kx / np.pi,
        "nbands": config.nbands,
        "sigma": solver.sigma,
        "elapsed_s": time.perf_counter() - t0,
    }
    (data_dir / "perturbation_vs_gamma_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    logger.info(f"Saved perturbation comparison figure: {out}")


if __name__ == "__main__":
    main()
