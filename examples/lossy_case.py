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
from src.reproduction import (
    collect_loss_comparison,
    omega_to_paper_units,
    paper_units_to_omega,
    write_json,
)
from src.utils import kpath_gamma_x, setup_logging
from src.visualization import plot_loss_comparison

PAPER_SHIFTS = [0.08, 0.12, 0.16, 0.25, 0.50, 0.75, 1.00, 1.12]
MODES_PER_SHIFT = 14
DEDUP_TOL = 2.5e-4


def main() -> None:
    logger = setup_logging()

    omega_p = float(paper_units_to_omega(1.0))
    gamma = 0.01 * omega_p

    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,  # lado s/a
        shape=GeometryShape.SQUARE,
        eps_inf_metal=1.0,
        omega_p_metal=omega_p,
        omega_0_metal=0.0,
        eps_inf_air=1.0,
        omega_p_air=0.0,
        omega_0_air=1.0e12,
        gamma=gamma,
        nk=40,
        nbands=45,
        sigma=paper_units_to_omega(0.35),
        output_dir=ROOT / "results",
        solver_tol=1e-9,
        solver_maxiter=3000,
    )

    grid = YeeGrid.from_config(config)
    base_solver = SolverConfig(
        nev=MODES_PER_SHIFT,
        sigma=config.default_sigma(),
        tol=config.solver_tol,
        maxiter=config.solver_maxiter,
        omega_min=config.omega_min,
    )

    kx_path = kpath_gamma_x(config.nk)
    cmp = []
    for ik, kx in enumerate(kx_path):
        k_cmp = collect_loss_comparison(
            grid,
            float(kx),
            0.0,
            gamma,
            PAPER_SHIFTS,
            base_solver,
            MODES_PER_SHIFT,
            ncv=64,
            duplicate_tol_paper=DEDUP_TOL,
        )
        cmp.extend(k_cmp)
        logger.info(
            f"k[{ik:02d}] kx/π={kx / np.pi:.3f} loss_modes={len(k_cmp)}"
        )

    if not cmp:
        raise RuntimeError("No loss comparison modes were generated.")

    out = config.output_dir / "figures" / "fig2_loss_comparison.png"
    cmp_for_plot = [c for c in cmp if omega_to_paper_units(c.omega_real) <= 1.15]
    plot_loss_comparison(
        cmp_for_plot,
        out,
        title=r"TE modal loss — $\gamma = 0.01\omega_p$",
        convert_to_paper_units=True,
        paper_style=True,
    )

    logger.info(f"Figura guardada en {out}")

    data_dir = config.output_dir / "band_structures"
    data_dir.mkdir(parents=True, exist_ok=True)
    data = np.array(
        [
            [
                omega_to_paper_units(c.omega_real),
                omega_to_paper_units(c.im_exact),
                omega_to_paper_units(c.im_perturbative),
                c.relative_error,
            ]
            for c in cmp
        ],
        dtype=float,
    )
    np.savetxt(
        data_dir / "fig2_loss_comparison.dat",
        data,
        header=(
            "omega_real_paper_units im_exact_paper_units "
            "im_perturbative_paper_units relative_error"
        ),
    )
    significant = np.abs(data[:, 1]) >= 1.0e-4
    write_json(
        data_dir / "fig2_loss_comparison_meta.json",
        {
            "case": "fig2_loss_comparison",
            "mode": config.mode.value,
            "resolution": config.resolution,
            "square_side_over_a": config.fill_fraction,
            "omega_p_internal": omega_p,
            "omega_p_paper_units": float(omega_to_paper_units(omega_p)),
            "gamma_internal": gamma,
            "gamma_over_omega_p": gamma / omega_p,
            "k_path": "Gamma-X",
            "nk": config.nk,
            "paper_shifts": PAPER_SHIFTS,
            "internal_sigma_values": [float(paper_units_to_omega(s)) for s in PAPER_SHIFTS],
            "modes_per_shift": MODES_PER_SHIFT,
            "duplicate_tolerance_paper_units": DEDUP_TOL,
            "num_modes": len(cmp),
            "frequency_min_paper_units": float(np.nanmin(data[:, 0])),
            "frequency_max_paper_units": float(np.nanmax(data[:, 0])),
            "max_relative_error": float(np.nanmax(data[:, 3])),
            "max_relative_error_loss_ge_1e-4": float(np.nanmax(data[significant, 3])),
            "mean_relative_error": float(np.nanmean(data[:, 3])),
        },
    )

    for c in cmp[:10]:
        logger.info(
            f"ωa/2πc={omega_to_paper_units(c.omega_real):.5f} "
            f"-Im_exact/2π={-omega_to_paper_units(c.im_exact):.6e} "
            f"-Im_pert/2π={-omega_to_paper_units(c.im_perturbative):.6e} "
            f"err={c.relative_error:.2%}"
        )


if __name__ == "__main__":
    main()
