#!/usr/bin/env python3
"""Spatial convergence of a representative TE frequency."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.constants import GeometryShape, Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian
from src.operators import assemble_systems
from src.reproduction import paper_units_to_omega
from src.utils import setup_logging
from src.visualization import plot_convergence


def main() -> None:
    logger = setup_logging()

    resolutions = [10, 15, 20, 25, 30]
    freqs: list[float] = []

    t0 = time.perf_counter()

    for n in resolutions:
        omega_p = float(paper_units_to_omega(1.0))
        config = SimConfig(
            mode=Polarization.TE,
            resolution=n,
            fill_fraction=0.25,
            shape=GeometryShape.SQUARE,
            eps_inf_metal=1.0,
            omega_p_metal=omega_p,
            omega_0_metal=0.0,
            eps_inf_air=1.0,
            omega_p_air=0.0,
            omega_0_air=1.0e12,
            nk=1,
            nbands=8,
            sigma=paper_units_to_omega(0.35),
            output_dir=ROOT / "results",
            solver_tol=1e-9,
            solver_maxiter=2500,
        )

        grid = YeeGrid.from_config(config)
        solver = SolverConfig(
            nev=config.nbands,
            sigma=config.default_sigma(),
            tol=config.solver_tol,
            maxiter=config.solver_maxiter,
            omega_min=config.omega_min,
        )

        h0, _, _ = assemble_systems(grid, 0.0, 0.0)
        res = solve_hermitian(h0, grid, solver)

        if res.nconv == 0:
            raise RuntimeError(f"No converged modes for N={n}")

        omega0 = float(res.eigenvalues[0])
        freqs.append(omega0)
        logger.info(f"N={n:2d} omega0={omega0:.8f}")

    ref = freqs[-1]
    if ref == 0:
        raise RuntimeError("Invalid zero reference frequency for convergence test.")

    output_dir = ROOT / "results"
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "band_structures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    out = figures_dir / "convergence_test.png"
    plot_convergence(resolutions, freqs, ref, out)

    errors = [abs(f - ref) / abs(ref) for f in freqs]
    np.savetxt(
        data_dir / "convergence_test.dat",
        np.column_stack([resolutions, freqs, errors]),
        header="resolution omega0 relative_error",
    )

    meta = {
        "case": "convergence_test",
        "resolutions": resolutions,
        "reference_resolution": resolutions[-1],
        "reference_frequency": ref,
        "elapsed_s": time.perf_counter() - t0,
    }
    (data_dir / "convergence_test_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    logger.info(f"Saved convergence figure: {out}")


if __name__ == "__main__":
    main()
