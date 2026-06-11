#!/usr/bin/env python3
"""Caso 3: convergencia espacial de la frecuencia fundamental."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.constants import Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian
from src.operators import assemble_systems
from src.utils import setup_logging
from src.visualization import plot_convergence


def main() -> None:
    logger = setup_logging()
    resolutions = [10, 15, 20, 25, 30]
    freqs = []
    ref = None

    for n in resolutions:
        config = SimConfig(
            mode=Polarization.TE,
            resolution=n,
            fill_fraction=0.25,
            nk=1,
            nbands=5,
            output_dir=ROOT / "results",
        )
        grid = YeeGrid.from_config(config)
        solver = SolverConfig(nev=5, sigma=config.default_sigma())
        h0, _, _ = assemble_systems(grid, 0.0, 0.0)
        res = solve_hermitian(h0, grid, solver)
        omega0 = res.eigenvalues[0] if res.nconv > 0 else 0.0
        freqs.append(omega0)
        logger.info(f"N={n:2d}  ω₀={omega0:.6f}")
        if n == resolutions[-1]:
            ref = omega0

    if ref is None or ref == 0:
        logger.error("No se obtuvo referencia de convergencia.")
        return

    out = ROOT / "results" / "figures" / "convergence_test.png"
    plot_convergence(resolutions, freqs, ref, out)
    logger.info(f"Gráfico de convergencia → {out}")


if __name__ == "__main__":
    main()
