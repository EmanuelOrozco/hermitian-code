#!/usr/bin/env python3
"""Caso 1: barras cuadradas plasmónicas TE (Fig. 1 del paper).

Parámetros del paper:
  - lado s = 0.25a  → fill_fraction = 0.25
  - ε∞ = 1, ω₀ = 0 (Drude), ωp = 1 en metal
  - resolución 20×20, ruta Γ→X
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import GeometryShape, Polarization, SimConfig, SolverConfig
from src.eigenmodes import extract_fields, normalize_mode_energy
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian
from src.operators import assemble_systems
from src.utils import kpath_gamma_x, setup_logging
from src.visualization import plot_band_structure, plot_field_intensity
import numpy as np


def main() -> None:
    logger = setup_logging()
    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,
        shape=GeometryShape.SQUARE,
        eps_inf_metal=1.0,
        omega_p_metal=1.0,
        omega_0_metal=0.0,
        nk=20,
        nbands=15,
        output_dir=ROOT / "results",
    )
    grid = YeeGrid.from_config(config)
    logger.info(grid.summary())

    solver = SolverConfig(nev=config.nbands, sigma=config.default_sigma())
    kx_path = kpath_gamma_x(config.nk)
    k_norm = kx_path / np.pi
    bands = []

    for kx in kx_path:
        h0, _, _ = assemble_systems(grid, float(kx), 0.0)
        res = solve_hermitian(h0, grid, solver)
        row = np.zeros(config.nbands)
        row[: res.nconv] = res.eigenvalues[: res.nconv]
        bands.append(row)

    bands = np.array(bands)
    out_fig = config.output_dir / "figures" / "square_rods_TE_bands.png"
    plot_band_structure(
        k_norm,
        bands,
        out_fig,
        title="Barras cuadradas plasmónicas — TE (Raman & Fan 2010)",
    )
    logger.info(f"Bandas guardadas en {out_fig}")

    # Perfiles de campo en k=0, modos 0 y 1
    h0, _, _ = assemble_systems(grid, 0.0, 0.0)
    res = solve_hermitian(h0, grid, solver)
    for mode_idx in (0, 1):
        if mode_idx < res.nconv:
            x = normalize_mode_energy(res.eigenvectors_x[mode_idx], grid)
            fields = extract_fields(x, grid, res.eigenvalues[mode_idx])
            plot_field_intensity(
                fields,
                grid,
                config.output_dir / "field_profiles" / f"TE_mode{mode_idx}_Gamma.png",
            )


if __name__ == "__main__":
    main()
