#!/usr/bin/env python3
"""Figura 1: barras cuadradas plasmónicas TE.

Parámetros del paper:
  - lado del cuadrado s = 0.25a
  - ε∞ = 1
  - ω0 = 0
  - ωp = 1
  - resolución 20×20
  - ruta Γ→X
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.constants import GeometryShape, Polarization, SimConfig, SolverConfig
from src.eigenmodes import extract_fields, normalize_mode_energy
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian
from src.operators import assemble_systems
from src.utils import kpath_gamma_x, setup_logging
from src.visualization import plot_band_structure, plot_field_intensity


def main() -> None:
    logger = setup_logging()

    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,  # aquí significa lado s/a, no área
        shape=GeometryShape.SQUARE,
        eps_inf_metal=1.0,
        omega_p_metal=1.0,
        omega_0_metal=0.0,
        eps_inf_air=1.0,
        omega_p_air=0.0,
        omega_0_air=1.0e12,
        gamma=0.0,
        nk=40,
        nbands=35,
        sigma=0.35,
        output_dir=ROOT / "results",
        solver_tol=1e-9,
        solver_maxiter=2000,
    )

    grid = YeeGrid.from_config(config)
    logger.info(grid.summary())

    solver = SolverConfig(
        nev=config.nbands,
        sigma=config.default_sigma(),
        tol=config.solver_tol,
        maxiter=config.solver_maxiter,
        omega_min=config.omega_min,
    )

    kx_path = kpath_gamma_x(config.nk)
    k_norm = kx_path / np.pi

    bands = np.full((config.nk, config.nbands), np.nan)

    for ik, kx in enumerate(kx_path):
        h0, _, _ = assemble_systems(grid, float(kx), 0.0)
        res = solve_hermitian(h0, grid, solver)

        n = min(config.nbands, res.nconv)
        bands[ik, :n] = res.eigenvalues[:n]

        logger.info(
            f"k[{ik:02d}] k/π={k_norm[ik]:.3f} "
            f"nconv={res.nconv} "
            f"ωa/2πc={', '.join(f'{w / (2 * np.pi):.4f}' for w in res.eigenvalues[:5])}"
        )

    out_fig = config.output_dir / "figures" / "fig1_square_rods_TE_bands.png"
    plot_band_structure(
        k_norm,
        bands,
        out_fig,
        title="Square plasmonic rods — TE",
        convert_to_paper_units=True,
    )
    logger.info(f"Bandas guardadas en {out_fig}")

    # Paneles tipo Fig. 1(b,c): perfiles en Γ.
    # Los índices exactos pueden necesitar ajuste visual porque ARPACK devuelve
    # modos cercanos a sigma, no necesariamente en el mismo orden gráfico del paper.
    h0, _, _ = assemble_systems(grid, 0.0, 0.0)
    res = solve_hermitian(h0, grid, solver)

    modes_to_plot = [0, min(4, res.nconv - 1)]

    for mode_idx in modes_to_plot:
        if mode_idx < 0 or mode_idx >= res.nconv:
            continue

        x = normalize_mode_energy(res.eigenvectors_x[mode_idx], grid)
        fields = extract_fields(x, grid, res.eigenvalues[mode_idx])

        out = config.output_dir / "field_profiles" / f"fig1_TE_mode{mode_idx}_Gamma.png"
        plot_field_intensity(fields, grid, out, components=["Ex", "Vx"])
        logger.info(f"Campo modo {mode_idx} guardado en {out}")


if __name__ == "__main__":
    main()
