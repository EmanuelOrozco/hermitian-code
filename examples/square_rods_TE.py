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

from src.constants import GeometryShape, Polarization, SimConfig, SolverConfig
from src.eigenmodes import extract_fields, normalize_mode_energy
from src.grid import YeeGrid
from src.perturbation import compute_mode_energy
from src.reproduction import (
    collect_lossless_modes,
    modes_to_band_row,
    omega_to_paper_units,
    paper_units_to_omega,
    write_json,
)
from src.utils import kpath_gamma_x, setup_logging
from src.visualization import plot_band_structure, plot_field_intensity

import numpy as np

PAPER_SHIFTS = [0.08, 0.12, 0.16, 0.25, 0.50, 0.75, 1.00, 1.12]
MODES_PER_SHIFT = 14
DEDUP_TOL = 2.5e-4


def main() -> None:
    logger = setup_logging()

    config = SimConfig(
        mode=Polarization.TE,
        resolution=20,
        fill_fraction=0.25,  # aquí significa lado s/a, no área
        shape=GeometryShape.SQUARE,
        eps_inf_metal=1.0,
        omega_p_metal=float(paper_units_to_omega(1.0)),
        omega_0_metal=0.0,
        eps_inf_air=1.0,
        omega_p_air=0.0,
        omega_0_air=1.0e12,
        gamma=0.0,
        nk=40,
        nbands=100,
        sigma=paper_units_to_omega(0.35),
        output_dir=ROOT / "results",
        solver_tol=1e-9,
        solver_maxiter=3000,
    )

    grid = YeeGrid.from_config(config)
    logger.info(grid.summary())

    base_solver = SolverConfig(
        nev=MODES_PER_SHIFT,
        sigma=config.default_sigma(),
        tol=config.solver_tol,
        maxiter=config.solver_maxiter,
        omega_min=config.omega_min,
    )

    kx_path = kpath_gamma_x(config.nk)
    k_norm = kx_path / np.pi

    bands = np.full((config.nk, config.nbands), np.nan)
    mode_counts: list[int] = []
    gamma_modes = []

    for ik, kx in enumerate(kx_path):
        modes = collect_lossless_modes(
            grid,
            float(kx),
            0.0,
            PAPER_SHIFTS,
            base_solver,
            MODES_PER_SHIFT,
            ncv=64,
            duplicate_tol_paper=DEDUP_TOL,
        )

        row = modes_to_band_row(modes, config.nbands)
        bands[ik, :] = row
        mode_counts.append(len(modes))
        if ik == 0:
            gamma_modes = modes

        logger.info(
            f"k[{ik:02d}] k/π={k_norm[ik]:.3f} "
            f"nmodes={len(modes)} "
            f"ωa/2πc={', '.join(f'{omega_to_paper_units(w):.4f}' for w in row[:8] if np.isfinite(w))}"
        )

    data_dir = config.output_dir / "band_structures"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        data_dir / "fig1_bands_TE.dat",
        np.column_stack([k_norm, omega_to_paper_units(bands)]),
        header="k_over_pi band_frequencies_in_omega_a_over_2pi_c",
    )

    out_fig = config.output_dir / "figures" / "fig1_square_rods_TE_bands.png"
    bands_for_plot = np.where(omega_to_paper_units(bands) <= 1.15, bands, np.nan)
    plot_band_structure(
        k_norm,
        bands_for_plot,
        out_fig,
        title="Square plasmonic rods — TE",
        convert_to_paper_units=True,
        marker_only=False,
        ylim=(0.0, 1.12),
        paper_style=True,
    )
    logger.info(f"Bandas guardadas en {out_fig}")

    if not gamma_modes:
        raise RuntimeError("No Gamma modes were computed for field profiles.")

    field_candidates = [
        (i, mode)
        for i, mode in enumerate(gamma_modes)
        if 0.10 <= mode.omega_paper <= 1.10
    ]
    if not field_candidates:
        field_candidates = list(enumerate(gamma_modes))

    energies = [
        (i, compute_mode_energy(normalize_mode_energy(mode.vector_x, grid), grid))
        for i, mode in field_candidates
    ]
    low_idx = min(energies, key=lambda item: item[1].metal_fraction)[0]
    high_idx = max(energies, key=lambda item: item[1].metal_fraction)[0]
    selected = [("mode0", low_idx), ("mode4", high_idx)]

    for label, mode_idx in selected:
        mode = gamma_modes[mode_idx]
        x = normalize_mode_energy(mode.vector_x, grid)
        fields = extract_fields(x, grid, mode.omega)

        out = config.output_dir / "field_profiles" / f"fig1_TE_{label}_Gamma.png"
        plot_field_intensity(fields, grid, out, components=["Ex", "Vx"])
        energy = compute_mode_energy(x, grid)
        logger.info(
            f"Campo {label} guardado en {out}; "
            f"ωa/2πc={mode.omega_paper:.4f}, V_energy_fraction={energy.metal_fraction:.3f}"
        )

    finite_bands = omega_to_paper_units(bands[np.isfinite(bands)])
    write_json(
        data_dir / "fig1_bands_TE_meta.json",
        {
            "case": "fig1_square_rods_TE",
            "mode": config.mode.value,
            "resolution": config.resolution,
            "square_side_over_a": config.fill_fraction,
            "nk": config.nk,
            "nbands": config.nbands,
            "paper_shifts": PAPER_SHIFTS,
            "omega_p_internal": config.omega_p_metal,
            "omega_p_paper_units": float(omega_to_paper_units(config.omega_p_metal)),
            "internal_sigma_values": [float(paper_units_to_omega(s)) for s in PAPER_SHIFTS],
            "modes_per_shift": MODES_PER_SHIFT,
            "duplicate_tolerance_paper_units": DEDUP_TOL,
            "mode_count_min": int(min(mode_counts)),
            "mode_count_max": int(max(mode_counts)),
            "frequency_min_paper_units": float(np.nanmin(finite_bands)),
            "frequency_max_paper_units": float(np.nanmax(finite_bands)),
            "field_selection": {
                "fig1_TE_mode0_Gamma": "minimum mechanical V energy fraction at Gamma",
                "fig1_TE_mode4_Gamma": "maximum mechanical V energy fraction at Gamma",
            },
        },
    )


if __name__ == "__main__":
    main()
