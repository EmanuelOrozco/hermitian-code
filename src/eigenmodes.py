"""Extracción de modos, energía y normalización."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import MU0, Polarization, TEBlocks, TMBlocks, block_offset
from .grid import YeeGrid


@dataclass
class ModeFields:
    """Componentes de campo de un modo en la malla."""

    omega: float
    H: Dict[str, np.ndarray]
    E: Dict[str, np.ndarray]
    V: Dict[str, np.ndarray]
    energy: float


def extract_fields(x: np.ndarray, grid: YeeGrid, omega: float) -> ModeFields:
    """Descompone el vector de estado x en bloques H, E, V."""
    n2 = grid.n_dof
    n = grid.N

    if grid.mode == Polarization.TM:
        h = {
            "Hx": x[block_offset(TMBlocks.BLK_HX, n2) : block_offset(TMBlocks.BLK_HX, n2) + n2],
            "Hy": x[block_offset(TMBlocks.BLK_HY, n2) : block_offset(TMBlocks.BLK_HY, n2) + n2],
        }
        e = {
            "Ez": x[block_offset(TMBlocks.BLK_EZ, n2) : block_offset(TMBlocks.BLK_EZ, n2) + n2],
        }
        v = {
            "Vz": x[block_offset(TMBlocks.BLK_VZ, n2) : block_offset(TMBlocks.BLK_VZ, n2) + n2],
        }
    else:
        h = {
            "Hz": x[block_offset(TEBlocks.BLK_HZ, n2) : block_offset(TEBlocks.BLK_HZ, n2) + n2],
        }
        e = {
            "Ex": x[block_offset(TEBlocks.BLK_EX, n2) : block_offset(TEBlocks.BLK_EX, n2) + n2],
            "Ey": x[block_offset(TEBlocks.BLK_EY, n2) : block_offset(TEBlocks.BLK_EY, n2) + n2],
        }
        v = {
            "Vx": x[block_offset(TEBlocks.BLK_VX, n2) : block_offset(TEBlocks.BLK_VX, n2) + n2],
            "Vy": x[block_offset(TEBlocks.BLK_VY, n2) : block_offset(TEBlocks.BLK_VY, n2) + n2],
        }

    energy = compute_energy_density_integral(x, grid)
    return ModeFields(omega=omega, H=h, E=e, V=v, energy=energy)


def compute_energy_density_integral(x: np.ndarray, grid: YeeGrid) -> float:
    """∫ W₀ dr (eq. 14) — suma discreta sin factor Δ² (cancela en ratios)."""
    n2 = grid.n_dof
    w = 0.0

    if grid.mode == Polarization.TM:
        for k in range(n2):
            w += 0.5 * grid.eps_z[k] * np.abs(x[block_offset(TMBlocks.BLK_EZ, n2) + k]) ** 2
            w += 0.5 * MU0 * (
                np.abs(x[block_offset(TMBlocks.BLK_HX, n2) + k]) ** 2
                + np.abs(x[block_offset(TMBlocks.BLK_HY, n2) + k]) ** 2
            )
            if grid.metal_mask[k] and grid.omega_p_mask[k] > 0:
                wp = grid.omega_p_mask[k]
                w += (
                    0.5
                    * np.abs(x[block_offset(TMBlocks.BLK_VZ, n2) + k]) ** 2
                    / (grid.eps_z[k] * wp**2)
                )
    else:
        for k in range(n2):
            w += 0.5 * MU0 * np.abs(x[block_offset(TEBlocks.BLK_HZ, n2) + k]) ** 2
            w += 0.5 * grid.eps_x[k] * np.abs(x[block_offset(TEBlocks.BLK_EX, n2) + k]) ** 2
            w += 0.5 * grid.eps_y[k] * np.abs(x[block_offset(TEBlocks.BLK_EY, n2) + k]) ** 2
            if grid.metal_mask[k] and grid.omega_p_mask[k] > 0:
                wp = grid.omega_p_mask[k]
                w += (
                    0.5
                    * np.abs(x[block_offset(TEBlocks.BLK_VX, n2) + k]) ** 2
                    / (grid.eps_x[k] * wp**2)
                )
                w += (
                    0.5
                    * np.abs(x[block_offset(TEBlocks.BLK_VY, n2) + k]) ** 2
                    / (grid.eps_y[k] * wp**2)
                )
    return float(w)


def normalize_mode_energy(x: np.ndarray, grid: YeeGrid) -> np.ndarray:
    """Normaliza modo para ∫ W₀ dr = 1 (como en el paper)."""
    w = compute_energy_density_integral(x, grid)
    if w <= 0:
        return x
    return x / np.sqrt(w)


def reshape_field(field_1d: np.ndarray, n: int) -> np.ndarray:
    """Reordena vector 1D a malla 2D (i,j)."""
    return field_1d.reshape(n, n)
