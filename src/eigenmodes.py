"""Extracción de modos, energía y normalización."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import MU0, Polarization, TEBlocks, TMBlocks, block_offset
from .grid import YeeGrid


@dataclass
class ModeFields:
    omega: float
    H: Dict[str, np.ndarray]
    E: Dict[str, np.ndarray]
    V: Dict[str, np.ndarray]
    energy: float


def extract_fields(x: np.ndarray, grid: YeeGrid, omega: float) -> ModeFields:
    n2 = grid.n_dof

    if grid.mode == Polarization.TM:
        h = {
            "Hx": x[
                block_offset(TMBlocks.BLK_HX, n2) : block_offset(TMBlocks.BLK_HX, n2)
                + n2
            ],
            "Hy": x[
                block_offset(TMBlocks.BLK_HY, n2) : block_offset(TMBlocks.BLK_HY, n2)
                + n2
            ],
        }
        e = {
            "Ez": x[
                block_offset(TMBlocks.BLK_EZ, n2) : block_offset(TMBlocks.BLK_EZ, n2)
                + n2
            ],
        }
        v = {
            "Vz": x[
                block_offset(TMBlocks.BLK_VZ, n2) : block_offset(TMBlocks.BLK_VZ, n2)
                + n2
            ],
        }
    else:
        h = {
            "Hz": x[
                block_offset(TEBlocks.BLK_HZ, n2) : block_offset(TEBlocks.BLK_HZ, n2)
                + n2
            ],
        }
        e = {
            "Ex": x[
                block_offset(TEBlocks.BLK_EX, n2) : block_offset(TEBlocks.BLK_EX, n2)
                + n2
            ],
            "Ey": x[
                block_offset(TEBlocks.BLK_EY, n2) : block_offset(TEBlocks.BLK_EY, n2)
                + n2
            ],
        }
        v = {
            "Vx": x[
                block_offset(TEBlocks.BLK_VX, n2) : block_offset(TEBlocks.BLK_VX, n2)
                + n2
            ],
            "Vy": x[
                block_offset(TEBlocks.BLK_VY, n2) : block_offset(TEBlocks.BLK_VY, n2)
                + n2
            ],
        }

    energy = compute_energy_density_integral(x, grid)
    return ModeFields(omega=float(omega), H=h, E=e, V=v, energy=energy)


def compute_energy_density_integral(x: np.ndarray, grid: YeeGrid) -> float:
    """Suma discreta de W0.

    Para Drude:
        W0 = 1/2 eps_inf |E|² + 1/2 mu0 |H|²
             + 1/(2 eps_inf omega_p²) |V|²

    El factor Δ² se omite porque cancela en normalizaciones relativas.
    """
    n2 = grid.n_dof
    w = 0.0

    if grid.mode == Polarization.TM:
        hx0 = block_offset(TMBlocks.BLK_HX, n2)
        hy0 = block_offset(TMBlocks.BLK_HY, n2)
        ez0 = block_offset(TMBlocks.BLK_EZ, n2)
        vz0 = block_offset(TMBlocks.BLK_VZ, n2)

        for k in range(n2):
            w += 0.5 * MU0 * (np.abs(x[hx0 + k]) ** 2 + np.abs(x[hy0 + k]) ** 2)
            w += 0.5 * grid.eps_z[k] * np.abs(x[ez0 + k]) ** 2

            wp = grid.omega_p_mask[k]
            if grid.metal_mask[k] and wp > 0:
                w += 0.5 * np.abs(x[vz0 + k]) ** 2 / (grid.eps_inf_mask[k] * wp**2)

    else:
        hz0 = block_offset(TEBlocks.BLK_HZ, n2)
        ex0 = block_offset(TEBlocks.BLK_EX, n2)
        ey0 = block_offset(TEBlocks.BLK_EY, n2)
        vx0 = block_offset(TEBlocks.BLK_VX, n2)
        vy0 = block_offset(TEBlocks.BLK_VY, n2)

        for k in range(n2):
            w += 0.5 * MU0 * np.abs(x[hz0 + k]) ** 2
            w += 0.5 * grid.eps_x[k] * np.abs(x[ex0 + k]) ** 2
            w += 0.5 * grid.eps_y[k] * np.abs(x[ey0 + k]) ** 2

            wp_x = grid.omega_p_x[k]
            if grid.metal_x[k] and wp_x > 0:
                w += 0.5 * np.abs(x[vx0 + k]) ** 2 / (grid.eps_x[k] * wp_x**2)

            wp_y = grid.omega_p_y[k]
            if grid.metal_y[k] and wp_y > 0:
                w += 0.5 * np.abs(x[vy0 + k]) ** 2 / (grid.eps_y[k] * wp_y**2)

    return float(w.real)


def normalize_mode_energy(x: np.ndarray, grid: YeeGrid) -> np.ndarray:
    w = compute_energy_density_integral(x, grid)
    if w <= 0:
        return x
    return x / np.sqrt(w)


def reshape_field(field_1d: np.ndarray, n: int) -> np.ndarray:
    return field_1d.reshape(n, n, order="C")
