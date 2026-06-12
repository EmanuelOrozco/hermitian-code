"""Malla de Yee 2D, geometría y máscaras de material."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .constants import (
    LATTICE_A,
    GeometryShape,
    Polarization,
    SimConfig,
    TEBlocks,
    TMBlocks,
)


@dataclass
class YeeGrid:
    """Malla Yee 2D con máscaras de material.

    Convención de indexación:
        field[i, j] -> field.ravel(order="C")[i * N + j]

    Para TE se usan parámetros materiales efectivos en las ubicaciones de Ex y Ey.
    """

    config: SimConfig
    N: int
    delta: float
    n_dof: int

    metal_mask: np.ndarray

    eps_z: np.ndarray
    eps_x: np.ndarray
    eps_y: np.ndarray

    eps_inf_mask: np.ndarray
    omega_p_mask: np.ndarray
    omega_0_mask: np.ndarray
    gamma_mask: np.ndarray

    metal_x: np.ndarray
    metal_y: np.ndarray
    omega_p_x: np.ndarray
    omega_p_y: np.ndarray
    gamma_x: np.ndarray
    gamma_y: np.ndarray

    @classmethod
    def from_config(cls, config: SimConfig) -> "YeeGrid":
        n = config.resolution
        delta = config.delta

        metal_2d = cls._build_metal_mask_2d(n, config)
        metal = metal_2d.ravel(order="C")

        eps_inf_2d = np.where(metal_2d, config.eps_inf_metal, config.eps_inf_air)
        omega_p_2d = np.where(metal_2d, config.omega_p_metal, config.omega_p_air)
        omega_0_2d = np.where(metal_2d, config.omega_0_metal, config.omega_0_air)
        gamma_2d = np.where(metal_2d, config.gamma, 0.0)

        eps_z = eps_inf_2d.ravel(order="C")

        eps_x_2d = cls._harmonic_avg_x(eps_inf_2d)
        eps_y_2d = cls._harmonic_avg_y(eps_inf_2d)

        # Para reproducir el caso Drude del paper, los grados V solo deben existir
        # en el metal. En una Yee grid estricta esto debería hacerse por arista.
        # Aquí se proyecta a aristas mediante OR de las celdas vecinas, robusto en
        # interfaces metal-aire.
        metal_x_2d = metal_2d | np.roll(metal_2d, -1, axis=0)
        metal_y_2d = metal_2d | np.roll(metal_2d, -1, axis=1)

        omega_p_x_2d = np.where(metal_x_2d, config.omega_p_metal, 0.0)
        omega_p_y_2d = np.where(metal_y_2d, config.omega_p_metal, 0.0)
        gamma_x_2d = np.where(metal_x_2d, config.gamma, 0.0)
        gamma_y_2d = np.where(metal_y_2d, config.gamma, 0.0)

        return cls(
            config=config,
            N=n,
            delta=delta,
            n_dof=n * n,
            metal_mask=metal,
            eps_z=eps_z,
            eps_x=eps_x_2d.ravel(order="C"),
            eps_y=eps_y_2d.ravel(order="C"),
            eps_inf_mask=eps_inf_2d.ravel(order="C"),
            omega_p_mask=omega_p_2d.ravel(order="C"),
            omega_0_mask=omega_0_2d.ravel(order="C"),
            gamma_mask=gamma_2d.ravel(order="C"),
            metal_x=metal_x_2d.ravel(order="C"),
            metal_y=metal_y_2d.ravel(order="C"),
            omega_p_x=omega_p_x_2d.ravel(order="C"),
            omega_p_y=omega_p_y_2d.ravel(order="C"),
            gamma_x=gamma_x_2d.ravel(order="C"),
            gamma_y=gamma_y_2d.ravel(order="C"),
        )

    @staticmethod
    def _build_metal_mask_2d(n: int, config: SimConfig) -> np.ndarray:
        """Máscara booleana de la inclusión metálica.

        Para cuadrados:
            config.fill_fraction = s/a.
        Para círculos:
            config.fill_fraction = área metálica / área celda.
        """
        mask = np.zeros((n, n), dtype=bool)
        center = 0.5 * LATTICE_A

        if config.shape == GeometryShape.SQUARE:
            side = config.fill_fraction * LATTICE_A
            half = 0.5 * side
            radius = None
        else:
            radius = np.sqrt(config.fill_fraction / np.pi) * LATTICE_A
            half = None

        for i in range(n):
            x = (i + 0.5) * config.delta
            for j in range(n):
                y = (j + 0.5) * config.delta

                if config.shape == GeometryShape.SQUARE:
                    inside = abs(x - center) <= half and abs(y - center) <= half
                else:
                    inside = (x - center) ** 2 + (y - center) ** 2 <= radius**2

                mask[i, j] = inside

        return mask

    @staticmethod
    def _harmonic_avg_x(eps_node: np.ndarray) -> np.ndarray:
        """Promedio armónico de ε∞ para Ex."""
        right = np.roll(eps_node, -1, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = 2.0 / (1.0 / eps_node + 1.0 / right)
        avg[~np.isfinite(avg)] = 1.0
        return avg

    @staticmethod
    def _harmonic_avg_y(eps_node: np.ndarray) -> np.ndarray:
        """Promedio armónico de ε∞ para Ey."""
        up = np.roll(eps_node, -1, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = 2.0 / (1.0 / eps_node + 1.0 / up)
        avg[~np.isfinite(avg)] = 1.0
        return avg

    @property
    def mode(self) -> Polarization:
        return self.config.mode

    @property
    def total_dof(self) -> int:
        nblk = (
            TMBlocks.N_BLOCKS_DRUDE
            if self.mode == Polarization.TM
            else TEBlocks.N_BLOCKS_DRUDE
        )
        return nblk * self.n_dof

    def node_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        i_idx, j_idx = np.meshgrid(np.arange(self.N), np.arange(self.N), indexing="ij")
        x = (i_idx + 0.5) * self.delta
        y = (j_idx + 0.5) * self.delta
        return x.ravel(order="C"), y.ravel(order="C")

    def summary(self) -> str:
        n_metal = int(np.sum(self.metal_mask))
        return (
            f"YeeGrid N={self.N}, Δ={self.delta:.4f}, "
            f"mode={self.mode.value}, metal_nodes={n_metal}/{self.n_dof}, "
            f"metal_fraction≈{n_metal / self.n_dof:.4f}, "
            f"total_dof={self.total_dof}"
        )


def bloch_phase_x(kx: float) -> complex:
    return np.exp(1j * kx * LATTICE_A)


def bloch_phase_y(ky: float) -> complex:
    return np.exp(1j * ky * LATTICE_A)
