"""Malla de Yee 2D, geometría y máscaras de material."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse

from .constants import (
    GeometryShape,
    LATTICE_A,
    MU0,
    Polarization,
    SimConfig,
    TEBlocks,
    TMBlocks,
    flat_index,
)
from .dispersive_materials import MaterialMap


@dataclass
class YeeGrid:
    """Malla Yee 2D con máscaras de material y promedios en aristas."""

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

    @classmethod
    def from_config(cls, config: SimConfig) -> YeeGrid:
        """Construye malla a partir de configuración."""
        n = config.resolution
        delta = config.delta
        metal = cls._build_metal_mask(n, config)
        mats = cls._build_material_maps(n, metal, config)
        return cls(
            config=config,
            N=n,
            delta=delta,
            n_dof=n * n,
            metal_mask=metal,
            eps_z=mats.eps_inf.copy(),
            eps_x=cls._harmonic_avg_x(n, mats.eps_inf),
            eps_y=cls._harmonic_avg_y(n, mats.eps_inf),
            eps_inf_mask=mats.eps_inf,
            omega_p_mask=mats.omega_p,
            omega_0_mask=mats.omega_0,
            gamma_mask=mats.gamma,
        )

    @staticmethod
    def _build_metal_mask(n: int, config: SimConfig) -> np.ndarray:
        """Máscara booleana en nodos primarios (i,j)."""
        mask = np.zeros((n, n), dtype=bool)
        side = np.sqrt(config.fill_fraction) * LATTICE_A
        half = 0.5 * side
        center = 0.5 * LATTICE_A

        for j in range(n):
            y = (j + 0.5) * config.delta
            for i in range(n):
                x = (i + 0.5) * config.delta
                if config.shape == GeometryShape.SQUARE:
                    inside = (
                        abs(x - center) < half and abs(y - center) < half
                    )
                else:
                    inside = (x - center) ** 2 + (y - center) ** 2 < half**2
                mask[i, j] = inside
        return mask.ravel()

    @staticmethod
    def _build_material_maps(
        n: int, metal: np.ndarray, config: SimConfig
    ) -> MaterialMap:
        """Asigna parámetros Drude/Lorentz en nodos primarios."""
        eps = np.where(metal, config.eps_inf_metal, config.eps_inf_air)
        wp = np.where(metal, config.omega_p_metal, config.omega_p_air)
        w0 = np.where(metal, config.omega_0_metal, config.omega_0_air)
        gamma = np.where(metal, config.gamma, 0.0)
        return MaterialMap(
            metal_mask=metal,
            eps_inf=eps,
            omega_p=wp,
            omega_0=w0,
            gamma=gamma,
        )

    @staticmethod
    def _harmonic_avg_x(n: int, eps_node: np.ndarray) -> np.ndarray:
        """Promedio armónico de ε∞ en aristas Ex (entre (i,j) y (i+1,j))."""
        eps2d = eps_node.reshape(n, n)
        left = eps2d
        right = np.roll(eps2d, -1, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = 2.0 / (1.0 / left + 1.0 / right)
        avg[~np.isfinite(avg)] = 1.0
        return avg.ravel()

    @staticmethod
    def _harmonic_avg_y(n: int, eps_node: np.ndarray) -> np.ndarray:
        """Promedio armónico de ε∞ en aristas Ey (entre (i,j) y (i,j+1))."""
        eps2d = eps_node.reshape(n, n)
        low = eps2d
        high = np.roll(eps2d, -1, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = 2.0 / (1.0 / low + 1.0 / high)
        avg[~np.isfinite(avg)] = 1.0
        return avg.ravel()

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
        """Coordenadas de nodos primarios."""
        i_idx, j_idx = np.meshgrid(
            np.arange(self.N), np.arange(self.N), indexing="ij"
        )
        x = (i_idx + 0.5) * self.delta
        y = (j_idx + 0.5) * self.delta
        return x.ravel(), y.ravel()

    def summary(self) -> str:
        """Resumen de la malla."""
        n_metal = int(np.sum(self.metal_mask))
        return (
            f"YeeGrid N={self.N}, Δ={self.delta:.4f}, "
            f"mode={self.mode.value}, metal_nodes={n_metal}/{self.n_dof}, "
            f"total_dof={self.total_dof}"
        )


def bloch_phase_x(kx: float) -> complex:
    """Fase de Bloch e^{ikx a} con a=1."""
    return np.exp(1j * kx * LATTICE_A)


def bloch_phase_y(ky: float) -> complex:
    """Fase de Bloch e^{iky a}."""
    return np.exp(1j * ky * LATTICE_A)
