"""Modelos dispersivos: Lorentz, Drude y múltiples polos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class LorentzPole:
    """Un polo de Lorentz en el modelo dieléctrico."""

    omega_p: float
    omega_0: float = 0.0
    gamma: float = 0.0

    @property
    def is_drude(self) -> bool:
        return self.omega_0 == 0.0

    def epsilon(self, omega: complex) -> complex:
        """ε(ω) = ε∞ [1 + ωp²/(ω₀² − ω² + iγω)] — factor de Lorentz solo."""
        denom = self.omega_0**2 - omega**2 + 1j * self.gamma * omega
        if abs(denom) < 1e-30:
            return complex(np.inf, 0)
        return self.omega_p**2 / denom


@dataclass
class DispersiveMaterial:
    """Material con ε∞ y uno o más polos de Lorentz."""

    eps_inf: float = 1.0
    poles: List[LorentzPole] = field(default_factory=list)

    @classmethod
    def drude(cls, eps_inf: float = 1.0, omega_p: float = 1.0, gamma: float = 0.0) -> DispersiveMaterial:
        """Modelo de Drude (ω₀ = 0)."""
        return cls(eps_inf=eps_inf, poles=[LorentzPole(omega_p=omega_p, omega_0=0.0, gamma=gamma)])

    @classmethod
    def vacuum(cls, eps_inf: float = 1.0) -> DispersiveMaterial:
        """Aire/vacío: sin polos activos (ωp → 0, ω₀ → ∞)."""
        return cls(eps_inf=eps_inf, poles=[])

    def epsilon(self, omega: complex) -> complex:
        """Permitividad completa ε(ω) = ε∞ + ε∞ Σ f_n."""
        result = complex(self.eps_inf, 0.0)
        for pole in self.poles:
            result += self.eps_inf * pole.epsilon(omega)
        return result

    def a_diagonal_v(self, pole_index: int = 0) -> float:
        """Entrada A para bloque V del polo n: ω₀²/(ε∞ ωp²) o 1/(ε∞ ωp²) si Drude."""
        if pole_index >= len(self.poles):
            return 1.0
        pole = self.poles[pole_index]
        if pole.omega_p <= 0:
            return 1.0
        if pole.is_drude:
            return 1.0 / (self.eps_inf * pole.omega_p**2)
        return pole.omega_0**2 / (self.eps_inf * pole.omega_p**2)

    def has_active_pole(self) -> bool:
        return any(p.omega_p > 0 for p in self.poles)


@dataclass
class MaterialMap:
    """Mapas de material en la malla primaria N×N."""

    metal_mask: np.ndarray
    eps_inf: np.ndarray
    omega_p: np.ndarray
    omega_0: np.ndarray
    gamma: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.metal_mask.size
