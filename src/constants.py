"""Constantes físicas, enumeraciones y configuración centralizada."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class Polarization(str, Enum):
    """Polarización 2D soportada."""

    TE = "TE"
    TM = "TM"


class GeometryShape(str, Enum):
    """Geometría de la inclusión metálica."""

    SQUARE = "square"
    CIRCLE = "circle"


# Unidades normalizadas: a = 1 (constante de red), c = 1, mu0 = 1.
PI: float = 3.141592653589793
MU0: float = 1.0
EPS0: float = 1.0
C_LIGHT: float = 1.0
LATTICE_A: float = 1.0


# Bloques del vector de estado TM-Drude: x = (Hx, Hy, Ez, Vz).
class TMBlocks:
    N_BLOCKS_DRUDE = 4
    BLK_HX = 0
    BLK_HY = 1
    BLK_EZ = 2
    BLK_VZ = 3


# Bloques del vector de estado TE-Drude: x = (Hz, Ex, Ey, Vx, Vy).
class TEBlocks:
    N_BLOCKS_DRUDE = 5
    BLK_HZ = 0
    BLK_EX = 1
    BLK_EY = 2
    BLK_VX = 3
    BLK_VY = 4


def block_offset(block_id: int, n_dof: int) -> int:
    """Desplazamiento de fila/columna para un bloque de campo."""
    return block_id * n_dof


def flat_index(i: int, j: int, n: int) -> int:
    """Indexación fila-mayor en malla N×N."""
    return i + j * n


@dataclass
class SimConfig:
    """Parámetros de simulación (equivalente a SimParams del proyecto C++)."""

    mode: Polarization = Polarization.TE
    resolution: int = 20
    fill_fraction: float = 0.25
    shape: GeometryShape = GeometryShape.SQUARE
    eps_inf_metal: float = 1.0
    omega_p_metal: float = 1.0
    omega_0_metal: float = 0.0
    eps_inf_air: float = 1.0
    omega_p_air: float = 0.0
    omega_0_air: float = 1e12
    gamma: float = 0.0
    nk: int = 20
    nbands: int = 15
    sigma: Optional[float] = None
    perturb: bool = True
    verify_ortho: bool = False
    verbosity: int = 1
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "results"
    )
    solver_tol: float = 1e-8
    solver_maxiter: int = 500
    ncv_factor: int = 2
    field_mode: int = -1
    field_kindex: int = 0

    @property
    def delta(self) -> float:
        """Paso espacial Δ = a/N."""
        return LATTICE_A / self.resolution

    def default_sigma(self) -> float:
        """Desplazamiento shift-invert (σ ≈ 0.3 ωp para bandas plasmónicas)."""
        if self.sigma is not None and self.sigma > 0:
            return self.sigma
        # σ pequeño (0.01) con which='LM' selecciona modos nulos en scipy;
        # σ ~ 0.3 apunta al rango de bandas planas del paper (Fig. 1).
        return 0.3 * self.omega_p_metal


@dataclass
class SolverConfig:
    """Configuración del solver de autovalores."""

    nev: int = 15
    tol: float = 1e-8
    maxiter: int = 500
    sigma: float = 0.01
    ncv: Optional[int] = None
    which: Literal["LM", "LR", "LA", "SA", "SM"] = "LM"
    hermitian: bool = True

    def resolve_ncv(self, n: int) -> int:
        """Número de vectores de Krylov (ncv > nev, típicamente 2*nev)."""
        if self.ncv is not None:
            return max(self.ncv, self.nev + 2)
        return min(n - 1, max(2 * self.nev + 1, 20))
