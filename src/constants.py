"""Constantes físicas, enumeraciones y configuración centralizada.

Convención usada para reproducir Raman & Fan, PRL 104, 087401 (2010):

- Unidades normalizadas: a = 1, c = 1, μ0 = ε0 = 1.
- Para el ejemplo de Fig. 1:
    side_length = 0.25 a
    eps_inf_metal = 1
    omega_p_metal = 1
    omega_0_metal = 0
- El parámetro histórico `fill_fraction` se conserva por compatibilidad, pero
  para `GeometryShape.SQUARE` se interpreta como lado normalizado s/a, no como
  fracción de área. Esto corrige el caso del paper: s = 0.25a.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class Polarization(str, Enum):
    """Polarización 2D soportada.

    TE: Hz, Ex, Ey, Vx, Vy.
    TM: Hx, Hy, Ez, Vz.
    """

    TE = "TE"
    TM = "TM"


class GeometryShape(str, Enum):
    """Geometría de la inclusión metálica."""

    SQUARE = "square"
    CIRCLE = "circle"


PI: float = 3.141592653589793
MU0: float = 1.0
EPS0: float = 1.0
C_LIGHT: float = 1.0
LATTICE_A: float = 1.0


class TMBlocks:
    """Bloques del vector de estado TM-Drude: x = (Hx, Hy, Ez, Vz)."""

    N_BLOCKS_DRUDE = 4
    BLK_HX = 0
    BLK_HY = 1
    BLK_EZ = 2
    BLK_VZ = 3


class TEBlocks:
    """Bloques del vector de estado TE-Drude: x = (Hz, Ex, Ey, Vx, Vy)."""

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
    """Índice plano para arrays con shape (N, N) y flatten order='C'.

    La convención única del código es:

        arr[i, j] -> arr.ravel(order="C")[i * n + j]

    donde i recorre x y j recorre y.
    """
    return i * n + j


@dataclass
class SimConfig:
    """Parámetros de simulación.

    Notas:
    - `fill_fraction` se conserva por compatibilidad con los ejemplos previos.
      Para cuadrados representa el lado normalizado s/a.
      Para círculos representa la fracción de área.
    """

    mode: Polarization = Polarization.TE
    resolution: int = 20
    fill_fraction: float = 0.25
    shape: GeometryShape = GeometryShape.SQUARE

    eps_inf_metal: float = 1.0
    omega_p_metal: float = 1.0
    omega_0_metal: float = 0.0

    eps_inf_air: float = 1.0
    omega_p_air: float = 0.0
    omega_0_air: float = 1.0e12

    gamma: float = 0.0
    nk: int = 20
    nbands: int = 15
    sigma: Optional[float] = None
    omega_min: float = 1.0e-8

    perturb: bool = True
    verify_ortho: bool = False
    verbosity: int = 1

    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "results"
    )

    solver_tol: float = 1e-8
    solver_maxiter: int = 1000
    ncv_factor: int = 3

    field_mode: int = -1
    field_kindex: int = 0

    @property
    def delta(self) -> float:
        """Paso espacial Δ = a/N."""
        return LATTICE_A / self.resolution

    def default_sigma(self) -> float:
        """Shift-invert recomendado para las bandas plasmónicas de Fig. 1.

        En el paper aparecen muchas bandas planas alrededor de frecuencias de
        resonancia superficial, por eso usar sigma≈0.3ωp es más útil que buscar
        simplemente los eigenvalores más cercanos a cero.
        """
        if self.sigma is not None and self.sigma > 0:
            return self.sigma
        return 0.3 * self.omega_p_metal


@dataclass
class SolverConfig:
    """Configuración del solver de autovalores."""

    nev: int = 15
    tol: float = 1e-8
    maxiter: int = 1000
    sigma: float = 0.3
    ncv: Optional[int] = None
    which: Literal["LM", "LR", "LA", "SA", "SM"] = "LM"
    hermitian: bool = True
    omega_min: float = 1.0e-8

    def resolve_ncv(self, n: int) -> int:
        """Número de vectores de Krylov."""
        if self.ncv is not None:
            return min(n - 1, max(self.ncv, self.nev + 2))
        return min(n - 1, max(3 * self.nev + 1, 30))
