"""Corrección perturbativa de pérdidas y ortogonalidad modal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .constants import MU0, Polarization, TEBlocks, TMBlocks, block_offset
from .grid import YeeGrid
from .hermitian_solver import ComplexEigenResult, EigenResult


@dataclass
class ModeEnergy:
    E_field: float

    H_field: float

    V_field: float

    total: float

    @property
    def metal_fraction(self) -> float:

        return self.V_field / self.total if self.total > 0 else 0.0


@dataclass
class PerturbResult:
    omega0: float

    omega1: complex

    Q_factor: float

    energy: ModeEnergy


@dataclass
class LossComparison:
    omega_real: float

    im_exact: float

    im_perturbative: float

    relative_error: float


@dataclass
class OrthogonalityReport:
    n_modes: int

    max_off_diagonal: float

    max_diagonal_error: float

    passed: bool


def integral_V_metal(x: np.ndarray, grid: YeeGrid) -> float:
    """Numerador tipo eq. 15: ∫ |V|²/(ε∞ωp²) dr.

    En el código la integral discreta omite Δ² porque cancela contra W0

    en la razón perturbativa.

    """

    n2 = grid.n_dof

    total = 0.0

    if grid.mode == Polarization.TM:
        v0 = block_offset(TMBlocks.BLK_VZ, n2)

        for k in range(n2):
            wp = grid.omega_p_mask[k]

            eps = grid.eps_inf_mask[k]

            if grid.metal_mask[k] and wp > 0:
                total += np.abs(x[v0 + k]) ** 2 / (eps * wp**2)

    else:
        vx0 = block_offset(TEBlocks.BLK_VX, n2)

        vy0 = block_offset(TEBlocks.BLK_VY, n2)

        for k in range(n2):
            wp_x = grid.omega_p_x[k]

            if grid.metal_x[k] and wp_x > 0:
                total += np.abs(x[vx0 + k]) ** 2 / (grid.eps_x[k] * wp_x**2)

            wp_y = grid.omega_p_y[k]

            if grid.metal_y[k] and wp_y > 0:
                total += np.abs(x[vy0 + k]) ** 2 / (grid.eps_y[k] * wp_y**2)

    return float(total)


def integral_W0(x: np.ndarray, grid: YeeGrid) -> float:

    from .eigenmodes import compute_energy_density_integral

    return compute_energy_density_integral(x, grid)


def compute_mode_energy(x: np.ndarray, grid: YeeGrid) -> ModeEnergy:

    n2 = grid.n_dof

    e_field = 0.0

    h_field = 0.0

    v_field = 0.0

    if grid.mode == Polarization.TM:
        hx0 = block_offset(TMBlocks.BLK_HX, n2)

        hy0 = block_offset(TMBlocks.BLK_HY, n2)

        ez0 = block_offset(TMBlocks.BLK_EZ, n2)

        vz0 = block_offset(TMBlocks.BLK_VZ, n2)

        for k in range(n2):
            h_field += 0.5 * MU0 * (np.abs(x[hx0 + k]) ** 2 + np.abs(x[hy0 + k]) ** 2)

            e_field += 0.5 * grid.eps_z[k] * np.abs(x[ez0 + k]) ** 2

            wp = grid.omega_p_mask[k]

            if grid.metal_mask[k] and wp > 0:
                v_field += (
                    0.5 * np.abs(x[vz0 + k]) ** 2 / (grid.eps_inf_mask[k] * wp**2)
                )

    else:
        hz0 = block_offset(TEBlocks.BLK_HZ, n2)

        ex0 = block_offset(TEBlocks.BLK_EX, n2)

        ey0 = block_offset(TEBlocks.BLK_EY, n2)

        vx0 = block_offset(TEBlocks.BLK_VX, n2)

        vy0 = block_offset(TEBlocks.BLK_VY, n2)

        for k in range(n2):
            h_field += 0.5 * MU0 * np.abs(x[hz0 + k]) ** 2

            e_field += 0.5 * grid.eps_x[k] * np.abs(x[ex0 + k]) ** 2

            e_field += 0.5 * grid.eps_y[k] * np.abs(x[ey0 + k]) ** 2

            wp_x = grid.omega_p_x[k]

            if grid.metal_x[k] and wp_x > 0:
                v_field += 0.5 * np.abs(x[vx0 + k]) ** 2 / (grid.eps_x[k] * wp_x**2)

            wp_y = grid.omega_p_y[k]

            if grid.metal_y[k] and wp_y > 0:
                v_field += 0.5 * np.abs(x[vy0 + k]) ** 2 / (grid.eps_y[k] * wp_y**2)

    total = e_field + h_field + v_field

    return ModeEnergy(float(e_field), float(h_field), float(v_field), float(total))


def compute_perturbative_loss(
    omega0: float, x: np.ndarray, grid: YeeGrid, gamma: float
) -> PerturbResult:
    """Corrección de primer orden para damping Drude.

    Con la convención de este código:

        Im(ω) < 0 representa decaimiento.

    """

    num = integral_V_metal(x, grid)

    den = integral_W0(x, grid)

    if den <= 0:
        raise ValueError("Densidad de energía nula en corrección perturbativa.")

    omega1 = -0.5j * gamma * (num / den)

    q = -omega0 / (2.0 * omega1.imag) if omega1.imag < 0 else np.inf

    return PerturbResult(
        omega0=float(omega0),
        omega1=omega1,
        Q_factor=float(q),
        energy=compute_mode_energy(x, grid),
    )


def compute_all_perturbative(
    lossless: EigenResult, grid: YeeGrid, gamma: float
) -> List[PerturbResult]:

    return [
        compute_perturbative_loss(
            lossless.eigenvalues[j],
            lossless.eigenvectors_x[j],
            grid,
            gamma,
        )
        for j in range(lossless.nconv)
    ]


def compare_loss_methods(
    lossless: EigenResult,
    exact: ComplexEigenResult,
    perturbative: List[PerturbResult],
) -> List[LossComparison]:
    """Compara bandas por orden de frecuencia real.

    Para una comparación más robusta se puede hacer matching por overlap modal,

    pero para reproducir Fig. 2 en el caso Γ pequeño basta ordenar por Re(ω).

    """

    n = min(lossless.nconv, exact.nconv, len(perturbative))

    out: list[LossComparison] = []

    for j in range(n):
        im_ex = float(exact.eigenvalues[j].imag)

        im_pt = float(perturbative[j].omega1.imag)

        denom = abs(im_ex) + 1e-30

        out.append(
            LossComparison(
                omega_real=float(lossless.eigenvalues[j]),
                im_exact=im_ex,
                im_perturbative=im_pt,
                relative_error=float(abs(im_ex - im_pt) / denom),
            )
        )

    return out


def verify_physical_orthogonality(
    result: EigenResult, a_diag: np.ndarray, threshold: float = 1e-5
) -> OrthogonalityReport:

    max_off = 0.0

    max_diag = 0.0

    for m in range(result.nconv):
        for n in range(result.nconv):
            xm = result.eigenvectors_x[m]

            xn = result.eigenvectors_x[n]

            g = np.vdot(xm, a_diag * xn)

            if m == n:
                max_diag = max(max_diag, abs(g.real - 1.0) + abs(g.imag))

            else:
                max_off = max(max_off, abs(g))

    return OrthogonalityReport(
        n_modes=result.nconv,
        max_off_diagonal=float(max_off),
        max_diagonal_error=float(max_diag),
        passed=max_off < threshold and max_diag < threshold,
    )
