"""Corrección perturbativa de pérdidas (eq. 15) y ortogonalidad modal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .constants import MU0, Polarization, TEBlocks, TMBlocks, block_offset
from .grid import YeeGrid
from .hermitian_solver import ComplexEigenResult, EigenResult


@dataclass
class ModeEnergy:
    """Descomposición energética de un modo."""

    E_field: float
    H_field: float
    V_field: float
    total: float

    @property
    def metal_fraction(self) -> float:
        return self.V_field / self.total if self.total > 0 else 0.0


@dataclass
class PerturbResult:
    """Corrección perturbativa de primer orden."""

    omega0: float
    omega1: complex
    Q_factor: float
    energy: ModeEnergy


@dataclass
class LossComparison:
    """Comparación exacta vs perturbativa."""

    omega_real: float
    im_exact: float
    im_perturbative: float
    relative_error: float


@dataclass
class OrthogonalityReport:
    """Reporte de ortogonalidad A-pesada (eq. 13)."""

    n_modes: int
    max_off_diagonal: float
    max_diagonal_error: float
    passed: bool


def integral_V_metal(x: np.ndarray, grid: YeeGrid) -> float:
    """Numerador de eq. (15): Σ_metal |V|²/(ε∞ ωp²)."""
    n2 = grid.n_dof
    total = 0.0

    if grid.mode == Polarization.TM:
        for k in range(n2):
            if not grid.metal_mask[k]:
                continue
            eps_i = grid.eps_inf_mask[k]
            wp_i = grid.omega_p_mask[k]
            if wp_i <= 0:
                continue
            idx = block_offset(TMBlocks.BLK_VZ, n2) + k
            total += np.abs(x[idx]) ** 2 / (eps_i * wp_i**2)
    else:
        for k in range(n2):
            if not grid.metal_mask[k]:
                continue
            wp = grid.omega_p_mask[k]
            if wp <= 0:
                continue
            for blk, eps in (
                (TEBlocks.BLK_VX, grid.eps_x[k]),
                (TEBlocks.BLK_VY, grid.eps_y[k]),
            ):
                idx = block_offset(blk, n2) + k
                total += np.abs(x[idx]) ** 2 / (eps * wp**2)
    return total


def integral_W0(x: np.ndarray, grid: YeeGrid) -> float:
    """Denominador de eq. (15): ∫ W₀ dr."""
    from .eigenmodes import compute_energy_density_integral

    return compute_energy_density_integral(x, grid)


def compute_mode_energy(x: np.ndarray, grid: YeeGrid) -> ModeEnergy:
    """Descompone energía en componentes E, H, V."""
    n2 = grid.n_dof
    e_field = h_field = v_field = 0.0

    if grid.mode == Polarization.TM:
        for k in range(n2):
            e_field += 0.5 * grid.eps_z[k] * np.abs(
                x[block_offset(TMBlocks.BLK_EZ, n2) + k]
            ) ** 2
            h_field += 0.5 * MU0 * (
                np.abs(x[block_offset(TMBlocks.BLK_HX, n2) + k]) ** 2
                + np.abs(x[block_offset(TMBlocks.BLK_HY, n2) + k]) ** 2
            )
            if grid.metal_mask[k] and grid.omega_p_mask[k] > 0:
                wp = grid.omega_p_mask[k]
                v_field += (
                    0.5
                    * np.abs(x[block_offset(TMBlocks.BLK_VZ, n2) + k]) ** 2
                    / (grid.eps_z[k] * wp**2)
                )
    else:
        for k in range(n2):
            h_field += 0.5 * MU0 * np.abs(
                x[block_offset(TEBlocks.BLK_HZ, n2) + k]
            ) ** 2
            e_field += 0.5 * grid.eps_x[k] * np.abs(
                x[block_offset(TEBlocks.BLK_EX, n2) + k]
            ) ** 2
            e_field += 0.5 * grid.eps_y[k] * np.abs(
                x[block_offset(TEBlocks.BLK_EY, n2) + k]
            ) ** 2
            if grid.metal_mask[k] and grid.omega_p_mask[k] > 0:
                wp = grid.omega_p_mask[k]
                v_field += (
                    0.5
                    * np.abs(x[block_offset(TEBlocks.BLK_VX, n2) + k]) ** 2
                    / (grid.eps_x[k] * wp**2)
                )
                v_field += (
                    0.5
                    * np.abs(x[block_offset(TEBlocks.BLK_VY, n2) + k]) ** 2
                    / (grid.eps_y[k] * wp**2)
                )

    total = e_field + h_field + v_field
    return ModeEnergy(e_field, h_field, v_field, total)


def compute_perturbative_loss(
    omega0: float, x: np.ndarray, grid: YeeGrid, gamma: float
) -> PerturbResult:
    """ω₁ = −(iΓ/2) × num/den  (eq. 15, convención e^{-iωt})."""
    num = integral_V_metal(x, grid)
    den = integral_W0(x, grid)
    if den <= 0:
        raise ValueError("Densidad de energía nula en corrección perturbativa.")

    omega1 = complex(0.0, -gamma / 2.0) * (num / den)
    q = -omega0 / (2.0 * omega1.imag) if omega1.imag < 0 else np.inf

    return PerturbResult(
        omega0=omega0,
        omega1=omega1,
        Q_factor=float(q),
        energy=compute_mode_energy(x, grid),
    )


def compute_all_perturbative(
    lossless: EigenResult, grid: YeeGrid, gamma: float
) -> List[PerturbResult]:
    """Correcciones perturbativas para todos los modos convergentes."""
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
    """Compara Im[ω] exacto vs perturbativo."""
    n = min(lossless.nconv, exact.nconv, len(perturbative))
    results = []
    for j in range(n):
        im_ex = exact.eigenvalues[j].imag
        im_pt = perturbative[j].omega1.imag
        denom = abs(im_ex) + 1e-30
        results.append(
            LossComparison(
                omega_real=lossless.eigenvalues[j],
                im_exact=im_ex,
                im_perturbative=im_pt,
                relative_error=abs(im_ex - im_pt) / denom,
            )
        )
    return results


def verify_physical_orthogonality(
    result: EigenResult, a_diag: np.ndarray, threshold: float = 1e-5
) -> OrthogonalityReport:
    """Verifica ⟨x_m|A|x_n⟩ = δ_mn (eq. 13)."""
    max_off = 0.0
    max_diag = 0.0
    for m in range(result.nconv):
        for p in range(result.nconv):
            xm = result.eigenvectors_x[m]
            xp = result.eigenvectors_x[p]
            g = np.vdot(xm, a_diag * xp)
            if m == p:
                max_diag = max(max_diag, abs(g.real - 1.0) + abs(g.imag))
            else:
                max_off = max(max_off, abs(g))
    return OrthogonalityReport(
        n_modes=result.nconv,
        max_off_diagonal=max_off,
        max_diagonal_error=max_diag,
        passed=max_off < threshold and max_diag < threshold,
    )
