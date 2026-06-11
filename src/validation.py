"""Validaciones numéricas: hermiticidad, ortogonalidad, modos planos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse

from .constants import SolverConfig
from .grid import YeeGrid
from .hermitian_solver import solve_hermitian
from .operators import build_CE_TE, build_CH_TE, build_H_hat
from .perturbation import verify_physical_orthogonality
from .utils import hermitian_defect, relative_frobenius_diff


@dataclass
class ValidationReport:
    """Resumen de validaciones."""

    hermitian_defect: float
    curl_adjoint_error: float
    orthogonality_passed: bool
    max_ortho_off_diag: float


def verify_curl_adjoint_te(grid: YeeGrid, kx: float, ky: float) -> float:
    """||C_H − C_E†||_F / ||C_E†||_F para TE."""
    ce = build_CE_TE(grid, kx, ky)
    ch = build_CH_TE(grid, kx, ky)
    ced = ce.conj().transpose().tocsr()
    return relative_frobenius_diff(ch, ced)


def validate_system(
    grid: YeeGrid, kx: float = 0.0, ky: float = 0.0
) -> Tuple[float, float]:
    """Retorna (defecto Hermitiano de Ĥ, error de adjunto del rotacional)."""
    h = build_H_hat(grid, kx, ky)
    h_def = hermitian_defect(h)
    curl_err = (
        verify_curl_adjoint_te(grid, kx, ky)
        if grid.mode.value == "TE"
        else 0.0
    )
    return h_def, curl_err


def run_orthogonality_test(
    grid: YeeGrid, config: SolverConfig, kx: float = 0.0
) -> Tuple[bool, float]:
    """Ejecuta solver y verifica ortogonalidad física."""
    from .operators import build_A_diagonal

    h = build_H_hat(grid, kx, 0.0)
    result = solve_hermitian(h, grid, config)
    a_diag = build_A_diagonal(grid)
    report = verify_physical_orthogonality(result, a_diag)
    return report.passed, report.max_off_diagonal


def vacuum_mode_frequency(k: float) -> float:
    """Modo plano en vacío: ω = c|k| (unidades normalizadas c=1)."""
    return abs(k)
