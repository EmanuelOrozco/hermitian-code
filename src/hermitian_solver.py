"""Solución numérica del problema de autovalores generalizado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

from .constants import SolverConfig
from .grid import YeeGrid
from .operators import build_sqrtA_inv


@dataclass
class EigenResult:
    """Resultado del solver Hermitiano (frecuencias reales)."""

    eigenvalues: np.ndarray
    eigenvectors_y: List[np.ndarray]
    eigenvectors_x: List[np.ndarray]
    nconv: int
    sigma: float
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ComplexEigenResult:
    """Resultado del solver no-Hermitiano (frecuencias complejas)."""

    eigenvalues: np.ndarray
    eigenvectors_y: List[np.ndarray]
    eigenvectors_x: List[np.ndarray]
    nconv: int
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


def _back_transform(
    vectors_y: List[np.ndarray], grid: YeeGrid
) -> List[np.ndarray]:
    """x = A^{-1/2} y (campos físicos)."""
    s_inv = build_sqrtA_inv(grid)
    return [v * s_inv for v in vectors_y]


def _sort_by_frequency(
    vals: np.ndarray, vecs: np.ndarray, real_only: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Ordena modos por frecuencia creciente (parte real)."""
    if real_only:
        order = np.argsort(vals.real)
    else:
        order = np.argsort(vals.real)
    return vals[order], vecs[:, order]


def _filter_physical_modes(
    vals: np.ndarray,
    vecs: np.ndarray,
    omega_min: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Descarta modos espurios del espacio nulo (ω ≈ 0)."""
    mask = vals.real > omega_min
    if not np.any(mask):
        return vals, vecs
    return vals[mask], vecs[:, mask]


def solve_hermitian(
    h_hat: sparse.csr_matrix,
    grid: YeeGrid,
    config: SolverConfig,
) -> EigenResult:
    """Resuelve ω ŷ = Ĥ ŷ con shift-invert (Lanczos / eigsh).

    Parameters
    ----------
    h_hat : sparse matrix
        Hamiltoniano Hermitiano Ĥ.
    grid : YeeGrid
        Malla para back-transformación.
    config : SolverConfig
        Parámetros del solver.

    Returns
    -------
    EigenResult
        Autovalores ω y autovectores físicos x.
    """
    n = h_hat.shape[0]
    nev = min(config.nev, n - 2)
    ncv = config.resolve_ncv(n)

    vals, vecs = eigsh(
        h_hat,
        k=nev,
        sigma=config.sigma,
        which=config.which,
        tol=config.tol,
        maxiter=config.maxiter,
        ncv=ncv,
        return_eigenvectors=True,
    )

    vals, vecs = _sort_by_frequency(vals, vecs, real_only=True)
    vals, vecs = _filter_physical_modes(vals, vecs)

    nconv = len(vals)
    vectors_y = [vecs[:, j].copy() for j in range(nconv)]
    vectors_x = _back_transform(vectors_y, grid)

    residuals = np.array(
        [
            np.linalg.norm(h_hat @ vectors_y[j] - vals[j] * vectors_y[j])
            / (np.linalg.norm(vectors_y[j]) + 1e-30)
            for j in range(nconv)
        ]
    )

    return EigenResult(
        eigenvalues=vals.real,
        eigenvectors_y=vectors_y,
        eigenvectors_x=vectors_x,
        nconv=nconv,
        sigma=config.sigma,
        residuals=residuals,
    )


def solve_nonhermitian(
    h_nh: sparse.csr_matrix,
    grid: YeeGrid,
    config: SolverConfig,
) -> ComplexEigenResult:
    """Resuelve el sistema con pérdidas (autovalores complejos)."""
    n = h_nh.shape[0]
    nev = min(config.nev, n - 2)
    ncv = config.resolve_ncv(n)

    vals, vecs = eigs(
        h_nh,
        k=nev,
        sigma=config.sigma,
        which="LR",
        tol=config.tol,
        maxiter=config.maxiter,
        ncv=ncv,
        return_eigenvectors=True,
    )

    vals, vecs = _sort_by_frequency(vals, vecs, real_only=False)
    mask = vals.real > 0
    vals, vecs = vals[mask], vecs[:, mask]

    nconv = len(vals)
    vectors_y = [vecs[:, j].copy() for j in range(nconv)]
    vectors_x = _back_transform(vectors_y, grid)

    residuals = np.array(
        [
            np.linalg.norm(h_nh @ vectors_y[j] - vals[j] * vectors_y[j])
            / (np.linalg.norm(vectors_y[j]) + 1e-30)
            for j in range(nconv)
        ]
    )

    return ComplexEigenResult(
        eigenvalues=vals,
        eigenvectors_y=vectors_y,
        eigenvectors_x=vectors_x,
        nconv=nconv,
        residuals=residuals,
    )


def make_shift_invert_op(
    h_hat: sparse.csr_matrix, sigma: float
) -> LinearOperator:
    """Operador (Ĥ − σI)⁻¹ para uso con métodos iterativos."""
    n = h_hat.shape[0]
    shifted = (h_hat - sigma * sparse.eye(n, dtype=np.complex128)).tocsc()
    lu = sparse.linalg.splu(shifted)

    def matvec(x: np.ndarray) -> np.ndarray:
        return lu.solve(x)

    return LinearOperator((n, n), matvec=matvec, dtype=np.complex128)
