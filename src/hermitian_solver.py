"""Solución numérica del problema de autovalores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, eigsh

from .constants import SolverConfig
from .grid import YeeGrid
from .operators import build_sqrtA_inv


@dataclass
class EigenResult:
    eigenvalues: np.ndarray
    eigenvectors_y: List[np.ndarray]
    eigenvectors_x: List[np.ndarray]
    nconv: int
    sigma: float
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ComplexEigenResult:
    eigenvalues: np.ndarray
    eigenvectors_y: List[np.ndarray]
    eigenvectors_x: List[np.ndarray]
    nconv: int
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))


def _back_transform(vectors_y: List[np.ndarray], grid: YeeGrid) -> List[np.ndarray]:
    """x = A^{-1/2} y."""
    s_inv = build_sqrtA_inv(grid)
    return [v * s_inv for v in vectors_y]


def _sort_by_real_frequency(
    vals: np.ndarray, vecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(vals.real)
    return vals[order], vecs[:, order]


def _filter_physical_modes(
    vals: np.ndarray,
    vecs: np.ndarray,
    omega_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Descarta modos negativos y nulos espurios."""
    mask = vals.real > omega_min
    return vals[mask], vecs[:, mask]


def _normalize_y_columns(vecs: np.ndarray) -> np.ndarray:
    out = vecs.copy()
    for j in range(out.shape[1]):
        norm = np.linalg.norm(out[:, j])
        if norm > 0:
            out[:, j] /= norm
    return out


def solve_hermitian(
    h_hat: sparse.csr_matrix,
    grid: YeeGrid,
    config: SolverConfig,
) -> EigenResult:
    """Resuelve ω y = Ĥ y con eigsh."""
    n = h_hat.shape[0]
    nev = min(config.nev, n - 2)
    ncv = config.resolve_ncv(n)

    try:
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
    except ArpackNoConvergence as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors
        if vals is None or vecs is None or len(vals) == 0:
            raise

    vals, vecs = _sort_by_real_frequency(vals, vecs)
    vals, vecs = _filter_physical_modes(vals, vecs, config.omega_min)

    if vecs.size:
        vecs = _normalize_y_columns(vecs)

    nconv = len(vals)
    vectors_y = [vecs[:, j].copy() for j in range(nconv)]
    vectors_x = _back_transform(vectors_y, grid)

    residuals = np.array(
        [
            np.linalg.norm(h_hat @ vectors_y[j] - vals[j] * vectors_y[j])
            / (np.linalg.norm(vectors_y[j]) + 1e-30)
            for j in range(nconv)
        ],
        dtype=float,
    )

    return EigenResult(
        eigenvalues=vals.real.astype(float),
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
    """Resuelve el sistema no-Hermitiano con pérdidas."""
    n = h_nh.shape[0]
    nev = min(config.nev, n - 2)
    ncv = config.resolve_ncv(n)

    try:
        vals, vecs = eigs(
            h_nh,
            k=nev,
            sigma=config.sigma,
            which="LM",
            tol=config.tol,
            maxiter=config.maxiter,
            ncv=ncv,
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors
        if vals is None or vecs is None or len(vals) == 0:
            raise

    vals, vecs = _sort_by_real_frequency(vals, vecs)
    vals, vecs = _filter_physical_modes(vals, vecs, config.omega_min)

    if vecs.size:
        vecs = _normalize_y_columns(vecs)

    nconv = len(vals)
    vectors_y = [vecs[:, j].copy() for j in range(nconv)]
    vectors_x = _back_transform(vectors_y, grid)

    residuals = np.array(
        [
            np.linalg.norm(h_nh @ vectors_y[j] - vals[j] * vectors_y[j])
            / (np.linalg.norm(vectors_y[j]) + 1e-30)
            for j in range(nconv)
        ],
        dtype=float,
    )

    return ComplexEigenResult(
        eigenvalues=vals,
        eigenvectors_y=vectors_y,
        eigenvectors_x=vectors_x,
        nconv=nconv,
        residuals=residuals,
    )


def make_shift_invert_op(h_hat: sparse.csr_matrix, sigma: float) -> LinearOperator:
    n = h_hat.shape[0]
    shifted = (h_hat - sigma * sparse.eye(n, dtype=np.complex128)).tocsc()
    lu = sparse.linalg.splu(shifted)

    def matvec(x: np.ndarray) -> np.ndarray:
        return lu.solve(x)

    return LinearOperator((n, n), matvec=matvec, dtype=np.complex128)
