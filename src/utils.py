"""Utilidades: logging, rutas k, helpers numéricos."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .constants import PI, LATTICE_A, SimConfig


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configura logging del framework."""
    logger = logging.getLogger("codigo_hermitian")
    logger.handlers.clear()
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def kpath_gamma_x(nk: int, a: float = LATTICE_A) -> np.ndarray:
    """Ruta Γ→X: kx ∈ [0, π/a], ky = 0."""
    return np.linspace(0.0, PI / a, nk)


def kpath_gamma_x_m_gamma(nk_segment: int, a: float = LATTICE_A) -> Tuple[np.ndarray, np.ndarray]:
    """Ruta Γ→X→M→Γ en la primera zona de Brillouin 2D."""
    kmax = PI / a
    seg1 = np.column_stack(
        [np.linspace(0, kmax, nk_segment), np.zeros(nk_segment)]
    )
    seg2 = np.column_stack(
        [
            np.full(nk_segment, kmax),
            np.linspace(0, kmax, nk_segment),
        ]
    )
    seg3 = np.column_stack(
        [
            np.linspace(kmax, 0, nk_segment),
            np.linspace(kmax, 0, nk_segment),
        ]
    )
    path = np.vstack([seg1, seg2[1:], seg3[1:]])
    labels = ["Γ", "X", "M", "Γ"]
    return path, labels


def ensure_dir(path: Path) -> Path:
    """Crea directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_frobenius_diff(a: np.ndarray, b: np.ndarray) -> float:
    """||A − B||_F / ||B||_F para matrices sparse o densas."""
    from scipy import sparse

    if sparse.issparse(a):
        diff = a - b
        if sparse.issparse(diff):
            num = float(np.sum(np.abs(diff.data) ** 2))
        else:
            num = float(np.sum(np.abs(diff) ** 2))
    else:
        num = float(np.sum(np.abs(a - b) ** 2))
    if sparse.issparse(b):
        den = float(np.sum(np.abs(b.data) ** 2))
    else:
        den = float(np.sum(np.abs(b) ** 2))
    return float(np.sqrt(num) / (np.sqrt(den) + 1e-30))


def hermitian_defect(matrix) -> float:
    """Máximo |H_ij − conj(H_ji)| sobre entradas almacenadas."""
    from scipy import sparse

    if not sparse.issparse(matrix):
        return float(np.max(np.abs(matrix - matrix.conj().T)))
    diff = matrix - matrix.conj().T
    return float(np.max(np.abs(diff.data))) if diff.nnz else 0.0


def dot_herm(a: np.ndarray, b: np.ndarray) -> complex:
    """Producto interno Hermitiano ⟨a|b⟩."""
    return np.vdot(a, b)


def banner(msg: str, logger: logging.Logger | None = None) -> None:
    """Imprime separador de sección."""
    line = f"── {msg} " + "─" * max(0, 58 - len(msg))
    if logger:
        logger.info(line)
    else:
        print(line)


def summarize_frequencies(freqs: Iterable[float], n: int = 5) -> str:
    """Resumen compacto de frecuencias."""
    arr = np.asarray(list(freqs))[:n]
    return ", ".join(f"{w:.5f}" for w in arr)
