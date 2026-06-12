"""Helpers for reproducing Raman and Fan, PRL 104, 087401.

The numerical operator uses angular frequency in normalized units ``c/a``.
The paper plots frequencies as ``omega a / 2 pi c``.  Keeping this conversion
explicit avoids selecting the wrong ARPACK shift-invert window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import json
import numpy as np

from .constants import SolverConfig
from .grid import YeeGrid
from .hermitian_solver import EigenResult, solve_hermitian, solve_nonhermitian
from .operators import assemble_systems
from .perturbation import LossComparison, compute_all_perturbative

TWO_PI = 2.0 * np.pi


@dataclass
class SpectralMode:
    omega: float
    vector_x: np.ndarray
    shift_paper: float
    residual: float

    @property
    def omega_paper(self) -> float:
        return omega_to_paper_units(self.omega)


def paper_units_to_omega(freq: float | np.ndarray) -> float | np.ndarray:
    """Convert paper units ``omega a / 2 pi c`` to internal ``omega``."""
    return TWO_PI * freq


def omega_to_paper_units(omega: float | np.ndarray) -> float | np.ndarray:
    """Convert internal ``omega`` to paper units ``omega a / 2 pi c``."""
    return omega / TWO_PI


def solver_for_shift(
    shift_paper: float,
    base: SolverConfig,
    nev: int,
    ncv: int | None = None,
) -> SolverConfig:
    return SolverConfig(
        nev=nev,
        tol=base.tol,
        maxiter=base.maxiter,
        sigma=float(paper_units_to_omega(shift_paper)),
        ncv=ncv if ncv is not None else base.ncv,
        which=base.which,
        hermitian=base.hermitian,
        omega_min=base.omega_min,
    )


def collect_lossless_modes(
    grid: YeeGrid,
    kx: float,
    ky: float,
    shifts_paper: Sequence[float],
    base_solver: SolverConfig,
    modes_per_shift: int,
    ncv: int | None = None,
    duplicate_tol_paper: float = 2.5e-4,
) -> list[SpectralMode]:
    h0, _, _ = assemble_systems(grid, kx, ky)
    raw: list[SpectralMode] = []

    for shift in shifts_paper:
        result = _solve_hermitian_shift(
            h0, grid, shift, base_solver, modes_per_shift, ncv=ncv
        )
        raw.extend(_modes_from_result(result, shift))

    return deduplicate_modes(raw, duplicate_tol_paper)


def collect_loss_comparison(
    grid: YeeGrid,
    kx: float,
    ky: float,
    gamma: float,
    shifts_paper: Sequence[float],
    base_solver: SolverConfig,
    modes_per_shift: int,
    ncv: int | None = None,
    duplicate_tol_paper: float = 2.5e-4,
    exact_match_tol_paper: float = 2.0e-3,
) -> list[LossComparison]:
    lossless_modes = collect_lossless_modes(
        grid,
        kx,
        ky,
        shifts_paper,
        base_solver,
        modes_per_shift,
        ncv=ncv,
        duplicate_tol_paper=duplicate_tol_paper,
    )

    _, h_lossy, _ = assemble_systems(grid, kx, ky, gamma=gamma)
    exact_raw: list[complex] = []
    for shift in shifts_paper:
        exact = _solve_nonhermitian_shift(
            h_lossy, grid, shift, base_solver, modes_per_shift, ncv=ncv
        )
        exact_raw.extend(complex(v) for v in exact.eigenvalues)

    exact_vals = deduplicate_complex_values(exact_raw, duplicate_tol_paper)
    perturbative = [
        compute_all_perturbative(
            EigenResult(
                eigenvalues=np.array([mode.omega]),
                eigenvectors_y=[],
                eigenvectors_x=[mode.vector_x],
                nconv=1,
                sigma=paper_units_to_omega(mode.shift_paper),
            ),
            grid,
            gamma,
        )[0]
        for mode in lossless_modes
    ]

    comparisons: list[LossComparison] = []
    used_exact: set[int] = set()
    for mode, perturb in zip(lossless_modes, perturbative):
        idx = _nearest_unused_exact(mode.omega, exact_vals, used_exact)
        if idx is None:
            continue
        exact_val = exact_vals[idx]
        if abs(omega_to_paper_units(exact_val.real - mode.omega)) > exact_match_tol_paper:
            continue
        used_exact.add(idx)
        im_exact = float(exact_val.imag)
        im_pert = float(perturb.omega1.imag)
        denom = abs(im_exact) + 1e-30
        comparisons.append(
            LossComparison(
                omega_real=float(mode.omega),
                im_exact=im_exact,
                im_perturbative=im_pert,
                relative_error=float(abs(im_exact - im_pert) / denom),
            )
        )

    return sorted(comparisons, key=lambda c: c.omega_real)


def deduplicate_modes(
    modes: Iterable[SpectralMode],
    duplicate_tol_paper: float,
) -> list[SpectralMode]:
    """Remove the same ARPACK mode returned by overlapping shifts.

    Near-degenerate modes from the same shift window are preserved.  If two
    overlapping shifts return the same frequency, keep the lower-residual copy.
    """
    kept: list[SpectralMode] = []

    for mode in sorted(modes, key=lambda m: (m.omega, m.residual)):
        dup_idx = None
        for i, existing in enumerate(kept):
            same_freq = (
                abs(mode.omega_paper - existing.omega_paper) <= duplicate_tol_paper
            )
            same_shift = mode.shift_paper == existing.shift_paper
            same_mode = _modal_overlap(mode.vector_x, existing.vector_x) > 0.98
            if same_freq and same_mode and not same_shift:
                dup_idx = i
                break

        if dup_idx is None:
            kept.append(mode)
        elif mode.residual < kept[dup_idx].residual:
            kept[dup_idx] = mode

    return sorted(kept, key=lambda m: m.omega)


def deduplicate_complex_values(
    values: Iterable[complex],
    duplicate_tol_paper: float,
) -> list[complex]:
    kept: list[complex] = []

    for value in sorted(values, key=lambda z: z.real):
        if value.real <= 0:
            continue
        is_duplicate = any(
            abs(omega_to_paper_units(value.real - existing.real))
            <= duplicate_tol_paper
            for existing in kept
        )
        if not is_duplicate:
            kept.append(value)

    return kept


def modes_to_band_row(modes: Sequence[SpectralMode], nbands: int) -> np.ndarray:
    row = np.full(nbands, np.nan, dtype=float)
    n = min(nbands, len(modes))
    if n:
        row[:n] = [modes[i].omega for i in range(n)]
    return row


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _modes_from_result(result: EigenResult, shift_paper: float) -> list[SpectralMode]:
    return [
        SpectralMode(
            omega=float(result.eigenvalues[i]),
            vector_x=result.eigenvectors_x[i],
            shift_paper=float(shift_paper),
            residual=float(result.residuals[i]) if i < len(result.residuals) else np.inf,
        )
        for i in range(result.nconv)
    ]


def _solve_hermitian_shift(
    h0,
    grid: YeeGrid,
    shift_paper: float,
    base_solver: SolverConfig,
    modes_per_shift: int,
    ncv: int | None,
) -> EigenResult:
    last_error: Exception | None = None
    for offset in (0.0, 1.0e-5, -1.0e-5, 5.0e-5, -5.0e-5):
        solver = solver_for_shift(shift_paper + offset, base_solver, modes_per_shift, ncv=ncv)
        try:
            return solve_hermitian(h0, grid, solver)
        except RuntimeError as exc:
            if "singular" not in str(exc).lower():
                raise
            last_error = exc
    raise RuntimeError(f"All shifts near {shift_paper} failed.") from last_error


def _solve_nonhermitian_shift(
    h_lossy,
    grid: YeeGrid,
    shift_paper: float,
    base_solver: SolverConfig,
    modes_per_shift: int,
    ncv: int | None,
):
    last_error: Exception | None = None
    for offset in (0.0, 1.0e-5, -1.0e-5, 5.0e-5, -5.0e-5):
        solver = solver_for_shift(shift_paper + offset, base_solver, modes_per_shift, ncv=ncv)
        try:
            return solve_nonhermitian(h_lossy, grid, solver)
        except RuntimeError as exc:
            if "singular" not in str(exc).lower():
                raise
            last_error = exc
    raise RuntimeError(f"All lossy shifts near {shift_paper} failed.") from last_error


def _modal_overlap(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(abs(np.vdot(a, b)) / norm)


def _nearest_unused_exact(
    omega: float,
    exact_vals: Sequence[complex],
    used: set[int],
) -> int | None:
    best_idx = None
    best_dist = np.inf
    for i, value in enumerate(exact_vals):
        if i in used:
            continue
        dist = abs(value.real - omega)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx
