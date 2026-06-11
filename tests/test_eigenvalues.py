"""Tests del solver de autovalores."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import Polarization, SimConfig, SolverConfig
from src.grid import YeeGrid
from src.hermitian_solver import solve_hermitian
from src.operators import build_H_hat


def test_eigenvalues_real_and_positive():
    grid = YeeGrid.from_config(
        SimConfig(mode=Polarization.TE, resolution=8, nbands=5)
    )
    h = build_H_hat(grid, 0.0, 0.0)
    cfg = SolverConfig(nev=5, sigma=0.3)
    res = solve_hermitian(h, grid, cfg)
    assert res.nconv >= 1
    assert np.allclose(res.eigenvalues.imag, 0, atol=1e-6)
    assert np.all(res.eigenvalues > 0)


def test_residuals_small():
    grid = YeeGrid.from_config(SimConfig(resolution=8, nbands=3))
    h = build_H_hat(grid, 0.3, 0.0)
    cfg = SolverConfig(nev=3, sigma=0.3)
    res = solve_hermitian(h, grid, cfg)
    assert np.all(res.residuals < 1e-4)
