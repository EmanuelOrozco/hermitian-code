"""Tests de hermiticidad del Hamiltoniano."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import Polarization, SimConfig
from src.grid import YeeGrid
from src.operators import build_H_hat, build_B
from src.utils import hermitian_defect


@pytest.mark.parametrize("mode", [Polarization.TE, Polarization.TM])
@pytest.mark.parametrize("kx", [0.0, 0.5, 1.0])
def test_hermitian_H(mode, kx):
    grid = YeeGrid.from_config(
        SimConfig(mode=mode, resolution=6, fill_fraction=0.25)
    )
    h = build_H_hat(grid, kx * np.pi, 0.0)
    assert hermitian_defect(h) < 1e-9


def test_B_hermitian_coupling_structure():
    """B debe ser Hermitiana: B = B†."""
    grid = YeeGrid.from_config(SimConfig(resolution=6))
    b = build_B(grid, 0.2, 0.0)
    diff = b - b.conj().T
    if diff.nnz == 0:
        assert True
    else:
        assert np.max(np.abs(diff.data)) < 1e-10
