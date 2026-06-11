"""Tests de operadores discretos."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import Polarization, SimConfig
from src.grid import YeeGrid
from src.operators import (
    build_CE_TE,
    build_CH_TE,
    build_H_hat,
    build_A_diagonal,
)
from src.utils import hermitian_defect, relative_frobenius_diff


@pytest.fixture
def te_grid():
    return YeeGrid.from_config(
        SimConfig(mode=Polarization.TE, resolution=8, fill_fraction=0.25)
    )


def test_curl_adjoint_te(te_grid):
    ce = build_CE_TE(te_grid, 0.3, 0.0)
    ch = build_CH_TE(te_grid, 0.3, 0.0)
    err = relative_frobenius_diff(ch, ce.conj().T)
    assert err < 1e-12


def test_H_hermitian(te_grid):
    h = build_H_hat(te_grid, 0.5, 0.0)
    defect = hermitian_defect(h)
    assert defect < 1e-10


def test_A_positive(te_grid):
    a = build_A_diagonal(te_grid)
    assert np.all(a > 0)
