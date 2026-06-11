"""Tests del modelo dispersivo."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dispersive_materials import DispersiveMaterial, LorentzPole


def test_drude_epsilon():
    mat = DispersiveMaterial.drude(eps_inf=1.0, omega_p=1.0, gamma=0.0)
    eps = mat.epsilon(0.5)
    expected = 1.0 + 1.0 / (0.0 - 0.25 + 0j)  # Lorentz factor × eps_inf
    assert abs(eps - expected) < 1e-10


def test_vacuum_no_pole():
    mat = DispersiveMaterial.vacuum()
    assert not mat.has_active_pole()
    assert mat.epsilon(1.0) == 1.0


def test_multi_pole():
    mat = DispersiveMaterial(
        eps_inf=2.0,
        poles=[
            LorentzPole(omega_p=1.0, omega_0=0.5),
            LorentzPole(omega_p=0.5, omega_0=0.0),
        ],
    )
    eps = mat.epsilon(0.3)
    assert np.isfinite(eps.real)
