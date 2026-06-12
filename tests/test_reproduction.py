import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reproduction import (  # noqa: E402
    SpectralMode,
    deduplicate_modes,
    omega_to_paper_units,
    paper_units_to_omega,
)

REFERENCE = ROOT / "tests" / "reference_data" / "raman_fan_prl_104_087401.json"


def test_paper_unit_round_trip():
    freq = np.array([0.1, 0.5, 1.0])
    assert np.allclose(omega_to_paper_units(paper_units_to_omega(freq)), freq)


def test_deduplicate_modes_preserves_same_shift_degeneracy():
    modes = [
        SpectralMode(
            omega=float(paper_units_to_omega(0.50000)),
            vector_x=np.array([1.0, 0.0]),
            shift_paper=0.5,
            residual=1e-8,
        ),
        SpectralMode(
            omega=float(paper_units_to_omega(0.50001)),
            vector_x=np.array([0.0, 1.0]),
            shift_paper=0.5,
            residual=1e-8,
        ),
        SpectralMode(
            omega=float(paper_units_to_omega(0.50002)),
            vector_x=np.array([1.0, 0.0]),
            shift_paper=0.75,
            residual=1e-9,
        ),
    ]

    out = deduplicate_modes(modes, duplicate_tol_paper=2.5e-4)

    assert len(out) == 2
    vectors = {tuple(mode.vector_x.tolist()) for mode in out}
    assert vectors == {(1.0, 0.0), (0.0, 1.0)}


def test_reference_data_documents_strict_ranges():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))

    assert ref["figure_1"]["square_side_over_a"] == 0.25
    assert ref["figure_1"]["resolution"] == [20, 20]
    assert ref["figure_1"]["y_range_omega_a_over_2pi_c"][1] >= 1.1
    assert ref["figure_2"]["gamma_over_omega_p"] == 0.01
