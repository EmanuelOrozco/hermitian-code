#!/usr/bin/env python3
"""Reproducible runner for numerical results.

Profiles
--------
smoke:
    Fast CI sanity check.

paper:
    Main paper-like numerical results:
    - TE band structure.
    - Field profiles.
    - Loss comparison.
    - Perturbation comparison.

publication:
    Full publication bundle:
    - Everything from paper.
    - Spatial convergence.
    - Output validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REFERENCE = ROOT / "tests" / "reference_data" / "raman_fan_prl_104_087401.json"

PUBLICATION_REQUIRED_FILES = [
    RESULTS / "figures" / "fig1_square_rods_TE_bands.png",
    RESULTS / "figures" / "fig2_loss_comparison.png",
    RESULTS / "figures" / "perturbation_vs_gamma.png",
    RESULTS / "figures" / "convergence_test.png",
    RESULTS / "field_profiles" / "fig1_TE_mode0_Gamma.png",
    RESULTS / "field_profiles" / "fig1_TE_mode4_Gamma.png",
    RESULTS / "band_structures" / "fig1_bands_TE.dat",
    RESULTS / "band_structures" / "fig2_loss_comparison.dat",
    RESULTS / "band_structures" / "perturbation_vs_gamma.dat",
    RESULTS / "band_structures" / "convergence_test.dat",
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_dirs() -> None:
    for rel in [
        "results",
        "results/figures",
        "results/band_structures",
        "results/field_profiles",
        "results/logs",
        "results/publication",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def run_smoke() -> None:
    run(
        [
            sys.executable,
            "-m",
            "src.main",
            "--mode",
            "TE",
            "--resolution",
            "8",
            "--nk",
            "3",
            "--nbands",
            "3",
            "--sigma",
            "0.3",
            "--output-dir",
            str(RESULTS),
        ]
    )


def run_paper() -> None:
    run([sys.executable, "examples/square_rods_TE.py"])
    run([sys.executable, "examples/lossy_case.py"])
    run([sys.executable, "examples/perturbation_comparison.py"])


def run_publication() -> None:
    run_paper()
    run([sys.executable, "examples/convergence_test.py"])
    validate_publication_outputs()


def validate_publication_outputs() -> None:
    missing = [p for p in PUBLICATION_REQUIRED_FILES if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p.relative_to(ROOT)}" for p in missing)
        raise FileNotFoundError(
            "Publication outputs are incomplete. Missing files:\n" + msg
        )
    validate_reference_ranges()


def validate_reference_ranges() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    fig1 = np.loadtxt(RESULTS / "band_structures" / "fig1_bands_TE.dat")
    if fig1.ndim != 2 or fig1.shape[1] < 2:
        raise ValueError("fig1_bands_TE.dat must contain k plus band columns.")

    k = fig1[:, 0]
    freqs = fig1[:, 1:]
    finite_freqs = freqs[np.isfinite(freqs)]
    if finite_freqs.size == 0:
        raise ValueError("fig1_bands_TE.dat contains no finite band frequencies.")

    x_min, x_max = reference["figure_1"]["x_range_k_over_pi"]
    y_min, y_max = reference["figure_1"]["y_range_omega_a_over_2pi_c"]
    if np.nanmin(k) > x_min + 1e-8 or np.nanmax(k) < x_max - 1e-8:
        raise ValueError("Fig. 1 k-path does not cover Gamma-X.")
    if np.nanmin(finite_freqs) < y_min - 1e-8:
        raise ValueError("Fig. 1 includes negative paper-unit frequencies.")
    if np.nanmax(finite_freqs) < y_max:
        raise ValueError(
            "Fig. 1 does not cover the high-frequency range shown in the paper."
        )

    for lo, hi in reference["figure_1"]["required_frequency_windows"]:
        if not np.any((finite_freqs >= lo) & (finite_freqs <= hi)):
            raise ValueError(f"Fig. 1 has no modes in required window [{lo}, {hi}].")

    fig2 = np.loadtxt(RESULTS / "band_structures" / "fig2_loss_comparison.dat")
    if fig2.ndim != 2 or fig2.shape[1] < 4:
        raise ValueError("fig2_loss_comparison.dat has an invalid shape.")

    omega = fig2[:, 0]
    loss = np.abs(fig2[:, 1])
    rel_err = fig2[:, 3]
    required_span = reference["figure_2"]["required_frequency_span_min"]
    loss_floor = reference["figure_2"]["relative_error_loss_floor"]
    max_rel_err = reference["figure_2"]["max_relative_error_exact_vs_perturbative"]
    if np.nanmax(omega) - np.nanmin(omega) < required_span:
        raise ValueError("Fig. 2 frequency coverage is too narrow.")
    significant = loss >= loss_floor
    if not np.any(significant):
        raise ValueError("Fig. 2 has no significant-loss points to compare.")
    if np.nanmax(rel_err[significant]) > max_rel_err:
        raise ValueError(
            f"Fig. 2 perturbative error exceeds {max_rel_err:.3f}: "
            f"{np.nanmax(rel_err[significant]):.3f}"
        )


def write_manifest(profile: str, elapsed_s: float) -> None:
    files = sorted(
        str(p.relative_to(ROOT))
        for p in RESULTS.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )

    manifest = {
        "profile": profile,
        "elapsed_s": elapsed_s,
        "num_files": len(files),
        "files": files,
    }

    out = RESULTS / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest written to {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["smoke", "paper", "publication"],
        default="smoke",
    )
    args = parser.parse_args()

    ensure_dirs()

    t0 = time.perf_counter()

    if args.profile == "smoke":
        run_smoke()
    elif args.profile == "paper":
        run_paper()
    else:
        run_publication()

    elapsed = time.perf_counter() - t0
    write_manifest(args.profile, elapsed)

    print(f"\nDone. profile={args.profile}, elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
