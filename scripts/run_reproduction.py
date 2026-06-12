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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

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
