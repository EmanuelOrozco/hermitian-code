#!/usr/bin/env python3
"""Runner reproducible para generar resultados.

Perfiles:
- smoke: rápido para CI en cada push.
- paper: más pesado; reproduce resultados principales tipo Fig. 1 y Fig. 2.
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
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def run_smoke() -> None:
    """Genera resultados pequeños para verificar que el pipeline funciona."""
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
    """Genera resultados principales de reproducción del paper."""
    run([sys.executable, "examples/square_rods_TE.py"])
    run([sys.executable, "examples/lossy_case.py"])
    run([sys.executable, "examples/perturbation_comparison.py"])


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
        choices=["smoke", "paper"],
        default="smoke",
        help="Tipo de reproducción a ejecutar.",
    )
    args = parser.parse_args()

    ensure_dirs()

    t0 = time.perf_counter()

    if args.profile == "smoke":
        run_smoke()
    else:
        run_paper()

    elapsed = time.perf_counter() - t0
    write_manifest(args.profile, elapsed)

    print(f"\nDone. Profile={args.profile}, elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
