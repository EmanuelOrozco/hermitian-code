"""Punto de entrada CLI — cálculo de bandas fotónicas Hermitianas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .constants import GeometryShape, Polarization, SimConfig, SolverConfig
from .eigenmodes import extract_fields, normalize_mode_energy
from .grid import YeeGrid
from .hermitian_solver import solve_hermitian, solve_nonhermitian
from .operators import assemble_systems, build_A_diagonal
from .perturbation import (
    compare_loss_methods,
    compute_all_perturbative,
    verify_physical_orthogonality,
)
from .utils import banner, kpath_gamma_x, setup_logging, summarize_frequencies
from .visualization import (
    plot_band_structure,
    plot_field_intensity,
    plot_loss_comparison,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Estructura de bandas fotónicas — formulación Hermitiana "
            "(Raman & Fan, PRL 104, 087401)"
        )
    )
    p.add_argument("--mode", choices=["TE", "TM"], default="TE")
    p.add_argument("--resolution", type=int, default=20)
    p.add_argument("--fill-fraction", type=float, default=0.25)
    p.add_argument("--shape", choices=["square", "circle"], default="square")
    p.add_argument("--nk", type=int, default=20)
    p.add_argument("--nbands", type=int, default=15)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--omega-min", type=float, default=1e-8)
    p.add_argument("--perturb", action="store_true", default=False)
    p.add_argument("--verify-ortho", action="store_true")
    p.add_argument("--field-mode", type=int, default=-1)
    p.add_argument("--field-kindex", type=int, default=0)
    p.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(
        mode=Polarization(args.mode),
        resolution=args.resolution,
        fill_fraction=args.fill_fraction,
        shape=GeometryShape(args.shape),
        gamma=args.gamma,
        nk=args.nk,
        nbands=args.nbands,
        sigma=args.sigma,
        omega_min=args.omega_min,
        perturb=args.perturb or args.gamma > 0,
        verify_ortho=args.verify_ortho,
        verbosity=args.verbosity,
        output_dir=args.output_dir,
        field_mode=args.field_mode,
        field_kindex=args.field_kindex,
    )


def build_solver_config(config: SimConfig) -> SolverConfig:
    return SolverConfig(
        nev=config.nbands,
        tol=config.solver_tol,
        maxiter=config.solver_maxiter,
        sigma=config.default_sigma(),
        omega_min=config.omega_min,
    )


def compute_bands(config: SimConfig, logger) -> dict:
    grid = YeeGrid.from_config(config)
    if config.verbosity >= 1:
        logger.info(grid.summary())

    solver_cfg = build_solver_config(config)

    kx_path = kpath_gamma_x(config.nk)
    k_norm = kx_path / np.pi
    all_bands: list[np.ndarray] = []

    t0 = time.perf_counter()
    banner("Calculando bandas fotónicas", logger)

    for ki, kx in enumerate(kx_path):
        h0, _, _ = assemble_systems(grid, float(kx), 0.0, gamma=0.0)

        if config.verbosity >= 2:
            from .validation import validate_system

            h_def, curl_err = validate_system(grid, float(kx), 0.0)
            logger.info(f"k[{ki}] Herm defect={h_def:.2e}, curl adj={curl_err:.2e}")

        result = solve_hermitian(h0, grid, solver_cfg)

        if config.verify_ortho:
            a_diag = build_A_diagonal(grid)
            ortho = verify_physical_orthogonality(result, a_diag)
            logger.info(
                f"Ortogonalidad k[{ki}]: off={ortho.max_off_diagonal:.2e} "
                f"diag={ortho.max_diagonal_error:.2e} "
                f"[{'PASS' if ortho.passed else 'FAIL'}]"
            )

        if config.verbosity >= 1:
            logger.info(
                f"k[{ki}] k/pi={k_norm[ki]:.3f} "
                f"omega={summarize_frequencies(result.eigenvalues)}"
            )

        row = np.full(config.nbands, np.nan)
        n = min(config.nbands, result.nconv)
        row[:n] = result.eigenvalues[:n]
        all_bands.append(row)

    bands = np.array(all_bands)
    elapsed = time.perf_counter() - t0
    logger.info(f"Bandas completadas en {elapsed:.2f} s")

    out_dir = config.output_dir / "band_structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        out_dir / "bands.dat",
        np.column_stack([k_norm, bands]),
        header="k_over_pi omega_1 omega_2 ... omega_n",
    )

    fig_path = config.output_dir / "figures" / "band_structure.png"
    plot_band_structure(k_norm, bands, fig_path)

    meta = {
        "config": {
            "mode": config.mode.value,
            "resolution": config.resolution,
            "fill_fraction": config.fill_fraction,
            "shape": config.shape.value,
            "nk": config.nk,
            "nbands": config.nbands,
            "sigma": solver_cfg.sigma,
            "omega_min": solver_cfg.omega_min,
        },
        "elapsed_s": elapsed,
    }
    (out_dir / "bands_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return {
        "grid": grid,
        "bands": bands,
        "k_norm": k_norm,
        "solver_cfg": solver_cfg,
    }


def compute_loss(
    config: SimConfig,
    grid: YeeGrid,
    solver_cfg: SolverConfig,
    logger,
) -> None:
    banner("Cálculo de pérdidas modales", logger)

    ki = config.nk // 2
    kx = float(kpath_gamma_x(config.nk)[ki])

    h0, h_lossy, _ = assemble_systems(grid, kx, 0.0, gamma=config.gamma)
    lossless = solve_hermitian(h0, grid, solver_cfg)
    exact = solve_nonhermitian(h_lossy, grid, solver_cfg)

    perturb = compute_all_perturbative(lossless, grid, config.gamma)
    cmp = compare_loss_methods(lossless, exact, perturb)

    out = config.output_dir / "band_structures" / "loss_comparison.dat"
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.array(
        [[c.omega_real, c.im_exact, c.im_perturbative, c.relative_error] for c in cmp]
    )
    np.savetxt(out, data, header="omega_real im_exact im_perturb relative_error")

    plot_loss_comparison(cmp, config.output_dir / "figures" / "loss_comparison.png")
    logger.info(f"Comparación de pérdidas -> {out}")


def export_field(
    config: SimConfig,
    grid: YeeGrid,
    solver_cfg: SolverConfig,
    mode_idx: int,
    k_index: int,
    logger,
) -> None:
    kx = float(kpath_gamma_x(config.nk)[k_index])
    h0, _, _ = assemble_systems(grid, kx, 0.0)
    result = solve_hermitian(h0, grid, solver_cfg)

    if mode_idx >= result.nconv:
        logger.warning(f"Modo {mode_idx} no convergió (nconv={result.nconv})")
        return

    x = normalize_mode_energy(result.eigenvectors_x[mode_idx], grid)
    fields = extract_fields(x, grid, result.eigenvalues[mode_idx])

    out = config.output_dir / "field_profiles" / f"mode_{mode_idx}_k{k_index}.png"
    plot_field_intensity(fields, grid, out)
    logger.info(f"Perfil de campo -> {out}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    log_file = config.output_dir / "logs" / "run.log"
    logger = setup_logging(
        level={0: 40, 1: 20, 2: 10}[config.verbosity],
        log_file=log_file,
    )

    try:
        result = compute_bands(config, logger)
        grid = result["grid"]
        solver_cfg = result["solver_cfg"]

        if config.gamma > 0:
            compute_loss(config, grid, solver_cfg, logger)

        if config.field_mode >= 0:
            export_field(
                config,
                grid,
                solver_cfg,
                config.field_mode,
                config.field_kindex,
                logger,
            )

        return 0
    except Exception as exc:
        logger.exception("Error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
