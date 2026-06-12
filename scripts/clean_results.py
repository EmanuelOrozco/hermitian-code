#!/usr/bin/env python3
"""Clean generated numerical and publication outputs."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)

    for subdir in (
        "figures",
        "band_structures",
        "field_profiles",
        "logs",
        "publication",
    ):
        (RESULTS / subdir).mkdir(parents=True, exist_ok=True)

    (RESULTS / ".gitkeep").write_text("", encoding="utf-8")
    print(f"Recreated clean results directory: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
