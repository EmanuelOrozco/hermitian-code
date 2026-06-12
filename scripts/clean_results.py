#!/usr/bin/env python3
"""Elimina salidas generadas en results/.

No borra docs/ porque docs/ contiene documentación fuente versionable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> int:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)

    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS / "band_structures").mkdir(parents=True, exist_ok=True)
    (RESULTS / "field_profiles").mkdir(parents=True, exist_ok=True)
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)

    keep = RESULTS / ".gitkeep"
    keep.write_text("", encoding="utf-8")

    print(f"Clean results directory recreated at: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
