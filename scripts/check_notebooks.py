"""Rechaza definiciones de funciones o clases dentro de notebooks.

La logica vive en `src/`. Un notebook que define funciones se convierte, en
cuestion de semanas, en el proyecto real: sin tests, sin revision y sin
posibilidad de reutilizar nada. El hook es agresivo a proposito.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^\s*(def |class )", re.MULTILINE)


def check(path: Path) -> list[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: no se pudo leer ({exc})"]

    problems: list[str] = []
    for i, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if PATTERN.search(source):
            problems.append(f"{path}: celda {i} define una funcion o clase")
    return problems


def main(argv: list[str]) -> int:
    problems: list[str] = []
    for arg in argv:
        problems.extend(check(Path(arg)))
    if problems:
        print("Logica en notebooks. Mueve esto a src/finvex/:")
        for p in problems:
            print(f"  {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
