"""Punto de entrada de linea de comandos.

La CLI es un adaptador de entrada: traduce argumentos a casos de uso y no
contiene logica de investigacion. En la fase 0 solo expone `doctor`, que
reporta el entorno y sirve para verificar que la instalacion quedo bien.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys

import finvex

_OPTIONAL_GROUPS: dict[str, tuple[str, ...]] = {
    "core": ("numpy", "pandas", "scipy", "pydantic", "yaml", "pyarrow", "duckdb"),
    "research": ("statsmodels", "arch", "cvxpy", "highspy", "clarabel"),
    "options": ("QuantLib",),
    "notebooks": ("jupyterlab", "matplotlib"),
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _doctor() -> int:
    print(f"finvex        {finvex.__version__}")
    print(f"python        {platform.python_version()}  ({sys.executable})")
    print(f"plataforma    {platform.platform()}")
    print()
    missing_core = False
    for group, modules in _OPTIONAL_GROUPS.items():
        marks = []
        for m in modules:
            ok = _installed(m)
            if group == "core" and not ok:
                missing_core = True
            marks.append(f"{'ok' if ok else '--'} {m}")
        print(f"{group:<10}  " + "   ".join(marks))
    print()
    print("proveedores   ninguno registrado todavia (fase 1)")
    if missing_core:
        print()
        print("Faltan dependencias del grupo core. Corre: make setup")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finvex", description="FINVEX-FIC")
    parser.add_argument("--version", action="version", version=finvex.__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Reporta entorno y disponibilidad de dependencias")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
