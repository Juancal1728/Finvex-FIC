"""La regla de dependencia hexagonal, verificada por test.

Una regla de arquitectura que solo vive en un documento se rompe el dia que
alguien tiene prisa. Esta la rompe el CI.

    adapters      ->  ports, domain, config, adapters
    application   ->  ports, domain, application
    ports         ->  domain, ports
    domain        ->  domain            (y nada mas del proyecto)
    config        ->  domain, config

El composition root (`finvex.bootstrap`) y la CLI son la unica excepcion:
existen precisamente para conocer las implementaciones concretas.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "finvex"

ALLOWED: dict[str, set[str]] = {
    "domain": {"domain"},
    "ports": {"domain", "ports"},
    "application": {"domain", "ports", "application"},
    "config": {"domain", "config"},
    "adapters": {"domain", "ports", "config", "adapters"},
}

# Modulos que pueden conocerlo todo: son el cableado, no la logica.
COMPOSITION_ROOT = {"bootstrap", "adapters.cli.main"}


def _layer_of(path: pathlib.Path) -> str | None:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else None


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _finvex_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(a.name for a in node.names if a.name.startswith("finvex"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("finvex"):
            found.append(node.module)
    return found


def test_dependency_rule_holds() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        layer = _layer_of(path)
        if layer is None or layer not in ALLOWED:
            continue
        if _module_name(path) in COMPOSITION_ROOT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _finvex_imports(tree):
            parts = imported.split(".")
            if len(parts) < 2:
                continue
            target = parts[1]
            if target not in ALLOWED[layer]:
                violations.append(f"{_module_name(path)} ({layer}) importa {imported} ({target})")
    assert not violations, "Violaciones de la regla de dependencia:\n  " + "\n  ".join(violations)


def test_domain_imports_no_adapters_transitively() -> None:
    """El dominio no puede depender de pandas, duckdb ni de ningun cliente de datos.

    El nucleo trabaja sobre estructuras propias y arreglos numericos. Si
    necesita pandas para existir, deja de ser portable y de ser testeable sin
    montar un dataset.
    """
    forbidden = {"duckdb", "pyarrow", "requests", "httpx", "wrds", "lseg", "refinitiv"}
    violations: list[str] = []
    for path in sorted((SRC / "domain").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in forbidden:
                    violations.append(f"{_module_name(path)} importa {name}")
    assert not violations, "El dominio importa infraestructura:\n  " + "\n  ".join(violations)
