"""Invariantes I1 e I3: sin reloj del sistema y sin aleatoriedad global.

Los dos fallos que estos tests previenen son silenciosos. Un `date.today()`
enterrado en un modulo de investigacion produce un backtest que se ve bien y
no significa nada. Un `np.random.normal()` sin generador explicito produce
resultados que no se pueden reproducir, y no hay forma de notarlo mirando la
salida.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "finvex"

# La CLI y el sellado de artefactos pueden mirar el reloj: es su trabajo.
CLOCK_EXEMPT = {
    "adapters/cli/main.py",
    "utils/clock.py",
    "adapters/persistence/artifacts.py",
}

FORBIDDEN_ATTRS = {"now", "today", "utcnow"}

# `default_rng` y `Generator` son la forma correcta de obtener aleatoriedad
# explicita y reproducible: crean un generador propio en vez de mutar el
# estado global de numpy. Lo que se prohibe es el resto de `np.random.*`, que
# es la interfaz heredada y comparte un estado invisible entre modulos.
RANDOM_ATTRIBUTES_ALLOWED = {"default_rng", "Generator", "SeedSequence", "PCG64"}


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC).as_posix()


def test_no_system_clock_in_research_code() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if _rel(path) in CLOCK_EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in FORBIDDEN_ATTRS
            ):
                violations.append(f"{_rel(path)}:{node.lineno} llama a .{node.func.attr}()")
    assert not violations, (
        "Reloj del sistema en codigo de investigacion. Toda funcion recibe as_of:\n  "
        + "\n  ".join(violations)
    )


def test_no_global_numpy_random() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "random"
                and getattr(node.value.value, "id", None) in {"np", "numpy"}
                and node.attr not in RANDOM_ATTRIBUTES_ALLOWED
            ):
                violations.append(f"{_rel(path)}:{node.lineno} usa np.random.{node.attr}")
    assert not violations, (
        "Aleatoriedad del estado global de numpy. Crea un Generator explicito "
        "con np.random.default_rng(semilla) y pasalo como argumento:\n  " + "\n  ".join(violations)
    )
