"""El paquete se importa y la CLI responde."""

from __future__ import annotations

import finvex
from finvex.adapters.cli.main import main


def test_version_disponible() -> None:
    assert finvex.__version__


def test_doctor_corre(capsys) -> None:  # type: ignore[no-untyped-def]
    code = main(["doctor"])
    salida = capsys.readouterr().out
    assert "finvex" in salida
    assert "python" in salida
    assert code in (0, 1)
