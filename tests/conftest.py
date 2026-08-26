"""Fixtures compartidas.

La mayoria de los tests del proyecto no debe tocar datos reales. Las fixtures
sinteticas viven aqui y crecen con el proyecto (fase 1).
"""

from __future__ import annotations

import pytest
from numpy.random import Generator, default_rng


@pytest.fixture
def rng() -> Generator:
    """Generador con semilla fija: nunca aleatoriedad global en tests."""
    return default_rng(20260826)
