"""AsOf: el tipo que sostiene el invariante anti-look-ahead."""

from __future__ import annotations

import datetime as dt

import pytest

from finvex.domain.errors import DomainValidationError
from finvex.domain.values import AsOf


def test_parse_acepta_iso_date_datetime_y_as_of() -> None:
    esperado = AsOf(dt.date(2024, 3, 15))
    assert AsOf.parse("2024-03-15") == esperado
    assert AsOf.parse(dt.date(2024, 3, 15)) == esperado
    assert AsOf.parse(dt.datetime(2024, 3, 15, 16, 0)) == esperado
    assert AsOf.parse(esperado) == esperado


def test_datetime_directo_es_rechazado() -> None:
    """Un datetime traeria una hora implicita y la frontera dejaria de ser clara."""
    with pytest.raises(DomainValidationError):
        AsOf(dt.datetime(2024, 3, 15, 16, 0))  # type: ignore[arg-type]


def test_iso_invalida_es_rechazada() -> None:
    with pytest.raises(DomainValidationError):
        AsOf.parse("15/03/2024")


def test_covers_incluye_el_mismo_dia_y_excluye_el_futuro() -> None:
    as_of = AsOf.parse("2024-03-15")
    assert as_of.covers(dt.date(2024, 3, 14))
    assert as_of.covers(dt.date(2024, 3, 15))
    assert not as_of.covers(dt.date(2024, 3, 16))


def test_es_inmutable_y_ordenable() -> None:
    a, b = AsOf.parse("2024-01-01"), AsOf.parse("2024-06-01")
    assert a < b
    assert sorted([b, a]) == [a, b]
    with pytest.raises(AttributeError):
        a.date = dt.date(2030, 1, 1)  # type: ignore[misc]


def test_shift_days() -> None:
    assert AsOf.parse("2024-02-28").shift_days(2) == AsOf.parse("2024-03-01")
