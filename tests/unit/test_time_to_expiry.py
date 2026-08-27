"""La convencion de tiempo, que es donde se esconden los errores de replica."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from finvex.domain.errors import DomainValidationError
from finvex.domain.values import MINUTES_PER_YEAR, SettlementConvention, TimeToExpiry


def test_un_ano_completo_son_exactamente_los_minutos_de_la_convencion() -> None:
    assert TimeToExpiry(MINUTES_PER_YEAR).years == pytest.approx(1.0)


def test_la_liquidacion_am_y_la_pm_no_dan_el_mismo_tiempo() -> None:
    """La diferencia es de seis horas y media, y no es despreciable.

    Es exactamente el error que hace que una replica del VIX difiera en
    decimas de punto sin que la integral tenga nada malo.
    """
    valuation_date = dt.date(2024, 1, 2)
    expiration_date = dt.date(2024, 2, 2)

    morning = TimeToExpiry.from_expiration_date(
        valuation_date, expiration_date, SettlementConvention.MORNING
    )
    afternoon = TimeToExpiry.from_expiration_date(
        valuation_date, expiration_date, SettlementConvention.AFTERNOON
    )

    minutes_between_open_and_close = (16 - 9) * 60 - 30
    assert afternoon.minutes - morning.minutes == minutes_between_open_and_close
    assert afternoon.years > morning.years


def test_el_tiempo_se_mide_desde_el_cierre_de_la_fecha_de_observacion() -> None:
    """Una observacion de fin de dia se valua al cierre, no a medianoche."""
    time_to_expiry = TimeToExpiry.from_expiration_date(
        valuation_date=dt.date(2024, 1, 2),
        expiration_date=dt.date(2024, 1, 3),
        settlement=SettlementConvention.AFTERNOON,
    )
    assert time_to_expiry.minutes == 24 * 60


def test_rechaza_instantes_sin_zona_horaria() -> None:
    naive_moment = dt.datetime(2024, 1, 2, 16, 0)
    aware_moment = dt.datetime(2024, 2, 2, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(DomainValidationError, match="zona horaria"):
        TimeToExpiry.between_moments(naive_moment, aware_moment)


def test_rechaza_un_vencimiento_ya_pasado() -> None:
    with pytest.raises(DomainValidationError, match="positivo"):
        TimeToExpiry.from_expiration_date(
            valuation_date=dt.date(2024, 3, 1),
            expiration_date=dt.date(2024, 2, 1),
            settlement=SettlementConvention.AFTERNOON,
        )


def test_es_ordenable() -> None:
    corto = TimeToExpiry(1_000)
    largo = TimeToExpiry(10_000)
    assert corto < largo
    assert sorted([largo, corto]) == [corto, largo]
