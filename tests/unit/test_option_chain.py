"""La cadena de opciones: agrupaciones y guardias de homogeneidad."""

from __future__ import annotations

import datetime as dt

import pytest

from finvex.domain.errors import DomainValidationError
from finvex.domain.model import OptionChain, OptionQuote
from finvex.domain.values import AsOf, ExerciseStyle, OptionType, SettlementConvention

OBSERVATION_DATE = AsOf.parse("2024-01-02")
EXPIRATION_DATE = dt.date(2024, 2, 2)
OTHER_EXPIRATION_DATE = dt.date(2024, 3, 1)


def build_quote(
    option_type: OptionType,
    strike_price: float,
    *,
    expiration_date: dt.date = EXPIRATION_DATE,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    bid_price: float | None = 10.0,
    ask_price: float | None = 12.0,
    as_of: AsOf = OBSERVATION_DATE,
    underlying_symbol: str = "SPX",
) -> OptionQuote:
    return OptionQuote(
        as_of=as_of,
        underlying_symbol=underlying_symbol,
        contract_root="SPX",
        option_type=option_type,
        strike_price=strike_price,
        expiration_date=expiration_date,
        exercise_style=exercise_style,
        settlement_convention=SettlementConvention.MORNING,
        underlying_price=5000.0,
        data_source="test",
        bid_price=bid_price,
        ask_price=ask_price,
    )


def test_el_precio_medio_y_el_spread_relativo_se_derivan_de_las_posturas() -> None:
    quote = build_quote(OptionType.CALL, 5000.0, bid_price=10.0, ask_price=12.0)
    assert quote.mid_price == pytest.approx(11.0)
    assert quote.bid_ask_spread == pytest.approx(2.0)
    assert quote.relative_bid_ask_spread == pytest.approx(2.0 / 11.0)


def test_sin_posturas_los_derivados_son_nulos_en_vez_de_inventados() -> None:
    """Un dato ausente se propaga como ausente. Rellenarlo seria falsearlo."""
    quote = build_quote(OptionType.CALL, 5000.0, bid_price=None, ask_price=None)
    assert quote.mid_price is None
    assert quote.relative_bid_ask_spread is None


def test_se_emparejan_call_y_put_solo_donde_existen_las_dos_patas() -> None:
    chain = OptionChain.from_quotes(
        [
            build_quote(OptionType.CALL, 4900.0),
            build_quote(OptionType.PUT, 4900.0),
            build_quote(OptionType.CALL, 5000.0),
            build_quote(OptionType.PUT, 5000.0),
            build_quote(OptionType.CALL, 5100.0),  # sin put: no forma par
        ]
    )
    pairs = chain.call_put_pairs(EXPIRATION_DATE)
    assert [pair.strike_price for pair in pairs] == [4900.0, 5000.0]


def test_los_vencimientos_salen_ordenados_y_sin_repetir() -> None:
    chain = OptionChain.from_quotes(
        [
            build_quote(OptionType.CALL, 5000.0, expiration_date=OTHER_EXPIRATION_DATE),
            build_quote(OptionType.CALL, 5000.0, expiration_date=EXPIRATION_DATE),
            build_quote(OptionType.PUT, 5000.0, expiration_date=EXPIRATION_DATE),
        ]
    )
    assert chain.expiration_dates == (EXPIRATION_DATE, OTHER_EXPIRATION_DATE)


def test_una_sola_serie_americana_invalida_la_extraccion_libre_de_modelo() -> None:
    """La comprobacion es a nivel de cadena, no de cotizacion.

    Basta una serie americana para contaminar el momento del vencimiento
    completo, asi que la pregunta correcta es sobre el conjunto.
    """
    chain = OptionChain.from_quotes(
        [
            build_quote(OptionType.CALL, 5000.0),
            build_quote(OptionType.PUT, 5000.0, exercise_style=ExerciseStyle.AMERICAN),
        ]
    )
    assert chain.supports_model_free_moment_extraction() is False
    assert chain.exercise_styles_present() == frozenset(
        {ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN}
    )


def test_una_cadena_con_fechas_mezcladas_se_rechaza() -> None:
    """Mezclar fechas dentro de una cadena es una via directa al look-ahead."""
    with pytest.raises(DomainValidationError, match="look-ahead"):
        OptionChain(
            as_of=OBSERVATION_DATE,
            underlying_symbol="SPX",
            underlying_price=5000.0,
            quotes=(
                build_quote(OptionType.CALL, 5000.0),
                build_quote(OptionType.PUT, 5000.0, as_of=AsOf.parse("2024-01-03")),
            ),
        )


def test_una_cadena_con_subyacentes_mezclados_se_rechaza() -> None:
    with pytest.raises(DomainValidationError, match="NDX"):
        OptionChain(
            as_of=OBSERVATION_DATE,
            underlying_symbol="SPX",
            underlying_price=5000.0,
            quotes=(build_quote(OptionType.CALL, 5000.0, underlying_symbol="NDX"),),
        )


def test_una_cotizacion_ya_vencida_se_rechaza() -> None:
    with pytest.raises(DomainValidationError, match="posterior"):
        build_quote(OptionType.CALL, 5000.0, expiration_date=dt.date(2023, 12, 1))


def test_filtrar_devuelve_una_cadena_nueva_y_no_modifica_la_original() -> None:
    chain = OptionChain.from_quotes(
        [build_quote(OptionType.CALL, 5000.0), build_quote(OptionType.PUT, 5000.0)]
    )
    calls_only = chain.of_type(OptionType.CALL)
    assert len(calls_only) == 1
    assert len(chain) == 2
