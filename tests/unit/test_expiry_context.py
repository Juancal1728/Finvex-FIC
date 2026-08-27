"""El contexto de vencimiento: forward implicito, anclaje y cobertura de colas."""

from __future__ import annotations

import math

import pytest

from finvex.adapters.data import SyntheticOptionsDataProvider
from finvex.application import PointInTimeMarketDataStore
from finvex.domain.errors import DomainValidationError
from finvex.domain.model import ExpiryContext, RateCurve
from finvex.domain.options import largest_strike_at_or_below
from finvex.domain.values import AsOf


def build_context(
    store: PointInTimeMarketDataStore,
    provider: SyntheticOptionsDataProvider,
    as_of: AsOf,
    index_of_expiration: int = 0,
) -> tuple[ExpiryContext, float]:
    """Devuelve el contexto extraido y el forward verdadero de la densidad."""
    chain = store.option_chain(as_of, "SPX")
    curve = store.rate_curve(as_of)
    expiration_date = chain.expiration_dates[index_of_expiration]
    settlement = provider._settlement_for(expiration_date)
    context = ExpiryContext.from_chain(chain, expiration_date, settlement, curve)
    true_forward = provider.risk_neutral_density_for(as_of, expiration_date).forward_price
    return context, true_forward


@pytest.mark.parametrize("index_of_expiration", [0, 1, 2])
def test_el_forward_implicito_recupera_el_forward_verdadero(
    market_data_store: PointInTimeMarketDataStore,
    synthetic_provider: SyntheticOptionsDataProvider,
    first_observation_date: AsOf,
    index_of_expiration: int,
) -> None:
    """La prueba central de la paridad put-call.

    El mercado sintetico tiene un forward que **no** coincide con el spot,
    porque hay tasa y rendimiento de dividendos. Recuperarlo exactamente
    demuestra que la extraccion funciona sin necesidad de conocer ninguno de
    los dos.
    """
    context, true_forward = build_context(
        market_data_store, synthetic_provider, first_observation_date, index_of_expiration
    )
    assert context.forward_price == pytest.approx(true_forward, rel=1e-12)


def test_el_forward_extraido_difiere_del_spot(
    market_data_store: PointInTimeMarketDataStore,
    synthetic_provider: SyntheticOptionsDataProvider,
    first_observation_date: AsOf,
) -> None:
    """Si coincidieran, el test anterior no probaria nada."""
    context, _ = build_context(market_data_store, synthetic_provider, first_observation_date)
    spot_price = synthetic_provider.spot_price_on(first_observation_date)
    assert context.forward_price != pytest.approx(spot_price, rel=1e-6)


def test_el_strike_de_anclaje_queda_al_nivel_del_forward_o_por_debajo(
    market_data_store: PointInTimeMarketDataStore,
    synthetic_provider: SyntheticOptionsDataProvider,
    first_observation_date: AsOf,
) -> None:
    context, _ = build_context(market_data_store, synthetic_provider, first_observation_date)
    assert context.anchor_strike_price <= context.forward_price


def test_la_clasificacion_fuera_del_dinero_se_mide_contra_el_forward(
    market_data_store: PointInTimeMarketDataStore,
    synthetic_provider: SyntheticOptionsDataProvider,
    first_observation_date: AsOf,
) -> None:
    context, _ = build_context(market_data_store, synthetic_provider, first_observation_date)
    anchor = context.anchor_strike_price
    assert context.is_out_of_the_money(anchor * 1.10, is_call=True)
    assert not context.is_out_of_the_money(anchor * 0.90, is_call=True)
    assert context.is_out_of_the_money(anchor * 0.90, is_call=False)
    assert not context.is_out_of_the_money(anchor * 1.10, is_call=False)


def test_la_moneyness_logaritmica_es_cero_en_el_forward(
    market_data_store: PointInTimeMarketDataStore,
    synthetic_provider: SyntheticOptionsDataProvider,
    first_observation_date: AsOf,
) -> None:
    context, _ = build_context(market_data_store, synthetic_provider, first_observation_date)
    assert context.log_moneyness(context.forward_price) == pytest.approx(0.0, abs=1e-12)


def test_la_cobertura_de_strikes_es_mas_profunda_hacia_abajo(
    market_data_store: PointInTimeMarketDataStore,
    synthetic_provider: SyntheticOptionsDataProvider,
    first_observation_date: AsOf,
) -> None:
    """El mercado sintetico reproduce la asimetria real de la rejilla.

    Los mercados de opciones sobre indices cotizan strikes mucho mas abajo
    que arriba, y eso importa porque la asimetria implicita se estima
    justamente con el ala izquierda.
    """
    context, _ = build_context(market_data_store, synthetic_provider, first_observation_date)
    coverage = context.strike_coverage
    assert coverage.standard_deviations_below_forward > coverage.standard_deviations_above_forward
    assert coverage.usable_strike_count > 50


def test_la_curva_plana_descuenta_de_forma_consistente() -> None:
    curve = RateCurve.flat(0.04)
    assert curve.rate_at(0.5) == pytest.approx(0.04)
    assert curve.discount_factor_at(0.5) == pytest.approx(math.exp(-0.04 * 0.5))


def test_una_curva_desordenada_se_rechaza_en_vez_de_ordenarse_sola() -> None:
    """Ordenar en silencio esconderia un error del proveedor."""
    with pytest.raises(DomainValidationError, match="orden creciente"):
        RateCurve(tenors_in_years=(1.0, 0.5), continuously_compounded_rates=(0.04, 0.03))


def test_sin_strikes_bajo_el_forward_no_se_puede_anclar() -> None:
    with pytest.raises(DomainValidationError, match="anclar"):
        largest_strike_at_or_below((6000.0, 6500.0), 5000.0)
