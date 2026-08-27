"""Validacion del mercado sintetico como instrumento de medida.

Antes de usar este mercado para validar BKM hay que validar el mercado. Estos
tests comprueban las tres propiedades de las que depende todo lo demas:
paridad exacta, determinismo y consistencia entre los precios generados y la
densidad que dice haberlos generado.
"""

from __future__ import annotations

import math

import pytest

from finvex.adapters.data import SyntheticMarketConfiguration, SyntheticOptionsDataProvider
from finvex.application import PointInTimeMarketDataStore
from finvex.domain.values import AsOf, OptionType, SettlementConvention

pytestmark = pytest.mark.validation

SMALL_CONFIGURATION = SyntheticMarketConfiguration(
    number_of_observation_dates=3,
    maximum_days_to_expiry=40,
    compute_vendor_implied_volatility=False,
)


def test_la_paridad_put_call_es_exacta_en_toda_la_cadena_generada() -> None:
    """Se comprueba contrato por contrato, no solo en el strike de referencia.

    Es la propiedad que permite que el test de extraccion del forward mida el
    extractor y no el ruido de los datos.
    """
    provider = SyntheticOptionsDataProvider(SMALL_CONFIGURATION)
    store = PointInTimeMarketDataStore(provider)
    as_of = AsOf(provider.available_observation_dates("SPX")[0])
    chain = store.option_chain(as_of, "SPX")

    largest_absolute_residual = 0.0
    number_of_pairs_checked = 0

    for expiration_date in chain.expiration_dates:
        density = provider.risk_neutral_density_for(as_of, expiration_date)
        for pair in chain.call_put_pairs(expiration_date):
            price_difference = pair.price_difference
            assert price_difference is not None
            theoretical_difference = density.discount_factor * (
                density.forward_price - pair.strike_price
            )
            largest_absolute_residual = max(
                largest_absolute_residual, abs(price_difference - theoretical_difference)
            )
            number_of_pairs_checked += 1

    assert number_of_pairs_checked > 100
    assert largest_absolute_residual < 1e-8


def test_el_punto_medio_coincide_con_el_precio_teorico() -> None:
    """El spread es proporcional al precio, asi que el medio es exacto.

    De aqui depende la paridad exacta: con un spread absoluto, el punto medio
    de una serie muy barata quedaria desplazado.
    """
    provider = SyntheticOptionsDataProvider(SMALL_CONFIGURATION)
    as_of = AsOf(provider.available_observation_dates("SPX")[0])
    chain = provider.option_chain(as_of, "SPX")

    for expiration_date in chain.expiration_dates[:2]:
        density = provider.risk_neutral_density_for(as_of, expiration_date)
        for quote in chain.for_expiration(expiration_date):
            theoretical_price = density.european_option_price(
                quote.strike_price, is_call=quote.option_type is OptionType.CALL
            )
            assert quote.mid_price == pytest.approx(theoretical_price, rel=1e-12)


def test_el_mercado_es_reproducible_con_la_misma_semilla() -> None:
    """Dos proveedores con la misma configuracion generan el mismo mercado."""
    first_provider = SyntheticOptionsDataProvider(SMALL_CONFIGURATION)
    second_provider = SyntheticOptionsDataProvider(SMALL_CONFIGURATION)
    as_of = AsOf(first_provider.available_observation_dates("SPX")[0])

    assert first_provider.spot_price_on(as_of) == second_provider.spot_price_on(as_of)
    first_chain = first_provider.option_chain(as_of, "SPX")
    second_chain = second_provider.option_chain(as_of, "SPX")
    assert len(first_chain) == len(second_chain)
    assert [quote.mid_price for quote in first_chain] == [quote.mid_price for quote in second_chain]


def test_conviven_las_dos_convenciones_de_liquidacion() -> None:
    """Igual que en el complejo SPX real: series AM del tercer viernes y PM semanales.

    Que convivan importa porque ejercita el manejo de la hora de expiracion
    desde el principio, en vez de descubrir el problema cuando lleguen datos
    reales.
    """
    provider = SyntheticOptionsDataProvider(SMALL_CONFIGURATION)
    as_of = AsOf(provider.available_observation_dates("SPX")[0])
    chain = provider.option_chain(as_of, "SPX")

    settlements_present = {quote.settlement_convention for quote in chain}
    roots_present = {quote.contract_root for quote in chain}

    assert settlements_present == {
        SettlementConvention.MORNING,
        SettlementConvention.AFTERNOON,
    }
    assert roots_present == {"SPX", "SPXW"}


def test_la_volatilidad_implicita_del_proveedor_reproduce_los_precios() -> None:
    """Cuando el proveedor la calcula, la inversion tiene que cerrar el circulo.

    Se usa una configuracion propia y minima porque invertir la volatilidad
    de cada contrato es lo que domina el tiempo de generacion.
    """
    configuration = SyntheticMarketConfiguration(
        number_of_observation_dates=1,
        minimum_days_to_expiry=25,
        maximum_days_to_expiry=35,
        compute_vendor_implied_volatility=True,
    )
    provider = SyntheticOptionsDataProvider(configuration)
    as_of = AsOf(provider.available_observation_dates("SPX")[0])
    chain = provider.option_chain(as_of, "SPX")
    assert len(chain) > 0

    from finvex.domain.options import black_scholes_price

    for expiration_date in chain.expiration_dates:
        density = provider.risk_neutral_density_for(as_of, expiration_date)
        for quote in chain.for_expiration(expiration_date):
            assert quote.vendor_implied_volatility is not None
            reconstructed_price = black_scholes_price(
                option_type=quote.option_type,
                forward_price=density.forward_price,
                strike_price=quote.strike_price,
                time_to_expiry_years=density.time_to_expiry_years,
                volatility=quote.vendor_implied_volatility,
                discount_factor=density.discount_factor,
            )
            assert quote.mid_price is not None
            assert reconstructed_price == pytest.approx(quote.mid_price, rel=1e-6, abs=1e-8)


def test_la_sonrisa_de_volatilidad_tiene_pendiente_negativa() -> None:
    """La densidad con escenario de caida debe producir skew, no una linea plana.

    Es la comprobacion de que el mercado sintetico es interesante. Si la
    volatilidad implicita fuera constante en el strike, la distribucion seria
    lognormal, MFIS seria cero y no habria nada que estudiar.
    """
    configuration = SyntheticMarketConfiguration(
        number_of_observation_dates=1,
        minimum_days_to_expiry=25,
        maximum_days_to_expiry=35,
        compute_vendor_implied_volatility=True,
    )
    provider = SyntheticOptionsDataProvider(configuration)
    as_of = AsOf(provider.available_observation_dates("SPX")[0])
    chain = provider.option_chain(as_of, "SPX")
    expiration_date = chain.expiration_dates[0]
    density = provider.risk_neutral_density_for(as_of, expiration_date)

    puts_below_the_money = sorted(
        (
            quote
            for quote in chain.for_expiration(expiration_date).of_type(OptionType.PUT)
            if quote.strike_price < density.forward_price
        ),
        key=lambda quote: quote.strike_price,
    )
    assert len(puts_below_the_money) > 20

    lowest_strike_volatility = puts_below_the_money[0].vendor_implied_volatility
    at_the_money_volatility = puts_below_the_money[-1].vendor_implied_volatility
    assert lowest_strike_volatility is not None
    assert at_the_money_volatility is not None
    assert lowest_strike_volatility > at_the_money_volatility

    log_moneyness_span = abs(math.log(puts_below_the_money[0].strike_price / density.forward_price))
    smile_slope = (lowest_strike_volatility - at_the_money_volatility) / log_moneyness_span
    assert smile_slope > 0.05
