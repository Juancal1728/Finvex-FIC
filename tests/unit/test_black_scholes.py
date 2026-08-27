"""Valuacion auxiliar: paridad, monotonia e inversion de volatilidad implicita."""

from __future__ import annotations

import math

import pytest

from finvex.domain.errors import DomainValidationError
from finvex.domain.options import (
    black_scholes_call_price,
    black_scholes_put_price,
    black_scholes_vega,
    implied_volatility_from_price,
)
from finvex.domain.values import OptionType

FORWARD_PRICE = 5000.0
TIME_TO_EXPIRY_YEARS = 30.0 / 365.0
VOLATILITY = 0.18
DISCOUNT_FACTOR = math.exp(-0.04 * TIME_TO_EXPIRY_YEARS)


@pytest.mark.parametrize("strike_price", [3500.0, 4500.0, 5000.0, 5500.0, 7000.0])
def test_la_paridad_put_call_se_cumple_exactamente(strike_price: float) -> None:
    """El put se deriva del call, asi que la paridad es exacta por construccion."""
    call_price = black_scholes_call_price(
        FORWARD_PRICE, strike_price, TIME_TO_EXPIRY_YEARS, VOLATILITY, DISCOUNT_FACTOR
    )
    put_price = black_scholes_put_price(
        FORWARD_PRICE, strike_price, TIME_TO_EXPIRY_YEARS, VOLATILITY, DISCOUNT_FACTOR
    )
    expected_difference = DISCOUNT_FACTOR * (FORWARD_PRICE - strike_price)
    assert call_price - put_price == pytest.approx(expected_difference, abs=1e-10)


def test_el_precio_del_call_crece_con_la_volatilidad() -> None:
    """La monotonia es lo que hace bien definida la inversion a volatilidad implicita."""
    prices = [
        black_scholes_call_price(FORWARD_PRICE, 5000.0, TIME_TO_EXPIRY_YEARS, volatility)
        for volatility in (0.05, 0.10, 0.20, 0.40, 0.80)
    ]
    assert prices == sorted(prices)
    assert len(set(prices)) == len(prices)


def test_con_volatilidad_cero_el_precio_es_el_valor_intrinseco() -> None:
    in_the_money = black_scholes_call_price(5000.0, 4000.0, TIME_TO_EXPIRY_YEARS, 0.0)
    out_of_the_money = black_scholes_call_price(5000.0, 6000.0, TIME_TO_EXPIRY_YEARS, 0.0)
    assert in_the_money == pytest.approx(1000.0)
    assert out_of_the_money == pytest.approx(0.0)


def test_vega_es_positiva_y_maxima_cerca_del_dinero() -> None:
    at_the_money = black_scholes_vega(FORWARD_PRICE, 5000.0, TIME_TO_EXPIRY_YEARS, VOLATILITY)
    far_out = black_scholes_vega(FORWARD_PRICE, 8000.0, TIME_TO_EXPIRY_YEARS, VOLATILITY)
    assert at_the_money > 0.0
    assert far_out >= 0.0
    assert at_the_money > far_out


@pytest.mark.parametrize("strike_price", [4200.0, 4600.0, 5000.0, 5400.0, 5800.0])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_la_inversion_de_volatilidad_implicita_recupera_la_volatilidad_original(
    strike_price: float, option_type: OptionType
) -> None:
    """Ida y vuelta: precio desde volatilidad, y volatilidad desde precio.

    El rango de strikes cubre unas cuatro desviaciones estandar implicitas a
    cada lado, que es donde una cadena real tiene precios informativos. Mas
    alla, el valor temporal cae por debajo de la resolucion numerica y la
    volatilidad implicita deja de estar identificada: ese caso tiene su
    propio test.
    """
    price = (
        black_scholes_call_price if option_type is OptionType.CALL else black_scholes_put_price
    )(FORWARD_PRICE, strike_price, TIME_TO_EXPIRY_YEARS, VOLATILITY, DISCOUNT_FACTOR)

    recovered_volatility = implied_volatility_from_price(
        option_price=price,
        option_type=option_type,
        forward_price=FORWARD_PRICE,
        strike_price=strike_price,
        time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
        discount_factor=DISCOUNT_FACTOR,
    )
    assert recovered_volatility == pytest.approx(VOLATILITY, rel=1e-8)


def test_un_precio_por_debajo_del_intrinseco_se_rechaza() -> None:
    intrinsic_value = DISCOUNT_FACTOR * (FORWARD_PRICE - 4000.0)
    with pytest.raises(DomainValidationError, match="no-arbitraje"):
        implied_volatility_from_price(
            option_price=intrinsic_value * 0.9,
            option_type=OptionType.CALL,
            forward_price=FORWARD_PRICE,
            strike_price=4000.0,
            time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
            discount_factor=DISCOUNT_FACTOR,
        )


def test_muy_lejos_del_dinero_la_volatilidad_implicita_no_esta_identificada() -> None:
    """El precio de una serie muy alejada no contiene informacion de volatilidad.

    No es un fallo del algoritmo: es una propiedad del problema. Cuando el
    valor temporal cae por debajo de la resolucion numerica, distintas
    volatilidades producen el mismo precio y la inversion deja de estar
    definida. La funcion lo dice en vez de devolver la cota inferior, que
    seria una respuesta falsa con apariencia de respuesta.

    Esta es la razon de fondo por la que la extraccion de momentos trunca las
    alas en vez de integrar hasta cero, y por la que la cobertura efectiva de
    strikes es un dato que hay que reportar junto a cada momento.
    """
    very_low_strike = 3000.0
    negligible_price = black_scholes_put_price(
        FORWARD_PRICE, very_low_strike, TIME_TO_EXPIRY_YEARS, VOLATILITY, DISCOUNT_FACTOR
    )
    assert negligible_price < 1e-9

    with pytest.raises(DomainValidationError, match="no esta identificada"):
        implied_volatility_from_price(
            option_price=negligible_price,
            option_type=OptionType.PUT,
            forward_price=FORWARD_PRICE,
            strike_price=very_low_strike,
            time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
            discount_factor=DISCOUNT_FACTOR,
        )


def test_una_volatilidad_en_porcentaje_en_vez_de_decimal_produce_un_precio_absurdo() -> None:
    """Guardia contra el error de unidades mas comun del area.

    Pasar 18 en vez de 0.18 no lanza ningun error: devuelve un precio, y ese
    precio se acerca al forward completo. Una call a un mes que vale casi
    tanto como el subyacente es el sintoma, y el test lo documenta para que
    quien lo vea alguna vez sepa donde mirar.
    """
    absurd_price = black_scholes_call_price(FORWARD_PRICE, 5000.0, TIME_TO_EXPIRY_YEARS, 18.0)
    reasonable_price = black_scholes_call_price(FORWARD_PRICE, 5000.0, TIME_TO_EXPIRY_YEARS, 0.18)
    assert absurd_price > 0.98 * FORWARD_PRICE
    assert absurd_price > 20.0 * reasonable_price
