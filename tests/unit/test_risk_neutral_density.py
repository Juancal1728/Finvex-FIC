"""La densidad de referencia del checkpoint V1.

Si estos tests fallan, no tiene sentido probar BKM contra nada: el valor de
referencia estaria mal y el estimador se estaria comparando con un numero
equivocado.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.random import Generator

from finvex.domain.errors import DomainValidationError
from finvex.domain.model import (
    LognormalMixtureComponent,
    LognormalMixtureRiskNeutralDensity,
    LogReturnMoments,
)

SPOT_PRICE = 5000.0
TIME_TO_EXPIRY_YEARS = 30.0 / 365.0
RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.015
FORWARD_PRICE = SPOT_PRICE * math.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * TIME_TO_EXPIRY_YEARS)
DISCOUNT_FACTOR = math.exp(-RISK_FREE_RATE * TIME_TO_EXPIRY_YEARS)


def build_black_scholes_density(volatility: float) -> LognormalMixtureRiskNeutralDensity:
    return LognormalMixtureRiskNeutralDensity.from_single_lognormal(
        spot_price=SPOT_PRICE,
        forward_price=FORWARD_PRICE,
        time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
        annualized_volatility=volatility,
        discount_factor=DISCOUNT_FACTOR,
    )


def build_crash_density() -> LognormalMixtureRiskNeutralDensity:
    return LognormalMixtureRiskNeutralDensity.from_crash_scenario(
        spot_price=SPOT_PRICE,
        forward_price=FORWARD_PRICE,
        time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
        base_volatility=0.14,
        crash_probability=0.05,
        crash_price_drop=0.12,
        crash_volatility=0.30,
        discount_factor=DISCOUNT_FACTOR,
    )


# ------------------------------------------------------- caso base: lognormal
@pytest.mark.parametrize("volatility", [0.10, 0.18, 0.35])
def test_una_sola_lognormal_reproduce_los_momentos_de_black_scholes(volatility: float) -> None:
    """El caso base, verificable a mano.

    Con un solo regimen, el log-retorno es exactamente normal. Su varianza es
    la volatilidad al cuadrado por el tiempo, su asimetria es cero y su
    curtosis es tres. Cualquier desviacion aqui es un error de calculo, no
    una propiedad de la distribucion.
    """
    moments = build_black_scholes_density(volatility).log_return_moments()
    assert moments.variance == pytest.approx(volatility**2 * TIME_TO_EXPIRY_YEARS, rel=1e-12)
    assert moments.skewness == pytest.approx(0.0, abs=1e-10)
    assert moments.kurtosis == pytest.approx(3.0, rel=1e-10)
    assert moments.excess_kurtosis == pytest.approx(0.0, abs=1e-9)


def test_la_media_del_log_retorno_lleva_la_correccion_de_convexidad() -> None:
    """E[ln(S_T/S_0)] = ln(F/S_0) - sigma^2 T / 2, no ln(F/S_0).

    La diferencia es la correccion de Jensen. Olvidarla es un error clasico
    que desplaza todos los momentos centrales.
    """
    volatility = 0.20
    moments = build_black_scholes_density(volatility).log_return_moments()
    expected_mean = (
        math.log(FORWARD_PRICE / SPOT_PRICE) - 0.5 * volatility**2 * TIME_TO_EXPIRY_YEARS
    )
    assert moments.mean == pytest.approx(expected_mean, rel=1e-12)


# ------------------------------------------------- caso con asimetria y colas
def test_el_escenario_de_caida_produce_asimetria_negativa_y_colas_gruesas() -> None:
    moments = build_crash_density().log_return_moments()
    assert moments.skewness < -1.0
    assert moments.kurtosis > 5.0


def test_los_momentos_del_escenario_de_caida_estan_en_el_rango_realista() -> None:
    """El mercado sintetico debe parecerse al que se va a estudiar.

    Validar el estimador en una region de asimetria -4 que los datos reales
    nunca visitan podria esconder problemas de precision justo en el rango
    que si interesa.
    """
    moments = build_crash_density().log_return_moments()
    annualized_volatility = moments.standard_deviation / math.sqrt(TIME_TO_EXPIRY_YEARS)
    assert 0.15 < annualized_volatility < 0.25
    assert -2.5 < moments.skewness < -1.0
    assert 5.0 < moments.kurtosis < 15.0


def test_los_momentos_analiticos_coinciden_con_la_simulacion(
    random_number_generator: Generator,
) -> None:
    """Contraste independiente: forma cerrada contra Monte Carlo.

    Los momentos analiticos salen de una derivacion; la simulacion sale del
    muestreo. Que coincidan descarta un error algebraico en la derivacion,
    que es la clase de error que ningun test de propiedad detecta.

    Las tolerancias son amplias a proposito: con doscientos mil caminos el
    error estandar del cuarto momento muestral no es pequeno. El test busca
    detectar un signo cambiado o un termino perdido, no medir precision.
    """
    density = build_crash_density()
    analytic = density.log_return_moments()

    terminal_prices = density.sample_terminal_prices(random_number_generator, 200_000)
    simulated_log_returns = np.log(terminal_prices / SPOT_PRICE)

    simulated_mean = float(simulated_log_returns.mean())
    deviations = simulated_log_returns - simulated_mean
    simulated_variance = float((deviations**2).mean())
    simulated_skewness = float((deviations**3).mean()) / simulated_variance**1.5
    simulated_kurtosis = float((deviations**4).mean()) / simulated_variance**2

    assert simulated_mean == pytest.approx(analytic.mean, abs=5e-4)
    assert simulated_variance == pytest.approx(analytic.variance, rel=0.05)
    assert simulated_skewness == pytest.approx(analytic.skewness, rel=0.15)
    assert simulated_kurtosis == pytest.approx(analytic.kurtosis, rel=0.25)


# ------------------------------------------------------------- invariantes
def test_la_condicion_de_martingala_se_impone_en_el_constructor() -> None:
    """Una mezcla que no respeta la martingala no es neutral al riesgo.

    Sin esta comprobacion, la densidad generaria precios donde la paridad
    put-call no se cumple, y BKM devolveria momentos incorrectos por una
    razon que no tiene nada que ver con BKM.
    """
    with pytest.raises(DomainValidationError, match="martingala"):
        LognormalMixtureRiskNeutralDensity(
            spot_price=SPOT_PRICE,
            forward_price=FORWARD_PRICE,
            time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
            components=(
                LognormalMixtureComponent(weight=0.5, forward_ratio=0.9, annualized_volatility=0.2),
                LognormalMixtureComponent(weight=0.5, forward_ratio=0.9, annualized_volatility=0.2),
            ),
        )


def test_los_pesos_deben_sumar_uno() -> None:
    with pytest.raises(DomainValidationError, match="sumar uno"):
        LognormalMixtureRiskNeutralDensity(
            spot_price=SPOT_PRICE,
            forward_price=FORWARD_PRICE,
            time_to_expiry_years=TIME_TO_EXPIRY_YEARS,
            components=(
                LognormalMixtureComponent(weight=0.4, forward_ratio=1.0, annualized_volatility=0.2),
            ),
        )


@pytest.mark.parametrize("strike_price", [3000.0, 4000.0, 5000.0, 6000.0, 8000.0])
def test_la_paridad_put_call_es_exacta_en_toda_la_rejilla(strike_price: float) -> None:
    """Es la propiedad que hace utilizable la cadena sintetica.

    El pipeline extrae el forward de la paridad. Si los datos sinteticos solo
    la cumplieran aproximadamente, el test de esa extraccion mediria el error
    de la densidad en vez del error del extractor.
    """
    density = build_crash_density()
    assert density.put_call_parity_residual(strike_price) == pytest.approx(0.0, abs=1e-9)


def test_la_desigualdad_entre_curtosis_y_asimetria_se_verifica() -> None:
    """kurtosis >= skewness^2 + 1 vale para cualquier distribucion.

    Es una comprobacion gratuita que atrapa la mayoria de los errores de
    signo y de normalizacion, y por eso vive dentro del propio tipo.
    """
    with pytest.raises(DomainValidationError, match="desigualdad"):
        LogReturnMoments(mean=0.0, variance=0.01, skewness=-3.0, kurtosis=5.0)


def test_los_momentos_rechazan_una_varianza_no_positiva() -> None:
    with pytest.raises(DomainValidationError, match="positiva"):
        LogReturnMoments(mean=0.0, variance=0.0, skewness=0.0, kurtosis=3.0)
