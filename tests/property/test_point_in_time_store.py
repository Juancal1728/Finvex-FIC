"""La frontera de informacion, verificada sobre entradas generadas.

Un test de ejemplo prueba que el store funciona en el caso que se le ocurrio
a quien lo escribio. Un test de propiedad prueba que funciona en los casos
que no se le ocurrieron, que es donde vive el look-ahead.
"""

from __future__ import annotations

import datetime as dt

import pytest
from hypothesis import given, settings
from hypothesis import strategies as strategy

from finvex.adapters.data import SyntheticMarketConfiguration, SyntheticOptionsDataProvider
from finvex.application import PointInTimeMarketDataStore
from finvex.domain.errors import LookAheadError
from finvex.domain.model import OptionChain, RateCurve
from finvex.domain.values import AsOf

TEST_CONFIGURATION = SyntheticMarketConfiguration(
    number_of_observation_dates=10,
    maximum_days_to_expiry=45,
    compute_vendor_implied_volatility=False,
)
SHARED_PROVIDER = SyntheticOptionsDataProvider(TEST_CONFIGURATION)
SHARED_STORE = PointInTimeMarketDataStore(SHARED_PROVIDER)
AVAILABLE_DATES = SHARED_PROVIDER.available_observation_dates("SPX")


@given(index_of_date=strategy.integers(min_value=0, max_value=len(AVAILABLE_DATES) - 1))
@settings(max_examples=10, deadline=None)
def test_la_cadena_devuelta_nunca_contiene_informacion_posterior_al_corte(
    index_of_date: int,
) -> None:
    """Para cualquier fecha de corte, nada de lo devuelto la excede."""
    as_of = AsOf(AVAILABLE_DATES[index_of_date])
    chain = SHARED_STORE.option_chain(as_of, "SPX")

    assert chain.as_of == as_of
    for quote in chain:
        assert quote.as_of == as_of
        assert quote.expiration_date > as_of.date


@given(index_of_date=strategy.integers(min_value=0, max_value=len(AVAILABLE_DATES) - 1))
@settings(max_examples=10, deadline=None)
def test_las_fechas_de_entrenamiento_disponibles_nunca_incluyen_el_futuro(
    index_of_date: int,
) -> None:
    """Es el metodo que usa cualquier modelo para construir su ventana.

    Devolver aqui las fechas ya filtradas evita que cada modelo tenga que
    acordarse de filtrarlas, que es exactamente donde se cuela el error.
    """
    as_of = AsOf(AVAILABLE_DATES[index_of_date])
    dates_up_to_cutoff = SHARED_STORE.observation_dates_up_to(as_of, "SPX")

    assert all(date <= as_of.date for date in dates_up_to_cutoff)
    assert len(dates_up_to_cutoff) == index_of_date + 1


class ProviderThatReturnsTheWrongDate:
    """Doble de prueba que simula el fallo mas peligroso de un adaptador.

    Un proveedor real puede devolver "la ultima cotizacion disponible" en vez
    de la de la fecha pedida, o traer un archivo con las fechas corridas un
    dia. No es una hipotesis: es un modo de fallo documentado. El store tiene
    que detectarlo, y para probar que lo detecta hace falta un proveedor que
    se comporte mal a proposito.
    """

    def __init__(self, honest_provider: SyntheticOptionsDataProvider) -> None:
        self._honest_provider = honest_provider

    @property
    def provider_name(self) -> str:
        return "proveedor-desalineado"

    def is_available(self) -> bool:
        return True

    def available_underlying_symbols(self) -> tuple[str, ...]:
        return self._honest_provider.available_underlying_symbols()

    def available_observation_dates(self, underlying_symbol: str) -> tuple[dt.date, ...]:
        return self._honest_provider.available_observation_dates(underlying_symbol)

    def option_chain(self, as_of: AsOf, underlying_symbol: str) -> OptionChain:
        """Ignora la fecha pedida y devuelve siempre una posterior."""
        del as_of
        later_date = AsOf(self._honest_provider.available_observation_dates(underlying_symbol)[-1])
        return self._honest_provider.option_chain(later_date, underlying_symbol)

    def rate_curve(self, as_of: AsOf) -> RateCurve:
        return self._honest_provider.rate_curve(as_of)


def test_el_store_detecta_un_proveedor_desalineado_en_el_tiempo() -> None:
    misbehaving_store = PointInTimeMarketDataStore(ProviderThatReturnsTheWrongDate(SHARED_PROVIDER))
    requested_date = AsOf(AVAILABLE_DATES[0])

    with pytest.raises(LookAheadError, match="devolvio la del"):
        misbehaving_store.option_chain(requested_date, "SPX")


def test_el_proveedor_sintetico_satisface_el_puerto() -> None:
    """Tipado estructural: cumple el contrato por tener los metodos, sin heredar."""
    from finvex.ports import OptionsDataProvider

    assert isinstance(SHARED_PROVIDER, OptionsDataProvider)
