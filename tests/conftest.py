"""Fixtures compartidas.

Casi ningun test del proyecto debe tocar datos reales. El mercado sintetico
cubre el resto, y se construye una sola vez por sesion porque generar la
cadena completa con inversion de volatilidad implicita cuesta segundos.
"""

from __future__ import annotations

import pytest
from numpy.random import Generator, default_rng

from finvex.adapters.data import SyntheticMarketConfiguration, SyntheticOptionsDataProvider
from finvex.application import PointInTimeMarketDataStore
from finvex.domain.values import AsOf

FIXED_TEST_SEED = 20260826


@pytest.fixture
def random_number_generator() -> Generator:
    """Generador con semilla fija: nunca aleatoriedad global en los tests."""
    return default_rng(FIXED_TEST_SEED)


@pytest.fixture(scope="session")
def fast_market_configuration() -> SyntheticMarketConfiguration:
    """Mercado pequeno para los tests que no necesitan volatilidad implicita.

    Apagar el calculo de volatilidad implicita del proveedor quita la
    inversion numerica de cada contrato, que es lo que domina el tiempo de
    generacion. Los tests que si la necesitan usan una configuracion propia.
    """
    return SyntheticMarketConfiguration(
        number_of_observation_dates=5,
        maximum_days_to_expiry=45,
        compute_vendor_implied_volatility=False,
        random_seed=FIXED_TEST_SEED,
    )


@pytest.fixture(scope="session")
def synthetic_provider(
    fast_market_configuration: SyntheticMarketConfiguration,
) -> SyntheticOptionsDataProvider:
    return SyntheticOptionsDataProvider(fast_market_configuration)


@pytest.fixture(scope="session")
def market_data_store(
    synthetic_provider: SyntheticOptionsDataProvider,
) -> PointInTimeMarketDataStore:
    return PointInTimeMarketDataStore(synthetic_provider)


@pytest.fixture(scope="session")
def first_observation_date(synthetic_provider: SyntheticOptionsDataProvider) -> AsOf:
    return AsOf(synthetic_provider.available_observation_dates("SPX")[0])
