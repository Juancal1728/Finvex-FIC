"""Proveedor sintetico: un mercado de opciones con la respuesta conocida.

## Por que este adaptador se construye primero

El proyecto tiene una dependencia externa sin resolver —la fuente de datos de
opciones— y una pieza critica que validar —el estimador de momentos libres de
modelo—. Este adaptador rompe esa dependencia: genera cadenas de opciones a
partir de una densidad neutral al riesgo cuyos momentos se conocen en forma
cerrada, de modo que todo el motor de extraccion puede construirse y validarse
sin un solo dato real.

Y hace algo mas valioso todavia. Cuando la implementacion de BKM falle contra
esta cadena, el culpable esta acotado a una sola pieza: los precios son
exactos, no hay ruido de microestructura, no hay huecos de liquidez y no hay
cotizaciones rancias. Comparado con la replica del VIX —donde un desajuste
puede venir de la integral, de la superficie, de los filtros o del tiempo—
esta es una validacion que senala al culpable en vez de anunciar que hay uno.

## Que tan realista es el mercado que genera

Lo suficiente para ejercitar las partes que importan, sin pretender ser una
simulacion de microestructura:

- **Trayectoria del spot** con un movimiento browniano geometrico de semilla
  fija, para que las fechas de observacion no sean todas identicas.
- **Forward distinto del spot**, por tasa y por rendimiento de dividendos.
  Esto es esencial: si el forward coincidiera con el spot, la extraccion por
  paridad put-call se validaria trivialmente y no probaria nada.
- **Dos convenciones de liquidacion conviviendo**, igual que en el complejo
  SPX real: vencimientos del tercer viernes con liquidacion AM bajo la raiz
  `SPX`, y vencimientos semanales con liquidacion PM bajo la raiz `SPXW`.
  Asi el manejo de la hora de expiracion queda ejercitado desde el principio.
- **Rejilla de strikes de paso variable**, fina cerca del dinero y gruesa en
  las alas, que es como cotizan los mercados reales de indices.
- **Spread proporcional al precio**, mas ancho a medida que el strike se
  aleja del dinero. Al ser proporcional, el punto medio coincide exactamente
  con el precio teorico, lo cual mantiene la paridad exacta por construccion.

## Lo que deliberadamente NO simula

No hay cotizaciones rancias, ni violaciones de arbitraje, ni huecos, ni
sesgos de redondeo. Esas patologias son el objeto del motor de calidad de la
fase 2, y se inyectaran alli de forma controlada para probar cada filtro por
separado. Mezclarlas aqui haria que el checkpoint de momentos midiera dos
cosas a la vez.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

from finvex.domain.errors import DomainValidationError, ProviderUnavailableError
from finvex.domain.model.option_chain import OptionChain
from finvex.domain.model.option_quote import OptionQuote
from finvex.domain.model.rate_curve import RateCurve
from finvex.domain.model.risk_neutral_density import LognormalMixtureRiskNeutralDensity
from finvex.domain.options.black_scholes import implied_volatility_from_price
from finvex.domain.values.as_of import AsOf
from finvex.domain.values.enumerations import ExerciseStyle, OptionType, SettlementConvention
from finvex.domain.values.time_to_expiry import TimeToExpiry

MINIMUM_QUOTABLE_PRICE: float = 0.05
"""Precio minimo cotizable, en puntos de indice. Por debajo, la serie no se publica."""


@dataclass(frozen=True, slots=True)
class SyntheticMarketConfiguration:
    """Parametros del mercado sintetico.

    Los valores predeterminados describen un indice de renta variable
    parecido al S&P 500 y estan calibrados para reproducir un perfil de
    momentos implicitos realista. A treinta dias producen:

    - volatilidad implicita del log-retorno cercana al 18 % anualizado,
    - asimetria implicita alrededor de -1.5,
    - curtosis implicita alrededor de 8.7.

    Esas cifras estan en el rango que reporta la literatura para opciones
    sobre el S&P 500, y eso importa: un mercado sintetico con asimetria de
    -4 y curtosis de 24 validaria el estimador en una region que los datos
    reales nunca visitan, y podria esconder problemas de precision
    justamente en el rango que si interesa.

    La mezcla tiene asimetria negativa y curtosis muy por encima de tres, que
    es el patron que hace interesante el problema: si la distribucion fuera
    normal, MFIS y MFIK no aportarian informacion alguna sobre la que
    construir una tesis.
    """

    underlying_symbol: str = "SPX"
    monthly_contract_root: str = "SPX"
    weekly_contract_root: str = "SPXW"

    first_observation_date: dt.date = dt.date(2024, 1, 2)
    number_of_observation_dates: int = 60

    initial_spot_price: float = 5000.0
    spot_annualized_drift: float = 0.06
    spot_annualized_volatility: float = 0.15
    continuously_compounded_rate: float = 0.04
    dividend_yield: float = 0.015

    base_volatility: float = 0.14
    crash_probability: float = 0.05
    crash_price_drop: float = 0.12
    crash_volatility: float = 0.30

    minimum_days_to_expiry: int = 7
    maximum_days_to_expiry: int = 120
    base_strike_increment: float = 5.0
    wide_strike_increment_multiple: int = 5
    near_the_money_band_in_standard_deviations: float = 1.5
    strike_range_below_in_standard_deviations: float = 8.0
    strike_range_above_in_standard_deviations: float = 4.0

    minimum_relative_half_spread: float = 0.005
    relative_half_spread_slope: float = 0.05
    maximum_relative_half_spread: float = 0.25

    compute_vendor_implied_volatility: bool = True
    random_seed: int = 20260826

    provider_version: str = "synthetic-1.0"

    def __post_init__(self) -> None:
        if self.number_of_observation_dates <= 0:
            raise DomainValidationError("Se necesita al menos una fecha de observacion.")
        if self.minimum_days_to_expiry >= self.maximum_days_to_expiry:
            raise DomainValidationError(
                "El plazo minimo debe ser menor que el maximo: "
                f"{self.minimum_days_to_expiry} >= {self.maximum_days_to_expiry}."
            )


def _is_third_friday(candidate_date: dt.date) -> bool:
    """True si la fecha es el tercer viernes del mes.

    Es la regla de expiracion de las series estandar de opciones sobre
    indices en Estados Unidos. Se implementa por aritmetica de calendario y
    no con una tabla: el tercer viernes es, por definicion, el viernes cuyo
    dia del mes cae entre 15 y 21.
    """
    return candidate_date.weekday() == 4 and 15 <= candidate_date.day <= 21


class SyntheticOptionsDataProvider:
    """Adaptador que fabrica un mercado de opciones con momentos conocidos."""

    def __init__(self, configuration: SyntheticMarketConfiguration | None = None) -> None:
        self._configuration = configuration or SyntheticMarketConfiguration()
        self._observation_dates = self._build_observation_dates()
        self._spot_price_by_date = self._simulate_spot_price_path()
        self._expiration_dates = self._build_expiration_schedule()

    # ------------------------------------------------------- puerto publico
    @property
    def provider_name(self) -> str:
        return self._configuration.provider_version

    @property
    def configuration(self) -> SyntheticMarketConfiguration:
        return self._configuration

    def is_available(self) -> bool:
        """Siempre disponible: no depende de credenciales ni de red."""
        return True

    def available_underlying_symbols(self) -> tuple[str, ...]:
        return (self._configuration.underlying_symbol,)

    def available_observation_dates(self, underlying_symbol: str) -> tuple[dt.date, ...]:
        self._assert_known_underlying(underlying_symbol)
        return self._observation_dates

    def rate_curve(self, as_of: AsOf) -> RateCurve:
        """Curva plana. El proyecto no estudia la estructura temporal de tasas."""
        del as_of  # la curva sintetica no depende de la fecha
        return RateCurve.flat(self._configuration.continuously_compounded_rate)

    def option_chain(self, as_of: AsOf, underlying_symbol: str) -> OptionChain:
        """Cadena completa del subyacente en esa fecha de observacion."""
        self._assert_known_underlying(underlying_symbol)
        spot_price = self.spot_price_on(as_of)
        quotes: list[OptionQuote] = []
        for expiration_date in self.expiration_dates_available_on(as_of):
            quotes.extend(self._build_quotes_for_expiration(as_of, expiration_date, spot_price))
        return OptionChain(
            as_of=as_of,
            underlying_symbol=underlying_symbol,
            underlying_price=spot_price,
            quotes=tuple(quotes),
        )

    # --------------------------------------------- capacidades del sintetico
    def spot_price_on(self, as_of: AsOf) -> float:
        """Precio de contado en esa fecha de observacion."""
        try:
            return self._spot_price_by_date[as_of.date]
        except KeyError as error:
            raise ProviderUnavailableError(
                f"El mercado sintetico no cubre la fecha {as_of}. Rango disponible: "
                f"{self._observation_dates[0]} a {self._observation_dates[-1]}."
            ) from error

    def expiration_dates_available_on(self, as_of: AsOf) -> tuple[dt.date, ...]:
        """Vencimientos cotizados en esa fecha, dentro de la ventana de plazos."""
        configuration = self._configuration
        return tuple(
            expiration_date
            for expiration_date in self._expiration_dates
            if configuration.minimum_days_to_expiry
            <= (expiration_date - as_of.date).days
            <= configuration.maximum_days_to_expiry
        )

    def risk_neutral_density_for(
        self, as_of: AsOf, expiration_date: dt.date
    ) -> LognormalMixtureRiskNeutralDensity:
        """La densidad **verdadera** que genero esa cadena.

        Este es el metodo que convierte al proveedor sintetico en una
        herramienta de validacion y no solo en un generador de datos de
        relleno. Devuelve el objeto del que salen los precios, con sus
        momentos exactos: es el valor de referencia del checkpoint V1.

        No forma parte del puerto `OptionsDataProvider`, y eso es
        intencional. Ningun proveedor real puede ofrecerlo, asi que ningun
        modulo de investigacion debe poder depender de el. Solo lo usan los
        tests.
        """
        configuration = self._configuration
        settlement = self._settlement_for(expiration_date)
        time_to_expiry = TimeToExpiry.from_expiration_date(
            valuation_date=as_of.date,
            expiration_date=expiration_date,
            settlement=settlement,
        )
        spot_price = self.spot_price_on(as_of)
        forward_price = spot_price * math.exp(
            (configuration.continuously_compounded_rate - configuration.dividend_yield)
            * time_to_expiry.years
        )
        discount_factor = math.exp(
            -configuration.continuously_compounded_rate * time_to_expiry.years
        )
        return LognormalMixtureRiskNeutralDensity.from_crash_scenario(
            spot_price=spot_price,
            forward_price=forward_price,
            time_to_expiry_years=time_to_expiry.years,
            base_volatility=configuration.base_volatility,
            crash_probability=configuration.crash_probability,
            crash_price_drop=configuration.crash_price_drop,
            crash_volatility=configuration.crash_volatility,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------- internos
    def _assert_known_underlying(self, underlying_symbol: str) -> None:
        if underlying_symbol != self._configuration.underlying_symbol:
            raise ProviderUnavailableError(
                f"El mercado sintetico solo cubre {self._configuration.underlying_symbol}, "
                f"se pidio {underlying_symbol}."
            )

    def _build_observation_dates(self) -> tuple[dt.date, ...]:
        """Dias habiles consecutivos desde la fecha inicial.

        Se omiten sabados y domingos pero no los feriados de mercado. Para el
        proposito de este proveedor —ejercitar el pipeline— la diferencia es
        irrelevante, y arrastrar un calendario de feriados aqui seria
        complejidad sin retorno. Queda documentado para que nadie lo
        confunda con un calendario real.
        """
        dates: list[dt.date] = []
        candidate = self._configuration.first_observation_date
        while len(dates) < self._configuration.number_of_observation_dates:
            if candidate.weekday() < 5:
                dates.append(candidate)
            candidate += dt.timedelta(days=1)
        return tuple(dates)

    def _simulate_spot_price_path(self) -> dict[dt.date, float]:
        """Trayectoria del spot con un movimiento browniano geometrico.

        El generador se crea a partir de la semilla de la configuracion y se
        usa una sola vez, en la construccion. Asi dos proveedores con la
        misma configuracion producen exactamente el mismo mercado, que es la
        condicion para que un test sea reproducible.
        """
        configuration = self._configuration
        generator = np.random.default_rng(configuration.random_seed)
        number_of_steps = len(self._observation_dates)
        time_step_in_years = 1.0 / 252.0
        shocks = generator.standard_normal(number_of_steps)
        drift_term = (
            configuration.spot_annualized_drift - 0.5 * configuration.spot_annualized_volatility**2
        ) * time_step_in_years
        diffusion_term = configuration.spot_annualized_volatility * math.sqrt(time_step_in_years)
        log_prices = np.log(configuration.initial_spot_price) + np.cumsum(
            drift_term + diffusion_term * shocks
        )
        prices = np.exp(log_prices)
        return {
            observation_date: float(price)
            for observation_date, price in zip(self._observation_dates, prices, strict=True)
        }

    def _build_expiration_schedule(self) -> tuple[dt.date, ...]:
        """Todos los viernes que caen dentro del horizonte cubierto.

        Los viernes que son terceros del mes seran series estandar con
        liquidacion AM; los demas, semanales con liquidacion PM.
        """
        first_date = self._observation_dates[0]
        last_date = self._observation_dates[-1] + dt.timedelta(
            days=self._configuration.maximum_days_to_expiry
        )
        expirations: list[dt.date] = []
        candidate = first_date
        while candidate <= last_date:
            if candidate.weekday() == 4:
                expirations.append(candidate)
            candidate += dt.timedelta(days=1)
        return tuple(expirations)

    def _settlement_for(self, expiration_date: dt.date) -> SettlementConvention:
        return (
            SettlementConvention.MORNING
            if _is_third_friday(expiration_date)
            else SettlementConvention.AFTERNOON
        )

    def _contract_root_for(self, expiration_date: dt.date) -> str:
        return (
            self._configuration.monthly_contract_root
            if _is_third_friday(expiration_date)
            else self._configuration.weekly_contract_root
        )

    def _strike_grid(self, forward_price: float, total_volatility: float) -> tuple[float, ...]:
        """Rejilla de strikes de paso variable alrededor del forward.

        Cerca del dinero el paso es el incremento base; mas alla de la banda
        cercana el paso se multiplica. Reproduce como cotizan los mercados
        reales de opciones sobre indices y, de paso, mantiene manejable el
        numero de contratos por vencimiento.

        Todos los strikes se alinean a multiplos del incremento base, igual
        que en un mercado real: los strikes no son numeros arbitrarios.
        """
        configuration = self._configuration
        increment = configuration.base_strike_increment
        wide_increment = increment * configuration.wide_strike_increment_multiple

        # El rango es asimetrico a proposito: los mercados de opciones sobre
        # indices cotizan strikes mucho mas abajo que arriba, porque la
        # demanda de proteccion contra caidas es lo que sostiene esas series.
        # Reproducirlo importa porque la asimetria implicita se estima
        # justamente con la informacion del ala izquierda.
        lowest = forward_price * math.exp(
            -configuration.strike_range_below_in_standard_deviations * total_volatility
        )
        highest = forward_price * math.exp(
            configuration.strike_range_above_in_standard_deviations * total_volatility
        )
        near_lowest = forward_price * math.exp(
            -configuration.near_the_money_band_in_standard_deviations * total_volatility
        )
        near_highest = forward_price * math.exp(
            configuration.near_the_money_band_in_standard_deviations * total_volatility
        )

        def aligned_range(start: float, stop: float, step: float) -> list[float]:
            first_multiple = math.ceil(start / step)
            last_multiple = math.floor(stop / step)
            return [multiple * step for multiple in range(first_multiple, last_multiple + 1)]

        strikes = set(aligned_range(near_lowest, near_highest, increment))
        strikes.update(aligned_range(lowest, near_lowest, wide_increment))
        strikes.update(aligned_range(near_highest, highest, wide_increment))
        return tuple(sorted(strike for strike in strikes if strike > 0.0))

    def _relative_half_spread(self, log_moneyness: float) -> float:
        """Medio spread como fraccion del precio, creciente al alejarse del dinero.

        Que sea **proporcional al precio** no es un detalle: hace que el
        punto medio coincida exactamente con el precio teorico, de modo que
        la cadena satisface la paridad put-call de forma exacta y los tests
        de extraccion del forward miden el extractor y no el ruido.
        """
        configuration = self._configuration
        half_spread = (
            configuration.minimum_relative_half_spread
            + configuration.relative_half_spread_slope * abs(log_moneyness)
        )
        return min(half_spread, configuration.maximum_relative_half_spread)

    def _build_quotes_for_expiration(
        self, as_of: AsOf, expiration_date: dt.date, spot_price: float
    ) -> list[OptionQuote]:
        configuration = self._configuration
        settlement = self._settlement_for(expiration_date)
        contract_root = self._contract_root_for(expiration_date)
        density = self.risk_neutral_density_for(as_of, expiration_date)
        time_to_expiry_years = density.time_to_expiry_years
        total_volatility = configuration.base_volatility * math.sqrt(time_to_expiry_years)

        quotes: list[OptionQuote] = []
        for strike_price in self._strike_grid(density.forward_price, total_volatility):
            log_moneyness = math.log(strike_price / density.forward_price)
            half_spread = self._relative_half_spread(log_moneyness)
            for option_type in (OptionType.CALL, OptionType.PUT):
                theoretical_price = density.european_option_price(
                    strike_price, is_call=option_type is OptionType.CALL
                )
                if theoretical_price < MINIMUM_QUOTABLE_PRICE:
                    continue
                quotes.append(
                    self._build_single_quote(
                        as_of=as_of,
                        contract_root=contract_root,
                        option_type=option_type,
                        strike_price=strike_price,
                        expiration_date=expiration_date,
                        settlement=settlement,
                        spot_price=spot_price,
                        theoretical_price=theoretical_price,
                        half_spread=half_spread,
                        forward_price=density.forward_price,
                        time_to_expiry_years=time_to_expiry_years,
                        discount_factor=density.discount_factor,
                    )
                )
        return quotes

    def _build_single_quote(
        self,
        *,
        as_of: AsOf,
        contract_root: str,
        option_type: OptionType,
        strike_price: float,
        expiration_date: dt.date,
        settlement: SettlementConvention,
        spot_price: float,
        theoretical_price: float,
        half_spread: float,
        forward_price: float,
        time_to_expiry_years: float,
        discount_factor: float,
    ) -> OptionQuote:
        vendor_implied_volatility: float | None = None
        if self._configuration.compute_vendor_implied_volatility:
            vendor_implied_volatility = implied_volatility_from_price(
                option_price=theoretical_price,
                option_type=option_type,
                forward_price=forward_price,
                strike_price=strike_price,
                time_to_expiry_years=time_to_expiry_years,
                discount_factor=discount_factor,
            )
        return OptionQuote(
            as_of=as_of,
            underlying_symbol=self._configuration.underlying_symbol,
            contract_root=contract_root,
            option_type=option_type,
            strike_price=strike_price,
            expiration_date=expiration_date,
            exercise_style=ExerciseStyle.EUROPEAN,
            settlement_convention=settlement,
            underlying_price=spot_price,
            data_source=self.provider_name,
            bid_price=theoretical_price * (1.0 - half_spread),
            ask_price=theoretical_price * (1.0 + half_spread),
            traded_volume=None,
            open_interest=None,
            vendor_implied_volatility=vendor_implied_volatility,
        )
