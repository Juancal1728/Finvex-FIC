"""`TimeToExpiry`: el tiempo a vencimiento con la convencion de minutos.

Este value object existe por una razon concreta y comprobable: el tiempo a
vencimiento es la fuente de error silencioso mas frecuente al replicar
indices de volatilidad. La metodologia del VIX de Cboe **no** mide el tiempo
en dias calendario sobre 365, sino en **minutos sobre 525 600**, contando
hasta la hora exacta de liquidacion de cada serie.

La diferencia no es cosmetica. Para una serie que expira en 30 dias, tratar
la expiracion como "fin del dia" cuando en realidad liquida a las 09:30
introduce un error de aproximadamente medio dia sobre treinta, es decir un
1.7 % en T. La volatilidad implicita escala como la raiz de T, de modo que
el sesgo resultante es del orden de 0.8 % en volatilidad: suficiente para
que una replica del VIX difiera en decimas de punto y para que uno pierda
dias buscando el error en la integral, que esta bien.

Al hacer del tiempo un tipo propio, la conversion ocurre en un solo lugar y
queda imposible pasar "dias" donde se esperaban "anos".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from finvex.domain.errors import DomainValidationError
from finvex.domain.values.enumerations import SettlementConvention

MINUTES_PER_YEAR: int = 525_600
"""Minutos en un ano de 365 dias. Es la constante que usa Cboe, no 525 949."""

MARKET_TIMEZONE: ZoneInfo = ZoneInfo("America/New_York")
"""Zona horaria de referencia de los mercados de opciones de Estados Unidos."""

MARKET_CLOSE_TIME: dt.time = dt.time(hour=16, minute=0)
"""Hora de cierre. Es el momento de valuacion de una observacion de fin de dia."""


@dataclass(frozen=True, order=True, slots=True)
class TimeToExpiry:
    """Tiempo hasta el vencimiento, medido en minutos enteros.

    Se guarda en minutos y no en anos porque los minutos son la unidad
    natural del calendario de contratos: son exactos, enteros y no acumulan
    error de punto flotante al sumarse. La conversion a anos ocurre una sola
    vez, en la propiedad `years`.
    """

    minutes: int

    def __post_init__(self) -> None:
        if self.minutes <= 0:
            raise DomainValidationError(
                f"El tiempo a vencimiento debe ser positivo, se recibieron {self.minutes} minutos. "
                "Una serie ya vencida no puede entrar al calculo de momentos."
            )

    @classmethod
    def between_moments(
        cls,
        valuation_moment: dt.datetime,
        expiration_moment: dt.datetime,
    ) -> TimeToExpiry:
        """Construye el tiempo a vencimiento entre dos instantes con zona horaria.

        Ambos instantes deben ser conscientes de su zona horaria. Un
        `datetime` ingenuo se rechaza en vez de asumirsele una zona, porque
        la suposicion silenciosa es precisamente lo que produce errores de
        una hora dos veces al ano, cuando cambia el horario de verano.
        """
        moments_to_check = (
            ("valuation_moment", valuation_moment),
            ("expiration_moment", expiration_moment),
        )
        for name, moment in moments_to_check:
            if moment.tzinfo is None:
                raise DomainValidationError(
                    f"{name} debe tener zona horaria explicita. "
                    "Un datetime ingenuo hace ambiguo el cambio de horario de verano."
                )
        elapsed = expiration_moment - valuation_moment
        return cls(minutes=round(elapsed.total_seconds() / 60.0))

    @classmethod
    def from_expiration_date(
        cls,
        valuation_date: dt.date,
        expiration_date: dt.date,
        settlement: SettlementConvention,
        valuation_time: dt.time = MARKET_CLOSE_TIME,
    ) -> TimeToExpiry:
        """Construye el tiempo a vencimiento a partir de fechas y convencion de liquidacion.

        Es el constructor que se usa con datos de fin de dia: la observacion
        se toma al cierre de `valuation_date` y la serie expira a la hora que
        dicta su convencion de liquidacion.
        """
        valuation_moment = dt.datetime.combine(
            valuation_date, valuation_time, tzinfo=MARKET_TIMEZONE
        )
        expiration_moment = dt.datetime.combine(
            expiration_date, settlement.settlement_time_of_day, tzinfo=MARKET_TIMEZONE
        )
        return cls.between_moments(valuation_moment, expiration_moment)

    @property
    def years(self) -> float:
        """Tiempo a vencimiento en anos, con la convencion de minutos sobre 525 600."""
        return self.minutes / MINUTES_PER_YEAR

    @property
    def calendar_days(self) -> float:
        """Dias calendario equivalentes. Solo para reportar; nunca para calcular."""
        return self.minutes / (60.0 * 24.0)

    def __str__(self) -> str:
        return f"{self.calendar_days:.2f} dias ({self.years:.6f} anos)"
