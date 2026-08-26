"""`AsOf`: la fecha de formacion de informacion, hecha tipo.

Este es el cimiento del invariante I1 del blueprint. Ningun modulo del
proyecto lee "hoy": todos reciben un `AsOf` explicito. Convertir la fecha en
un tipo propio, y no en un `date` suelto, hace que el compilador de tipos y
el lector distingan entre "la fecha en que se formo la decision" y cualquier
otra fecha que ande por el codigo (vencimiento, fecha de liquidacion, fecha
de un retorno realizado). Confundirlas es la forma mas comun de look-ahead.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Union

from finvex.domain.errors import DomainValidationError

AsOfLike = Union[str, dt.date, dt.datetime, "AsOf"]


@dataclass(frozen=True, order=True, slots=True)
class AsOf:
    """Fecha de corte de informacion.

    Semantica: un `AsOf(d)` autoriza a ver todo lo observable **hasta el
    cierre de d, inclusive**, y nada mas. La frontera es explicita en
    `covers()` para que no dependa de la convencion de cada modulo.
    """

    date: dt.date

    def __post_init__(self) -> None:
        if not isinstance(self.date, dt.date) or isinstance(self.date, dt.datetime):
            raise DomainValidationError(
                f"AsOf requiere datetime.date, recibio {type(self.date).__name__}. "
                "Un datetime traeria una hora implicita y la frontera dejaria de ser clara."
            )

    @classmethod
    def parse(cls, value: AsOfLike) -> AsOf:
        """Construye un AsOf desde str ISO, date, datetime o AsOf."""
        if isinstance(value, AsOf):
            return value
        if isinstance(value, dt.datetime):
            return cls(value.date())
        if isinstance(value, dt.date):
            return cls(value)
        if isinstance(value, str):
            try:
                return cls(dt.date.fromisoformat(value))
            except ValueError as exc:
                raise DomainValidationError(
                    f"Fecha ISO invalida: {value!r}. Se espera YYYY-MM-DD."
                ) from exc
        raise DomainValidationError(f"No se puede interpretar {value!r} como AsOf.")

    def covers(self, moment: dt.date | dt.datetime) -> bool:
        """True si `moment` era observable a esta fecha de formacion."""
        day = moment.date() if isinstance(moment, dt.datetime) else moment
        return day <= self.date

    def shift_days(self, days: int) -> AsOf:
        return AsOf(self.date + dt.timedelta(days=days))

    @property
    def iso(self) -> str:
        return self.date.isoformat()

    def __str__(self) -> str:
        return self.iso
