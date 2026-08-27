"""Enumeraciones del dominio de opciones.

Cada una de estas enumeraciones representa una caracteristica del contrato
que **cambia la matematica**, no una etiqueta descriptiva:

- `OptionType` decide el signo del payoff y por tanto que rama de la paridad
  put-call aplica.
- `ExerciseStyle` decide si BKM es aplicable. El resultado de spanning
  estatico de Bakshi y Madan (2000) sobre el que se apoya BKM replica
  payoffs con opciones **europeas**; aplicar la integral sobre primas
  americanas incorpora el premio de ejercicio anticipado a los momentos
  extraidos, con sesgo sistematico mayor en puts.
- `SettlementConvention` decide la hora exacta de expiracion y, con ella, el
  tiempo a vencimiento. Confundir AM con PM mueve el tiempo medio dia, que
  a 30 dias es un error de 1.7 % en T y se propaga a la volatilidad.

Son enumeraciones de texto (`StrEnum`) para que sobrevivan legibles a un
parquet y a un CSV sin necesitar un mapa de traduccion.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum


class OptionType(StrEnum):
    """Direccion del payoff de la opcion."""

    CALL = "CALL"
    PUT = "PUT"

    @property
    def payoff_sign(self) -> int:
        """Signo omega de la convencion habitual: +1 para call, -1 para put.

        Permite escribir el payoff como `max(sign * (spot - strike), 0)` y la
        paridad put-call sin ramificar con `if`.
        """
        return 1 if self is OptionType.CALL else -1

    @property
    def opposite(self) -> OptionType:
        """El tipo contrario. Util para emparejar call y put del mismo strike."""
        return OptionType.PUT if self is OptionType.CALL else OptionType.CALL


class ExerciseStyle(StrEnum):
    """Cuando puede ejercerse la opcion.

    `AMERICAN` no es un caso mas: bloquea el motor BKM salvo que se haya
    aplicado de-americanizacion explicita.
    """

    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"

    @property
    def supports_model_free_moment_extraction(self) -> bool:
        """True solo si la extraccion libre de modelo es valida sin conversion previa."""
        return self is ExerciseStyle.EUROPEAN


class SettlementConvention(StrEnum):
    """Momento del dia en que se fija el valor de liquidacion.

    `MORNING` corresponde a los contratos AM-settled, cuyo valor de
    liquidacion se calcula con los precios de apertura del dia de expiracion:
    la opcion deja de existir a las 09:30 hora de Nueva York. `AFTERNOON`
    corresponde a los PM-settled, que expiran al cierre.

    En el complejo SPX conviven ambos: las series estandar del tercer viernes
    son AM-settled y las semanales SPXW son PM-settled. Las dos son de
    ejercicio europeo y liquidacion en efectivo, asi que las dos sirven para
    BKM, pero **no comparten hora de expiracion**.
    """

    MORNING = "AM"
    AFTERNOON = "PM"

    @property
    def settlement_time_of_day(self) -> dt.time:
        """Hora de liquidacion en la zona horaria del mercado."""
        if self is SettlementConvention.MORNING:
            return dt.time(hour=9, minute=30)
        return dt.time(hour=16, minute=0)


class MoneynessConvention(StrEnum):
    """Como se mide la distancia entre el strike y el nivel de referencia.

    El proyecto usa `LOG_STRIKE_OVER_FORWARD` de forma predeterminada porque
    es adimensional, simetrica alrededor del dinero y comparable entre
    vencimientos con distinto nivel de forward. Las otras dos existen para
    poder leer superficies de proveedores que las usan.
    """

    LOG_STRIKE_OVER_FORWARD = "LOG_STRIKE_OVER_FORWARD"
    STRIKE_OVER_SPOT = "STRIKE_OVER_SPOT"
    BLACK_SCHOLES_DELTA = "BLACK_SCHOLES_DELTA"
