"""Value objects: unidades y convenciones hechas tipo.

Aqui viven las enumeraciones de convencion ademas de los tipos numericos. La
razon es de modelado: `SettlementConvention` u `OptionType` no son entidades
con identidad propia y ciclo de vida, son **caracteristicas inmutables que
describen una convencion**. Ese es exactamente el criterio que distingue un
value object de una entidad.

Tiene ademas una consecuencia practica que se descubrio construyendo: como
`TimeToExpiry` necesita la convencion de liquidacion para calcular la hora
exacta de expiracion, tener las enumeraciones en `model/` creaba un ciclo de
importacion, porque `model/` a su vez depende de los value objects. Con el
modelado correcto la dependencia va en una sola direccion: `model` usa
`values`, nunca al reves. El ciclo era el sintoma; el modelado equivocado, la
causa.
"""

from finvex.domain.values.as_of import AsOf, AsOfLike
from finvex.domain.values.enumerations import (
    ExerciseStyle,
    MoneynessConvention,
    OptionType,
    SettlementConvention,
)
from finvex.domain.values.time_to_expiry import (
    MARKET_CLOSE_TIME,
    MARKET_TIMEZONE,
    MINUTES_PER_YEAR,
    TimeToExpiry,
)

__all__ = [
    "MARKET_CLOSE_TIME",
    "MARKET_TIMEZONE",
    "MINUTES_PER_YEAR",
    "AsOf",
    "AsOfLike",
    "ExerciseStyle",
    "MoneynessConvention",
    "OptionType",
    "SettlementConvention",
    "TimeToExpiry",
]
