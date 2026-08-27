"""`OptionQuote`: una cotizacion de opcion, con sus invariantes.

Este es el objeto de entrada del pipeline. Todo lo que el proyecto sabe del
mercado de opciones entra por aqui, asi que conviene entender tres decisiones
de diseno que no son obvias.

**Primera: el estilo de ejercicio y la convencion de liquidacion son
obligatorios.** No son metadatos descriptivos. El estilo de ejercicio decide
si BKM es aplicable, y la convencion de liquidacion decide la hora exacta de
expiracion y por tanto el tiempo a vencimiento. Un esquema que los deje
opcionales convierte una restriccion metodologica en una nota al pie que
nadie verifica.

**Segunda: el forward y la tasa no viven aqui.** Son propiedades de la pareja
(fecha de observacion, vencimiento), no del contrato individual. Ponerlos en
cada fila los duplicaria cientos de veces y abriria la puerta a que dos
contratos del mismo vencimiento llevaran forwards distintos: un error
silencioso que rompe la paridad put-call y, con ella, todo el calculo de
momentos. Viven en `ExpiryContext`.

**Tercera: el precio medio no se almacena, se calcula.** Guardar un valor
derivado junto a los valores de los que depende garantiza que algun dia se
desincronicen. La regla del proyecto es que todo lo derivable sea una
propiedad calculada.

Una advertencia sobre moneyness: la condicion de estar fuera del dinero que
usan BKM y la metodologia del VIX se mide respecto al **forward**, que es una
propiedad del vencimiento y no del contrato. Por eso esa pregunta no se puede
responder desde una cotizacion aislada y el metodo correspondiente vive en
`ExpiryContext`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from finvex.domain.errors import DomainValidationError
from finvex.domain.values.as_of import AsOf
from finvex.domain.values.enumerations import ExerciseStyle, OptionType, SettlementConvention


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """Cotizacion de fin de dia de un contrato de opcion.

    Atributos:
        as_of: fecha de observacion. Define que informacion estaba disponible.
        underlying_symbol: simbolo canonico interno del subyacente, por
            ejemplo `SPX`. No es el ticker del proveedor: la traduccion
            ocurre en el adaptador.
        contract_root: raiz del contrato, por ejemplo `SPX` o `SPXW`. Dos
            raices sobre el mismo subyacente pueden tener convenciones de
            liquidacion distintas, y por eso se guarda aparte del subyacente.
        option_type: call o put.
        strike_price: precio de ejercicio en puntos de indice.
        expiration_date: fecha de expiracion. La hora la determina
            `settlement_convention`.
        exercise_style: europeo o americano. Bloquea BKM si es americano.
        settlement_convention: liquidacion AM o PM.
        contract_multiplier: unidades de subyacente por contrato. Solo
            interviene si se calculan resultados en dinero; los precios se
            manejan en puntos de indice.
        bid_price: mejor postura de compra. Puede faltar.
        ask_price: mejor postura de venta. Puede faltar.
        traded_volume: contratos negociados en la sesion.
        open_interest: contratos abiertos al cierre.
        vendor_implied_volatility: volatilidad implicita que reporta el
            proveedor, en decimal anualizado. Se guarda para poder
            contrastarla con la que calcula el proyecto, nunca para usarla
            directamente: cada proveedor la calcula con supuestos propios de
            tasa, dividendos y ejercicio que no siempre documenta.
        underlying_price: precio de contado del subyacente al momento de la
            observacion, en puntos de indice.
        data_source: identificador del proveedor y su version.
        quality_flags: mascara de bits con los codigos de calidad. Cero
            significa que la cotizacion paso todos los filtros. Las
            cotizaciones **nunca se eliminan**, se marcan, de modo que el
            reporte de exclusiones y el analisis de sensibilidad a los
            filtros salen sin reprocesar nada.
    """

    as_of: AsOf
    underlying_symbol: str
    contract_root: str
    option_type: OptionType
    strike_price: float
    expiration_date: dt.date
    exercise_style: ExerciseStyle
    settlement_convention: SettlementConvention
    underlying_price: float
    data_source: str
    contract_multiplier: int = 100
    bid_price: float | None = None
    ask_price: float | None = None
    traded_volume: int | None = None
    open_interest: int | None = None
    vendor_implied_volatility: float | None = None
    quality_flags: int = field(default=0)

    def __post_init__(self) -> None:
        if self.strike_price <= 0.0:
            raise DomainValidationError(
                f"El strike debe ser positivo, se recibio {self.strike_price}."
            )
        if self.underlying_price <= 0.0:
            raise DomainValidationError(
                f"El precio del subyacente debe ser positivo, se recibio {self.underlying_price}."
            )
        if self.expiration_date <= self.as_of.date:
            raise DomainValidationError(
                f"La expiracion {self.expiration_date} no es posterior a la fecha de "
                f"observacion {self.as_of.date}. Una serie ya vencida no se cotiza."
            )
        if self.contract_multiplier <= 0:
            raise DomainValidationError(
                f"El multiplicador debe ser positivo, se recibio {self.contract_multiplier}."
            )
        if self.bid_price is not None and self.bid_price < 0.0:
            raise DomainValidationError(
                f"La postura de compra no puede ser negativa: {self.bid_price}."
            )
        if self.ask_price is not None and self.ask_price < 0.0:
            raise DomainValidationError(
                f"La postura de venta no puede ser negativa: {self.ask_price}."
            )

    # ------------------------------------------------------------- derivados
    @property
    def mid_price(self) -> float | None:
        """Punto medio entre posturas.

        Es el precio que usa el proyecto, y no el ultimo negociado. El ultimo
        puede tener horas de antiguedad y corresponder a un nivel del
        subyacente que ya no existe; el punto medio siempre describe el
        estado del libro en el momento de la observacion.
        """
        if self.bid_price is None or self.ask_price is None:
            return None
        return 0.5 * (self.bid_price + self.ask_price)

    @property
    def bid_ask_spread(self) -> float | None:
        """Diferencia absoluta entre posturas, en puntos de indice."""
        if self.bid_price is None or self.ask_price is None:
            return None
        return self.ask_price - self.bid_price

    @property
    def relative_bid_ask_spread(self) -> float | None:
        """Spread como fraccion del precio medio.

        Es la medida de confianza de la cotizacion. Un spread relativo alto
        indica que el punto medio es una estimacion pobre del precio justo, y
        alimenta tanto los filtros de calidad como la ponderacion del ajuste
        de la superficie.
        """
        mid = self.mid_price
        spread = self.bid_ask_spread
        if mid is None or spread is None or mid <= 0.0:
            return None
        return spread / mid

    @property
    def passed_quality_filters(self) -> bool:
        """True si la cotizacion no tiene ninguna marca de calidad activa."""
        return self.quality_flags == 0

    @property
    def contract_key(self) -> tuple[str, str, str, float, dt.date]:
        """Identidad del contrato dentro de una fecha de observacion.

        Se deriva en vez de almacenarse: guardar un identificador ademas de
        los campos que lo componen crea dos fuentes de verdad que pueden
        discrepar. Si el proveedor entrega su propio identificador, va en un
        campo aparte y solo para auditoria.
        """
        return (
            self.underlying_symbol,
            self.contract_root,
            str(self.option_type),
            self.strike_price,
            self.expiration_date,
        )
