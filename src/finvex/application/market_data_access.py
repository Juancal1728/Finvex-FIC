"""`PointInTimeMarketDataStore`: la unica puerta de lectura del proyecto.

## El problema que resuelve

El look-ahead —usar informacion que no estaba disponible en el momento de
tomar la decision— es el error mas caro de un proyecto de backtesting,
porque no se manifiesta como un fallo. Se manifiesta como un resultado
excelente. El notebook original de este proyecto lo tenia en dos sitios a la
vez, y el backtest se veia impecable.

La respuesta habitual es la disciplina: "acuerdate de no mirar el futuro".
La disciplina falla, sobre todo meses despues, sobre todo cuando uno tiene
prisa. La respuesta de este proyecto es estructural: **existe una sola
funcion capaz de leer datos, y esa funcion recibe una fecha de corte y no
puede devolver nada posterior**.

## Por que el store audita al proveedor

Podria bastar con pasarle la fecha al proveedor y confiar. No basta. Un
adaptador puede tener un error de filtro, un archivo de datos puede traer una
fila con fecha corrida, un proveedor puede devolver la ultima cotizacion
disponible en lugar de la de la fecha pedida. El store verifica lo que
recibe y lanza `LookAheadError` si algo no cuadra.

Es una comprobacion barata que convierte una clase entera de errores
silenciosos en un fallo ruidoso e inmediato.
"""

from __future__ import annotations

import datetime as dt

from finvex.domain.errors import LookAheadError
from finvex.domain.model.option_chain import OptionChain
from finvex.domain.model.rate_curve import RateCurve
from finvex.domain.values.as_of import AsOf
from finvex.ports.market_data import OptionsDataProvider


class PointInTimeMarketDataStore:
    """Acceso a datos de mercado con frontera de informacion garantizada.

    Envuelve un proveedor y verifica que todo lo que devuelve sea observable
    a la fecha de corte solicitada.
    """

    def __init__(self, provider: OptionsDataProvider) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    def is_available(self) -> bool:
        return self._provider.is_available()

    def observation_dates_up_to(self, as_of: AsOf, underlying_symbol: str) -> tuple[dt.date, ...]:
        """Fechas de observacion que ya habian ocurrido a la fecha de corte.

        Es el metodo que usa cualquier modelo que necesite una ventana de
        entrenamiento. Devolver aqui las fechas ya filtradas evita que cada
        modelo tenga que recordar filtrarlas, que es donde se cuela el error.
        """
        return tuple(
            date
            for date in self._provider.available_observation_dates(underlying_symbol)
            if as_of.covers(date)
        )

    def option_chain(self, as_of: AsOf, underlying_symbol: str) -> OptionChain:
        """Cadena de opciones observable a la fecha de corte.

        Lanza:
            LookAheadError: si el proveedor devuelve una cadena de otra fecha
            o con contratos ya vencidos a esa fecha.
        """
        chain = self._provider.option_chain(as_of, underlying_symbol)
        self._assert_chain_is_observable(chain, as_of)
        return chain

    def rate_curve(self, as_of: AsOf) -> RateCurve:
        """Curva cero observable a la fecha de corte."""
        return self._provider.rate_curve(as_of)

    # ------------------------------------------------------------ auditoria
    @staticmethod
    def _assert_chain_is_observable(chain: OptionChain, as_of: AsOf) -> None:
        if chain.as_of != as_of:
            raise LookAheadError(
                f"Se pidio la cadena del {as_of} y el proveedor devolvio la del {chain.as_of}. "
                "Un desfase de fecha en la fuente contamina todo lo que venga despues."
            )
        for quote in chain:
            if not as_of.covers(quote.as_of.date):
                raise LookAheadError(
                    f"La cotizacion {quote.contract_key} tiene fecha de observacion "
                    f"{quote.as_of}, posterior al corte {as_of}."
                )
            if quote.expiration_date <= as_of.date:
                raise LookAheadError(
                    f"La cotizacion {quote.contract_key} expira el {quote.expiration_date}, "
                    f"que no es posterior al corte {as_of}. Una serie vencida en la cadena "
                    "indica que la fuente esta desalineada en el tiempo."
                )
