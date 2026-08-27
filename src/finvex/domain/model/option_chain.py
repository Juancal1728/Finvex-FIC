"""`OptionChain`: todas las cotizaciones de un subyacente en una fecha.

Una cadena no es simplemente una lista de cotizaciones. Es el conjunto sobre
el que se hacen las preguntas que el pipeline necesita responder una y otra
vez: que vencimientos hay, que strikes tiene cada vencimiento, que pares
call-put comparten strike. Encapsularlas aqui evita que cada modulo aguas
abajo reimplemente el mismo agrupamiento con criterios ligeramente distintos.

La cadena es inmutable. Filtrar devuelve una cadena nueva en vez de modificar
la existente, de modo que un modulo no pueda alterar sin querer los datos que
otro esta usando. Con volumenes de fin de dia el costo de copiar es
despreciable frente al beneficio de poder razonar sobre el codigo.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from finvex.domain.errors import DomainValidationError
from finvex.domain.model.option_quote import OptionQuote
from finvex.domain.values.as_of import AsOf
from finvex.domain.values.enumerations import ExerciseStyle, OptionType


@dataclass(frozen=True, slots=True)
class CallPutPair:
    """Un call y un put del mismo subyacente, strike y vencimiento.

    Es la unidad que necesita la paridad put-call. Existe como tipo propio
    porque emparejar por strike a mano, cada vez, es donde se cuelan los
    errores de alineacion.
    """

    strike_price: float
    call: OptionQuote
    put: OptionQuote

    @property
    def price_difference(self) -> float | None:
        """Precio del call menos precio del put. Entrada de la paridad."""
        call_price = self.call.mid_price
        put_price = self.put.mid_price
        if call_price is None or put_price is None:
            return None
        return call_price - put_price

    @property
    def absolute_price_difference(self) -> float | None:
        """Valor absoluto de la diferencia. Selecciona el strike de referencia."""
        difference = self.price_difference
        return None if difference is None else abs(difference)


@dataclass(frozen=True, slots=True)
class OptionChain:
    """Conjunto inmutable de cotizaciones de un subyacente en una fecha."""

    as_of: AsOf
    underlying_symbol: str
    underlying_price: float
    quotes: tuple[OptionQuote, ...]

    def __post_init__(self) -> None:
        if self.underlying_price <= 0.0:
            raise DomainValidationError(
                f"El precio del subyacente debe ser positivo, se recibio {self.underlying_price}."
            )
        for quote in self.quotes:
            if quote.as_of != self.as_of:
                raise DomainValidationError(
                    f"La cotizacion de {quote.contract_root} tiene fecha {quote.as_of} y la "
                    f"cadena {self.as_of}. Mezclar fechas dentro de una cadena es una via "
                    "directa al look-ahead."
                )
            if quote.underlying_symbol != self.underlying_symbol:
                raise DomainValidationError(
                    f"La cadena es de {self.underlying_symbol} pero contiene una cotizacion "
                    f"de {quote.underlying_symbol}."
                )

    def __len__(self) -> int:
        return len(self.quotes)

    def __iter__(self) -> Iterator[OptionQuote]:
        return iter(self.quotes)

    # ------------------------------------------------------------ agrupaciones
    @property
    def expiration_dates(self) -> tuple[dt.date, ...]:
        """Vencimientos presentes, ordenados de mas cercano a mas lejano."""
        return tuple(sorted({quote.expiration_date for quote in self.quotes}))

    def strikes_for_expiration(self, expiration_date: dt.date) -> tuple[float, ...]:
        """Strikes disponibles en un vencimiento, ordenados."""
        return tuple(
            sorted(
                {
                    quote.strike_price
                    for quote in self.quotes
                    if quote.expiration_date == expiration_date
                }
            )
        )

    def filter_by(self, predicate: Callable[[OptionQuote], bool]) -> OptionChain:
        """Devuelve una cadena nueva con las cotizaciones que cumplen el predicado.

        Recibe la condicion como funcion y no como una lista de parametros
        booleanos. Un metodo con banderas del tipo `only_calls=True,
        only_liquid=True` crece sin control y obliga a leer su implementacion
        para saber que hace; un predicado se lee en el sitio donde se usa.
        """
        return OptionChain(
            as_of=self.as_of,
            underlying_symbol=self.underlying_symbol,
            underlying_price=self.underlying_price,
            quotes=tuple(quote for quote in self.quotes if predicate(quote)),
        )

    def for_expiration(self, expiration_date: dt.date) -> OptionChain:
        """Subcadena de un solo vencimiento."""
        return self.filter_by(lambda quote: quote.expiration_date == expiration_date)

    def of_type(self, option_type: OptionType) -> OptionChain:
        """Subcadena de un solo tipo de opcion."""
        return self.filter_by(lambda quote: quote.option_type is option_type)

    def passing_quality_filters(self) -> OptionChain:
        """Subcadena con las cotizaciones sin marcas de calidad activas."""
        return self.filter_by(lambda quote: quote.passed_quality_filters)

    def call_put_pairs(self, expiration_date: dt.date) -> tuple[CallPutPair, ...]:
        """Pares call-put que comparten strike dentro de un vencimiento.

        Solo devuelve los strikes donde existen ambas patas. Un strike con
        una sola pata no aporta a la paridad y arrastrarlo obligaria a
        comprobar `None` en cada uso.
        """
        expiration_chain = self.for_expiration(expiration_date)
        calls_by_strike = {
            quote.strike_price: quote for quote in expiration_chain.of_type(OptionType.CALL)
        }
        puts_by_strike = {
            quote.strike_price: quote for quote in expiration_chain.of_type(OptionType.PUT)
        }
        shared_strikes = sorted(set(calls_by_strike) & set(puts_by_strike))
        return tuple(
            CallPutPair(
                strike_price=strike,
                call=calls_by_strike[strike],
                put=puts_by_strike[strike],
            )
            for strike in shared_strikes
        )

    # ------------------------------------------------------------- validacion
    def exercise_styles_present(self) -> frozenset[ExerciseStyle]:
        """Estilos de ejercicio presentes en la cadena."""
        return frozenset(quote.exercise_style for quote in self.quotes)

    def supports_model_free_moment_extraction(self) -> bool:
        """True si toda la cadena es de ejercicio europeo.

        Es la comprobacion que el motor BKM ejecuta antes de integrar nada.
        Se responde a nivel de cadena y no de cotizacion porque una sola
        serie americana ya contamina el momento del vencimiento completo.
        """
        return all(
            quote.exercise_style.supports_model_free_moment_extraction for quote in self.quotes
        )

    @classmethod
    def from_quotes(cls, quotes: Sequence[OptionQuote]) -> OptionChain:
        """Construye la cadena deduciendo fecha, subyacente y spot de las cotizaciones.

        Falla si las cotizaciones no son homogeneas, en vez de escoger la
        primera y seguir adelante.
        """
        if not quotes:
            raise DomainValidationError("No se puede construir una cadena vacia sin contexto.")
        first = quotes[0]
        return cls(
            as_of=first.as_of,
            underlying_symbol=first.underlying_symbol,
            underlying_price=first.underlying_price,
            quotes=tuple(quotes),
        )
