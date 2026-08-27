"""Forward implicito a partir de la paridad put-call.

## Por que el forward se extrae y no se pronostica

Para calcular momentos implicitos hace falta el precio forward del
subyacente en cada vencimiento. La ruta directa seria construirlo desde el
spot: descontar por la tasa y restar el valor presente de los dividendos
esperados. Esa ruta tiene un problema serio: **exige pronosticar
dividendos**, y el error de ese pronostico entra directo en el forward, de
ahi en la moneyness de cada strike, y de ahi en los momentos.

La alternativa es dejar que el mercado responda. La paridad put-call es una
relacion de no-arbitraje entre precios observables:

    call - put = discount_factor * (forward_price - strike_price)

Despejando, el forward queda determinado por precios que ya estan en la
pantalla:

    forward_price = strike_price + (call - put) / discount_factor

El forward asi obtenido incorpora automaticamente los dividendos que el
mercado realmente espera, el costo de financiamiento efectivo y el costo de
prestamo del subyacente. No hay que estimarlos: ya estan dentro. Es la ruta
que sigue la metodologia del VIX de Cboe, y es la que sigue este proyecto.

## Por que importa cual strike se usa

La relacion es exacta en teoria y aproximada en datos reales: cada precio
trae su propio ruido de microestructura. Ese ruido se amplifica cuando la
diferencia entre call y put es grande en relacion con sus niveles, lo que
ocurre en strikes alejados del dinero, donde una pata es casi todo valor
intrinseco y la otra vale casi nada.

Por eso se elige el strike donde la diferencia absoluta entre call y put es
minima: es el punto donde las dos patas tienen tamano comparable y el error
relativo del cociente es menor. Cboe usa exactamente este criterio.
"""

from __future__ import annotations

from finvex.domain.errors import DomainValidationError
from finvex.domain.model.option_chain import CallPutPair


def select_reference_pair(call_put_pairs: tuple[CallPutPair, ...]) -> CallPutPair:
    """Elige el par call-put cuya diferencia de precios es menor en valor absoluto.

    Es el strike mas cercano al dinero segun los propios precios, no segun el
    spot. Esa distincion es deliberada: el mercado de opciones sabe donde
    esta el forward mejor que cualquier calculo hecho desde el spot con
    dividendos supuestos.
    """
    usable_pairs = [pair for pair in call_put_pairs if pair.absolute_price_difference is not None]
    if not usable_pairs:
        raise DomainValidationError(
            "No hay ningun strike con call y put cotizados a la vez. Sin al menos un par "
            "no se puede extraer el forward por paridad."
        )
    return min(usable_pairs, key=lambda pair: pair.absolute_price_difference or float("inf"))


def implied_forward_price(
    reference_strike_price: float,
    call_minus_put_price: float,
    discount_factor: float,
) -> float:
    """Forward implicito despejado de la paridad put-call.

    Argumentos:
        reference_strike_price: strike del par elegido.
        call_minus_put_price: precio del call menos precio del put en ese strike.
        discount_factor: valor presente de una unidad pagada al vencimiento.

    Devuelve:
        El precio forward implicito, en puntos de indice.
    """
    if discount_factor <= 0.0:
        raise DomainValidationError(
            f"El factor de descuento debe ser positivo, se recibio {discount_factor}."
        )
    return reference_strike_price + call_minus_put_price / discount_factor


def implied_forward_from_pairs(
    call_put_pairs: tuple[CallPutPair, ...],
    discount_factor: float,
) -> tuple[float, float]:
    """Extrae el forward de una lista de pares y devuelve tambien el strike usado.

    Devuelve la pareja `(forward_price, reference_strike_price)`. Se devuelve
    tambien el strike porque queda registrado en el `ExpiryContext`: sin el,
    reproducir despues el calculo es imposible.
    """
    reference_pair = select_reference_pair(call_put_pairs)
    price_difference = reference_pair.price_difference
    if price_difference is None:  # pragma: no cover - ya filtrado en la seleccion
        raise DomainValidationError("El par de referencia no tiene precios completos.")
    forward_price = implied_forward_price(
        reference_strike_price=reference_pair.strike_price,
        call_minus_put_price=price_difference,
        discount_factor=discount_factor,
    )
    return forward_price, reference_pair.strike_price


def parity_residual(
    call_price: float,
    put_price: float,
    strike_price: float,
    forward_price: float,
    discount_factor: float,
) -> float:
    """Cuanto se desvia un strike de la paridad, dado un forward ya fijado.

    Es el diagnostico que alimenta el codigo de calidad correspondiente. Una
    desviacion grande en un strike concreto senala una cotizacion rancia o
    una pata mal emparejada, no un fallo del forward: el forward se estimo en
    el strike de referencia, no en este.
    """
    theoretical_difference = discount_factor * (forward_price - strike_price)
    return (call_price - put_price) - theoretical_difference


def largest_strike_at_or_below(strike_prices: tuple[float, ...], reference_level: float) -> float:
    """Mayor strike disponible que no supera el nivel de referencia.

    Es el `K0` de la metodologia del VIX: el strike inmediatamente por debajo
    del forward, que separa los puts fuera del dinero de los calls fuera del
    dinero y ancla el termino de correccion de la formula de varianza.
    """
    candidates = [strike for strike in strike_prices if strike <= reference_level]
    if not candidates:
        raise DomainValidationError(
            f"Ningun strike disponible esta al nivel de referencia {reference_level} o por debajo. "
            "La cadena no cubre el dinero y no se puede anclar la integracion."
        )
    return max(candidates)
