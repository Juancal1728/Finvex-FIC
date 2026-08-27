"""Valuacion Black-Scholes en la parametrizacion sobre el forward.

Este modulo escribe Black-Scholes en terminos del **precio forward** y no del
precio spot. Es la formulacion que Black publico en 1976 para opciones sobre
futuros, y es la que conviene aqui por tres razones:

1. El proyecto obtiene el forward de la propia cadena mediante paridad
   put-call, no de un pronostico de dividendos. Escribir la formula sobre el
   forward hace que los dividendos y el costo de acarreo **desaparezcan** del
   pricer: ya estan dentro del forward. Un modelo con menos entradas es un
   modelo con menos formas de equivocarse.
2. La metodologia BKM y la del VIX estan escritas sobre el forward y sobre
   opciones fuera del dinero respecto de el. Mantener la misma referencia en
   todo el proyecto evita traducir de ida y vuelta.
3. El factor de descuento queda separado y explicito, en vez de escondido
   dentro de una tasa que tambien aparece en el drift. Esa separacion es la
   que permite descontar con la curva cero real sin tocar la formula.

Advertencia de alcance: esta valuacion **no** es el motor del proyecto. BKM
es libre de modelo y no la necesita. Black-Scholes cumple aqui dos papeles
auxiliares y ninguno mas: invertir precios a volatilidad implicita, que es
solo una convencion de cotizacion, y construir cadenas sinteticas de prueba.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

from finvex.domain.errors import DomainValidationError
from finvex.domain.values.enumerations import OptionType

MINIMUM_SEARCHABLE_VOLATILITY: float = 1e-8
"""Cota inferior del intervalo de busqueda de la volatilidad implicita."""

MAXIMUM_SEARCHABLE_VOLATILITY: float = 5.0
"""Cota superior: 500 % anualizado. Por encima, el precio no es informativo."""

_IMPLIED_VOLATILITY_TOLERANCE: float = 1e-12


def _standardized_log_moneyness_terms(
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
) -> tuple[float, float]:
    """Devuelve los terminos d1 y d2 de la formula.

    d1 y d2 son la moneyness logaritmica del strike respecto al forward,
    escalada por la desviacion estandar total del log-retorno y corregida por
    el termino de convexidad. Se calculan juntos porque comparten
    subexpresiones y porque separarlos invita a usar uno con la volatilidad
    de otro.
    """
    total_volatility = volatility * math.sqrt(time_to_expiry_years)
    log_moneyness = math.log(forward_price / strike_price)
    d1 = (log_moneyness + 0.5 * total_volatility**2) / total_volatility
    d2 = d1 - total_volatility
    return d1, d2


def _validate_pricing_inputs(
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    discount_factor: float,
) -> None:
    if forward_price <= 0.0:
        raise DomainValidationError(
            f"El precio forward debe ser positivo, se recibio {forward_price}."
        )
    if strike_price <= 0.0:
        raise DomainValidationError(f"El strike debe ser positivo, se recibio {strike_price}.")
    if time_to_expiry_years <= 0.0:
        raise DomainValidationError(
            f"El tiempo a vencimiento debe ser positivo, se recibieron {time_to_expiry_years} anos."
        )
    if volatility < 0.0:
        raise DomainValidationError(
            f"La volatilidad no puede ser negativa, se recibio {volatility}. "
            "Recuerda que la convencion del proyecto es decimal anualizado: 0.185, no 18.5."
        )
    if not 0.0 < discount_factor <= 1.0 + 1e-9:
        raise DomainValidationError(
            f"El factor de descuento debe estar en (0, 1], se recibio {discount_factor}."
        )


def black_scholes_call_price(
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    discount_factor: float = 1.0,
) -> float:
    """Precio de una call europea sobre el forward.

    Argumentos:
        forward_price: precio forward del subyacente al vencimiento, en
            puntos de indice.
        strike_price: precio de ejercicio, en puntos de indice.
        time_to_expiry_years: tiempo a vencimiento en anos, con la convencion
            de minutos sobre 525 600 (ver `TimeToExpiry`).
        volatility: volatilidad implicita en **decimal anualizado**.
        discount_factor: valor presente de una unidad pagada al vencimiento.
            El valor predeterminado 1.0 devuelve el precio sin descontar, que
            es lo que necesita la mezcla de lognormales.

    Devuelve:
        El precio de la call. Con volatilidad cero devuelve el valor
        intrinseco descontado, que es el limite correcto y evita una division
        por cero.
    """
    _validate_pricing_inputs(
        forward_price, strike_price, time_to_expiry_years, volatility, discount_factor
    )
    if volatility == 0.0:
        return discount_factor * max(forward_price - strike_price, 0.0)
    d1, d2 = _standardized_log_moneyness_terms(
        forward_price, strike_price, time_to_expiry_years, volatility
    )
    return float(discount_factor * (forward_price * norm.cdf(d1) - strike_price * norm.cdf(d2)))


def black_scholes_put_price(
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    discount_factor: float = 1.0,
) -> float:
    """Precio de un put europeo sobre el forward.

    Se calcula con su propia formula y **no** despejandolo del call por
    paridad, aunque esa segunda ruta sea mas corta. La razon es numerica y
    conviene entenderla, porque reaparece en varios sitios del proyecto.

    Un put muy fuera del dinero vale casi nada; su call gemelo vale casi todo
    el intrinseco. Obtener el put restando dos numeros de magnitudes muy
    distintas es cancelacion catastrofica: el resultado hereda el error
    absoluto del numero grande, que puede ser mayor que el numero pequeno que
    se buscaba. Y los puts muy fuera del dinero son precisamente las series
    que mas pesan en la asimetria y la curtosis implicitas.

    Calculado de forma directa, la paridad sigue cumpliendose a precision de
    maquina, porque `Phi(x) + Phi(-x)` vale uno salvo redondeo. Se gana
    estabilidad sin perder la propiedad.
    """
    _validate_pricing_inputs(
        forward_price, strike_price, time_to_expiry_years, volatility, discount_factor
    )
    if volatility == 0.0:
        return discount_factor * max(strike_price - forward_price, 0.0)
    d1, d2 = _standardized_log_moneyness_terms(
        forward_price, strike_price, time_to_expiry_years, volatility
    )
    return float(discount_factor * (strike_price * norm.cdf(-d2) - forward_price * norm.cdf(-d1)))


def black_scholes_price(
    option_type: OptionType,
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    discount_factor: float = 1.0,
) -> float:
    """Precio de la opcion segun su tipo. Punto de entrada unico para codigo generico."""
    pricer = black_scholes_call_price if option_type is OptionType.CALL else black_scholes_put_price
    return pricer(forward_price, strike_price, time_to_expiry_years, volatility, discount_factor)


def black_scholes_vega(
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    volatility: float,
    discount_factor: float = 1.0,
) -> float:
    """Sensibilidad del precio a la volatilidad, por unidad de volatilidad.

    Es identica para call y put: las dos difieren en un termino que no
    depende de la volatilidad. Aqui se usa para dos cosas concretas: acotar
    la precision alcanzable al invertir la volatilidad implicita, y ponderar
    por confianza los puntos de la superficie donde el precio es poco
    informativo.
    """
    _validate_pricing_inputs(
        forward_price, strike_price, time_to_expiry_years, volatility, discount_factor
    )
    if volatility == 0.0:
        return 0.0
    d1, _ = _standardized_log_moneyness_terms(
        forward_price, strike_price, time_to_expiry_years, volatility
    )
    return float(discount_factor * forward_price * norm.pdf(d1) * math.sqrt(time_to_expiry_years))


def implied_volatility_from_price(
    option_price: float,
    option_type: OptionType,
    forward_price: float,
    strike_price: float,
    time_to_expiry_years: float,
    discount_factor: float = 1.0,
) -> float:
    """Invierte el precio de una opcion para obtener su volatilidad implicita.

    La volatilidad implicita **no es un pronostico ni un parametro de un
    modelo que creamos cierto**: es una convencion de cotizacion. El precio
    es estrictamente creciente en la volatilidad, asi que la aplicacion es
    biyectiva sobre el rango de precios sin arbitraje y la inversion esta
    bien definida. Traducir a volatilidad solo sirve para poder interpolar en
    un espacio donde la superficie es suave; el numero que importa sigue
    siendo el precio.

    Se usa Brent y no Newton a proposito. Newton converge mas rapido cerca de
    la solucion pero se apoya en vega, que colapsa a cero en las alas — que
    es justo donde viven los strikes que mas pesan en la asimetria y la
    curtosis implicitas. Brent solo necesita que la funcion cambie de signo
    en el intervalo, y eso esta garantizado por la monotonia.

    Lanza:
        DomainValidationError: si el precio viola las cotas de no-arbitraje,
        o si la volatilidad implicada excede el intervalo de busqueda.
    """
    intrinsic_value = discount_factor * max(
        option_type.payoff_sign * (forward_price - strike_price), 0.0
    )
    upper_bound_value = discount_factor * (
        forward_price if option_type is OptionType.CALL else strike_price
    )
    if option_price < intrinsic_value - _IMPLIED_VOLATILITY_TOLERANCE:
        raise DomainValidationError(
            f"El precio {option_price} esta por debajo del valor intrinseco {intrinsic_value}: "
            "viola una cota de no-arbitraje y no admite volatilidad implicita."
        )
    if option_price > upper_bound_value + _IMPLIED_VOLATILITY_TOLERANCE:
        raise DomainValidationError(
            f"El precio {option_price} excede la cota superior {upper_bound_value} del contrato."
        )

    def pricing_error(candidate_volatility: float) -> float:
        return (
            black_scholes_price(
                option_type,
                forward_price,
                strike_price,
                time_to_expiry_years,
                candidate_volatility,
                discount_factor,
            )
            - option_price
        )

    error_at_lower_bound = pricing_error(MINIMUM_SEARCHABLE_VOLATILITY)
    error_at_upper_bound = pricing_error(MAXIMUM_SEARCHABLE_VOLATILITY)

    # El valor temporal es lo unico que aporta informacion sobre volatilidad:
    # el intrinseco no depende de ella. El umbral se escala con el intrinseco
    # porque en una serie muy dentro del dinero el valor temporal desaparece
    # dentro del redondeo de un numero grande.
    observed_time_value = -error_at_lower_bound
    minimum_resolvable_time_value = _IMPLIED_VOLATILITY_TOLERANCE * max(1.0, intrinsic_value)
    if observed_time_value <= minimum_resolvable_time_value:
        # Todo el precio es intrinseco: distintas volatilidades producen el
        # mismo numero y la inversion deja de estar definida. Devolver la cota
        # inferior seria dar una respuesta falsa con apariencia de respuesta.
        #
        # No es un caso patologico ni raro: le ocurre a cualquier serie lo
        # bastante alejada del dinero, y tambien a las muy dentro del dinero.
        # Es exactamente la razon por la que la extraccion de momentos trunca
        # las alas en vez de integrar hasta cero, y por la que la cobertura
        # efectiva de strikes es un dato que hay que reportar junto a cada
        # momento estimado.
        raise DomainValidationError(
            f"El valor temporal observado es {observed_time_value:.3g}, por debajo del minimo "
            f"resoluble {minimum_resolvable_time_value:.3g} para un intrinseco de "
            f"{intrinsic_value:.6g}: la volatilidad implicita no esta identificada para esta serie."
        )
    if error_at_upper_bound < 0.0:
        raise DomainValidationError(
            f"El precio {option_price} implica una volatilidad superior a "
            f"{MAXIMUM_SEARCHABLE_VOLATILITY:.0%}. Revisa las unidades de la entrada."
        )
    return float(
        brentq(
            pricing_error,
            MINIMUM_SEARCHABLE_VOLATILITY,
            MAXIMUM_SEARCHABLE_VOLATILITY,
            xtol=1e-12,
            rtol=1e-14,
            maxiter=200,
        )
    )
