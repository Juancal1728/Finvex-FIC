"""`ExpiryContext`: todo lo que un vencimiento necesita saber de si mismo.

Este objeto resuelve un problema de diseno concreto. El forward, la tasa, el
tiempo a vencimiento y el strike de anclaje son propiedades de la pareja
(fecha de observacion, vencimiento), no de cada contrato. Si viven en cada
cotizacion se duplican cientos de veces y nada impide que dos contratos del
mismo vencimiento acaben con forwards distintos, lo que rompe la paridad y
con ella el calculo de momentos.

Al agruparlos en un solo objeto, calculado una vez por vencimiento, esa clase
de inconsistencia deja de ser posible.

El contexto ademas transporta dos cifras que no son de calculo sino de
**evidencia**: la cobertura de strikes en cada ala, medida en desviaciones
estandar implicitas respecto del forward. Esos dos numeros dicen cuanta cola
se esta observando de verdad y cuanta se esta extrapolando. Son la primera
cosa que mira quien evalua una extraccion de momentos implicitos, porque la
asimetria y la curtosis viven precisamente en las alas.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from finvex.domain.errors import DomainValidationError
from finvex.domain.model.option_chain import OptionChain
from finvex.domain.model.rate_curve import RateCurve
from finvex.domain.options.put_call_parity import (
    implied_forward_from_pairs,
    largest_strike_at_or_below,
)
from finvex.domain.values.as_of import AsOf
from finvex.domain.values.enumerations import SettlementConvention
from finvex.domain.values.time_to_expiry import TimeToExpiry


@dataclass(frozen=True, slots=True)
class StrikeCoverage:
    """Hasta donde llega la cadena en cada ala, en desviaciones estandar.

    La unidad es deliberada. Decir "la cadena llega hasta el strike 3000" no
    informa: depende del nivel del indice y del plazo. Decir "la cadena llega
    a menos 3.2 desviaciones estandar implicitas" si informa, y es comparable
    entre fechas, entre vencimientos y entre subyacentes.
    """

    lowest_strike: float
    highest_strike: float
    standard_deviations_below_forward: float
    standard_deviations_above_forward: float
    usable_strike_count: int

    @property
    def is_symmetric(self) -> bool:
        """True si las dos alas llegan aproximadamente igual de lejos.

        Una cobertura muy asimetrica sesga la asimetria implicita estimada en
        una direccion conocida, asi que conviene detectarla y reportarla.
        """
        return (
            abs(self.standard_deviations_below_forward - self.standard_deviations_above_forward)
            < 0.5
        )


@dataclass(frozen=True, slots=True)
class ExpiryContext:
    """Parametros de mercado de un vencimiento en una fecha de observacion.

    Atributos:
        as_of: fecha de observacion.
        underlying_symbol: subyacente.
        expiration_date: fecha de expiracion de la serie.
        settlement_convention: liquidacion AM o PM; determina la hora exacta.
        time_to_expiry: tiempo a vencimiento con la convencion de minutos.
        forward_price: forward implicito extraido por paridad put-call.
        reference_strike_price: strike en el que se extrajo el forward. Se
            guarda para poder reproducir el calculo despues.
        anchor_strike_price: mayor strike que no supera el forward. Es el
            `K0` de la metodologia del VIX y el punto que separa los puts de
            los calls fuera del dinero.
        continuously_compounded_rate: tasa cero del plazo.
        discount_factor: valor presente de una unidad al vencimiento.
        strike_coverage: evidencia de cuanta cola se observa.
    """

    as_of: AsOf
    underlying_symbol: str
    expiration_date: dt.date
    settlement_convention: SettlementConvention
    time_to_expiry: TimeToExpiry
    forward_price: float
    reference_strike_price: float
    anchor_strike_price: float
    continuously_compounded_rate: float
    discount_factor: float
    strike_coverage: StrikeCoverage

    def __post_init__(self) -> None:
        if self.forward_price <= 0.0:
            raise DomainValidationError(
                f"El forward implicito resulto {self.forward_price}, que no es positivo. "
                "Revisa los precios del par de referencia."
            )
        if self.anchor_strike_price > self.forward_price:
            raise DomainValidationError(
                f"El strike de anclaje {self.anchor_strike_price} supera al forward "
                f"{self.forward_price}; por definicion debe quedar al nivel o por debajo."
            )

    @property
    def time_to_expiry_years(self) -> float:
        """Atajo de lectura frecuente."""
        return self.time_to_expiry.years

    def log_moneyness(self, strike_price: float) -> float:
        """Moneyness logaritmica del strike respecto al forward.

        Es la convencion del proyecto: adimensional, simetrica alrededor del
        dinero y comparable entre vencimientos con niveles de forward muy
        distintos.
        """
        return math.log(strike_price / self.forward_price)

    def is_out_of_the_money(self, strike_price: float, is_call: bool) -> bool:
        """True si la opcion esta fuera del dinero respecto del **forward**.

        BKM y la metodologia del VIX integran sobre opciones fuera del dinero
        para usar siempre la pata mas liquida y con menos valor intrinseco.
        La condicion se mide contra el forward, no contra el spot: usar el
        spot desplazaria el punto de corte y mezclaria patas con valor
        intrinseco dentro de la integral.
        """
        if is_call:
            return strike_price >= self.anchor_strike_price
        return strike_price <= self.anchor_strike_price

    @classmethod
    def from_chain(
        cls,
        chain: OptionChain,
        expiration_date: dt.date,
        settlement_convention: SettlementConvention,
        rate_curve: RateCurve,
        assumed_volatility_for_coverage: float = 0.20,
    ) -> ExpiryContext:
        """Construye el contexto de un vencimiento a partir de la cadena.

        El orden de las operaciones no es arbitrario y conviene seguirlo:

        1. El tiempo a vencimiento se obtiene de la convencion de liquidacion,
           porque la tasa se lee de la curva **en ese plazo**.
        2. El factor de descuento sale de la tasa, porque la paridad lo
           necesita para despejar el forward.
        3. El forward se extrae por paridad del strike donde call y put estan
           mas cerca.
        4. El strike de anclaje se elige respecto al forward ya extraido.

        Invertir los pasos uno y dos, que es el error tipico, produce un
        descuento del plazo equivocado y un forward sesgado.

        El argumento `assumed_volatility_for_coverage` solo escala la medida
        de cobertura de strikes; no entra en ningun precio ni en ningun
        momento. Se usa una volatilidad de referencia porque la cobertura hay
        que medirla **antes** de tener una superficie ajustada.
        """
        time_to_expiry = TimeToExpiry.from_expiration_date(
            valuation_date=chain.as_of.date,
            expiration_date=expiration_date,
            settlement=settlement_convention,
        )
        continuously_compounded_rate = rate_curve.rate_at(time_to_expiry.years)
        discount_factor = rate_curve.discount_factor_at(time_to_expiry.years)

        pairs = chain.call_put_pairs(expiration_date)
        forward_price, reference_strike_price = implied_forward_from_pairs(pairs, discount_factor)

        available_strikes = chain.strikes_for_expiration(expiration_date)
        anchor_strike_price = largest_strike_at_or_below(available_strikes, forward_price)

        total_volatility = assumed_volatility_for_coverage * math.sqrt(time_to_expiry.years)
        lowest_strike = min(available_strikes)
        highest_strike = max(available_strikes)
        coverage = StrikeCoverage(
            lowest_strike=lowest_strike,
            highest_strike=highest_strike,
            standard_deviations_below_forward=abs(
                math.log(lowest_strike / forward_price) / total_volatility
            ),
            standard_deviations_above_forward=abs(
                math.log(highest_strike / forward_price) / total_volatility
            ),
            usable_strike_count=len(available_strikes),
        )

        return cls(
            as_of=chain.as_of,
            underlying_symbol=chain.underlying_symbol,
            expiration_date=expiration_date,
            settlement_convention=settlement_convention,
            time_to_expiry=time_to_expiry,
            forward_price=forward_price,
            reference_strike_price=reference_strike_price,
            anchor_strike_price=anchor_strike_price,
            continuously_compounded_rate=continuously_compounded_rate,
            discount_factor=discount_factor,
            strike_coverage=coverage,
        )
