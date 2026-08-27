"""Densidades neutrales al riesgo con momentos conocidos en forma cerrada.

## Para que existe este modulo

El checkpoint V1 del proyecto pregunta algo muy concreto: *cuando le doy a mi
implementacion de BKM una cadena de opciones cuya distribucion subyacente
conozco exactamente, recupera los momentos verdaderos?*

Ese test es superior a comparar contra el VIX publicado, y conviene entender
por que. La replica del VIX es una validacion **conjunta**: si no cuadra,
el error puede estar en la integral, en la superficie interpolada, en los
filtros de calidad o en la convencion de tiempo, y no hay forma de saber en
cual. La validacion sintetica es una validacion **aislada**: la cadena se
construye a partir de precios analiticos exactos, sin ruido, sin spreads y
sin huecos, de modo que la unica pieza que puede fallar es el estimador.
Ademas el resultado esperado no es un dato de mercado sujeto a revision:
es un numero que se deriva a mano.

## Por que una mezcla de lognormales

Se necesita una familia que cumpla tres cosas a la vez, y son mas exigentes
de lo que parecen:

1. **Precios europeos en forma cerrada.** Sin esto no se puede generar la
   cadena sin introducir error de simulacion, y ese error contaminaria el
   test que quiere medir el error del estimador.
2. **Momentos del log-retorno en forma cerrada.** Sin esto no hay valor de
   referencia exacto contra el cual comparar.
3. **Asimetria y curtosis controlables.** Una lognormal simple tiene
   asimetria cero y curtosis tres en el log-retorno: sirve como caso base,
   pero no ejercita las partes del estimador que precisamente interesan.

La mezcla finita de lognormales cumple las tres. Es ademas la familia que la
literatura de densidades implicitas usa desde hace decadas para ajustar
sonrisas de volatilidad, precisamente porque hereda la forma cerrada de
Black-Scholes componente a componente y sigue siendo lo bastante flexible
para producir colas gruesas y asimetria pronunciada.

La intuicion economica es directa y es la razon de que la parametrizacion de
este modulo hable de "escenario de caida": el mercado no cotiza una sola
distribucion suave, cotiza una mezcla de regimenes. Un componente central de
volatilidad moderada, mas un componente de baja probabilidad centrado muy
por debajo y con volatilidad alta, reproduce exactamente el patron que
genera el skew observado en opciones sobre indices.

## La condicion de martingala

Bajo la medida neutral al riesgo, el precio descontado del subyacente es una
martingala. En terminos operativos: la esperanza del precio terminal tiene
que ser el forward. Esa condicion **no es opcional ni cosmetica**. Si se
viola, la densidad no es neutral al riesgo, la paridad put-call deja de
cumplirse en los precios generados, y BKM devolvera momentos incorrectos por
una razon que no tiene nada que ver con BKM.

Por eso la clase la impone en el constructor y falla si no se cumple, en vez
de confiar en que quien la use la respete.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from finvex.domain.errors import DomainValidationError
from finvex.domain.options.black_scholes import black_scholes_call_price

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from numpy.random import Generator

MARTINGALE_TOLERANCE: float = 1e-10
"""Cuanto puede desviarse la suma ponderada de forwards del forward total."""


@dataclass(frozen=True, slots=True)
class LogReturnMoments:
    """Los cuatro momentos del log-retorno bajo la medida neutral al riesgo.

    Estos son exactamente los objetos que BKM estima a partir de precios de
    opciones. Tenerlos como un tipo propio, y no como una tupla suelta, evita
    el error de intercambiar asimetria con curtosis al desempacar.

    Convencion fijada y no negociable en el proyecto: `skewness` y `kurtosis`
    son **estandarizados**, y `kurtosis` es la curtosis, no el exceso de
    curtosis. Bajo esta convencion la normal tiene curtosis 3, no 0. La
    literatura usa ambas y mezclarlas rompe la desigualdad de validacion
    `kurtosis >= skewness^2 + 1`.
    """

    mean: float
    variance: float
    skewness: float
    kurtosis: float

    def __post_init__(self) -> None:
        if self.variance <= 0.0:
            raise DomainValidationError(
                f"La varianza del log-retorno debe ser positiva, se obtuvo {self.variance}."
            )
        lower_bound_on_kurtosis = self.skewness**2 + 1.0
        if self.kurtosis < lower_bound_on_kurtosis - 1e-9:
            raise DomainValidationError(
                f"Se violo la desigualdad kurtosis >= skewness^2 + 1: "
                f"kurtosis={self.kurtosis}, cota={lower_bound_on_kurtosis}. "
                "Esta desigualdad vale para cualquier distribucion, asi que su "
                "violacion indica un error de calculo, no una distribucion exotica."
            )

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

    @property
    def excess_kurtosis(self) -> float:
        """Curtosis menos 3. Se expone para comparar con fuentes que usan esa convencion."""
        return self.kurtosis - 3.0


class RiskNeutralDensity(ABC):
    """Interfaz de una densidad neutral al riesgo del precio terminal.

    Define el contrato minimo que necesita el resto del proyecto: poner
    precio a opciones europeas y entregar los momentos verdaderos del
    log-retorno. Cualquier familia que sepa hacer esas dos cosas puede
    alimentar la validacion de BKM.

    El precio del put no es abstracto: se deriva del call por paridad. Asi la
    paridad se cumple exactamente en toda implementacion presente y futura, y
    no depende de que cada subclase la respete por su cuenta.
    """

    @property
    @abstractmethod
    def spot_price(self) -> float:
        """Precio de contado del subyacente en el momento de valuacion."""

    @property
    @abstractmethod
    def forward_price(self) -> float:
        """Precio forward al vencimiento. Es la esperanza neutral al riesgo del precio terminal."""

    @property
    @abstractmethod
    def time_to_expiry_years(self) -> float:
        """Tiempo a vencimiento en anos."""

    @property
    @abstractmethod
    def discount_factor(self) -> float:
        """Valor presente de una unidad monetaria pagada al vencimiento."""

    @abstractmethod
    def european_call_price(self, strike_price: float) -> float:
        """Precio de una call europea con ese strike."""

    @abstractmethod
    def log_return_moments(self) -> LogReturnMoments:
        """Momentos exactos del log-retorno bajo la medida neutral al riesgo."""

    def european_put_price(self, strike_price: float) -> float:
        """Precio de un put europeo, derivado del call por paridad put-call.

        La paridad dice que una call larga y un put corto del mismo strike
        replican un forward. En terminos de precios:

            call - put = discount_factor * (forward_price - strike_price)

        Derivar el put de aqui, en vez de integrar la densidad otra vez,
        garantiza que la cadena generada satisfaga la paridad **por
        construccion** y no solo aproximadamente. Eso importa porque el
        pipeline del proyecto extrae el forward implicito de la paridad: si
        los datos sinteticos no la cumplieran de forma exacta, el test de esa
        extraccion mediria el error de la densidad y no el del extractor.
        """
        return self.european_call_price(strike_price) - self.discount_factor * (
            self.forward_price - strike_price
        )

    def european_option_price(self, strike_price: float, is_call: bool) -> float:
        """Precio de la opcion segun su direccion. Punto de entrada para codigo generico."""
        if is_call:
            return self.european_call_price(strike_price)
        return self.european_put_price(strike_price)

    def put_call_parity_residual(self, strike_price: float) -> float:
        """Cuanto se desvia la cadena generada de la paridad put-call.

        Deberia ser cero salvo error de punto flotante. Existe como metodo
        publico porque es un test barato que se puede correr sobre cualquier
        densidad nueva antes de confiar en ella.
        """
        theoretical = self.discount_factor * (self.forward_price - strike_price)
        realized = self.european_call_price(strike_price) - self.european_put_price(strike_price)
        return realized - theoretical


@dataclass(frozen=True, slots=True)
class LognormalMixtureComponent:
    """Un regimen dentro de la mezcla.

    Atributos:
        weight: probabilidad neutral al riesgo de este regimen. Los pesos de
            todos los componentes suman uno.
        forward_ratio: forward de este componente dividido por el forward
            total. Un valor de 0.75 describe un regimen en el que el
            subyacente vale, en esperanza, un 25 % menos que el forward
            general. Parametrizar por razon y no por nivel hace la condicion
            de martingala legible: los pesos por las razones deben sumar uno.
        annualized_volatility: volatilidad del log-retorno dentro del
            regimen, en decimal anualizado.
    """

    weight: float
    forward_ratio: float
    annualized_volatility: float

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise DomainValidationError(
                f"El peso del componente debe estar en (0, 1], se recibio {self.weight}."
            )
        if self.forward_ratio <= 0.0:
            raise DomainValidationError(
                f"La razon de forward debe ser positiva, se recibio {self.forward_ratio}."
            )
        if self.annualized_volatility <= 0.0:
            raise DomainValidationError(
                f"La volatilidad del componente debe ser positiva, se recibio "
                f"{self.annualized_volatility}. Convencion del proyecto: decimal anualizado."
            )


class LognormalMixtureRiskNeutralDensity(RiskNeutralDensity):
    """Mezcla finita de lognormales con precios y momentos exactos.

    El log-retorno `X = ln(S_T / S_0)` sigue una mezcla de normales: con
    probabilidad `w_i`, `X ~ Normal(m_i, s_i^2)`, donde `s_i` es la
    volatilidad del componente escalada por la raiz del tiempo y `m_i` queda
    determinado por la razon de forward del componente:

        m_i = ln(F * ratio_i / S_0) - s_i^2 / 2

    Esa expresion sale de exigir que la esperanza del componente sea su
    propio forward: `E[S_0 * exp(X_i)] = S_0 * exp(m_i + s_i^2/2) = F_i`.
    """

    def __init__(
        self,
        spot_price: float,
        forward_price: float,
        time_to_expiry_years: float,
        components: tuple[LognormalMixtureComponent, ...],
        discount_factor: float = 1.0,
    ) -> None:
        if spot_price <= 0.0:
            raise DomainValidationError(
                f"El precio spot debe ser positivo, se recibio {spot_price}."
            )
        if forward_price <= 0.0:
            raise DomainValidationError(
                f"El precio forward debe ser positivo, se recibio {forward_price}."
            )
        if time_to_expiry_years <= 0.0:
            raise DomainValidationError(
                "El tiempo a vencimiento debe ser positivo, se recibieron "
                f"{time_to_expiry_years} anos."
            )
        if not components:
            raise DomainValidationError("La mezcla necesita al menos un componente.")

        total_weight = math.fsum(component.weight for component in components)
        if abs(total_weight - 1.0) > MARTINGALE_TOLERANCE:
            raise DomainValidationError(
                f"Los pesos de la mezcla deben sumar uno, suman {total_weight}."
            )

        weighted_forward_ratio = math.fsum(
            component.weight * component.forward_ratio for component in components
        )
        if abs(weighted_forward_ratio - 1.0) > MARTINGALE_TOLERANCE:
            raise DomainValidationError(
                f"Se viola la condicion de martingala: la suma ponderada de razones de forward "
                f"es {weighted_forward_ratio} y debe ser exactamente uno. Sin esto la densidad "
                "no es neutral al riesgo y la paridad put-call no se cumple en los precios "
                "generados."
            )

        self._spot_price = spot_price
        self._forward_price = forward_price
        self._time_to_expiry_years = time_to_expiry_years
        self._discount_factor = discount_factor
        self._components = components

    # ------------------------------------------------------------------ estado
    @property
    def spot_price(self) -> float:
        return self._spot_price

    @property
    def forward_price(self) -> float:
        return self._forward_price

    @property
    def time_to_expiry_years(self) -> float:
        return self._time_to_expiry_years

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def components(self) -> tuple[LognormalMixtureComponent, ...]:
        return self._components

    # ------------------------------------------------------------ constructores
    @classmethod
    def from_single_lognormal(
        cls,
        spot_price: float,
        forward_price: float,
        time_to_expiry_years: float,
        annualized_volatility: float,
        discount_factor: float = 1.0,
    ) -> LognormalMixtureRiskNeutralDensity:
        """El mundo de Black-Scholes: un solo regimen.

        Es el caso base de la validacion y el mas informativo, porque sus
        momentos son inmediatos de verificar a mano: la varianza del
        log-retorno es exactamente `volatilidad^2 * T`, la asimetria es cero
        y la curtosis es tres. Si el estimador falla aqui, no tiene sentido
        probarlo con nada mas complicado.
        """
        return cls(
            spot_price=spot_price,
            forward_price=forward_price,
            time_to_expiry_years=time_to_expiry_years,
            components=(
                LognormalMixtureComponent(
                    weight=1.0, forward_ratio=1.0, annualized_volatility=annualized_volatility
                ),
            ),
            discount_factor=discount_factor,
        )

    @classmethod
    def from_crash_scenario(
        cls,
        spot_price: float,
        forward_price: float,
        time_to_expiry_years: float,
        base_volatility: float,
        crash_probability: float,
        crash_price_drop: float,
        crash_volatility: float,
        discount_factor: float = 1.0,
    ) -> LognormalMixtureRiskNeutralDensity:
        """Dos regimenes: uno normal y uno de caida, con asimetria negativa.

        Es la parametrizacion que reproduce el patron real de las opciones
        sobre indices. El regimen de caida tiene probabilidad baja, forward
        muy por debajo del general y volatilidad alta; el regimen normal
        absorbe el resto de la masa. La condicion de martingala fija por si
        sola el forward del regimen normal, de modo que no hay libertad para
        equivocarse:

            w_normal * ratio_normal + w_caida * ratio_caida = 1

        Argumentos:
            crash_probability: probabilidad neutral al riesgo del regimen de
                caida, en (0, 1).
            crash_price_drop: caida proporcional del forward en ese regimen.
                0.25 significa que el forward del regimen es un 25 % menor
                que el forward general.
        """
        if not 0.0 < crash_probability < 1.0:
            raise DomainValidationError(
                f"La probabilidad de caida debe estar en (0, 1), se recibio {crash_probability}."
            )
        if not 0.0 < crash_price_drop < 1.0:
            raise DomainValidationError(
                f"La caida proporcional debe estar en (0, 1), se recibio {crash_price_drop}."
            )

        crash_forward_ratio = 1.0 - crash_price_drop
        normal_weight = 1.0 - crash_probability
        # Se despeja de la condicion de martingala: es la unica eleccion que
        # deja la densidad siendo neutral al riesgo.
        normal_forward_ratio = (1.0 - crash_probability * crash_forward_ratio) / normal_weight

        return cls(
            spot_price=spot_price,
            forward_price=forward_price,
            time_to_expiry_years=time_to_expiry_years,
            components=(
                LognormalMixtureComponent(
                    weight=normal_weight,
                    forward_ratio=normal_forward_ratio,
                    annualized_volatility=base_volatility,
                ),
                LognormalMixtureComponent(
                    weight=crash_probability,
                    forward_ratio=crash_forward_ratio,
                    annualized_volatility=crash_volatility,
                ),
            ),
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ precios
    def european_call_price(self, strike_price: float) -> float:
        """Precio exacto de la call como suma ponderada de precios Black-Scholes.

        El resultado es exacto, no una aproximacion. La esperanza del payoff
        bajo una mezcla es la mezcla de las esperanzas, y la esperanza de
        cada componente lognormal es justamente la formula de Black-Scholes
        evaluada en el forward de ese componente. La reutilizacion del pricer
        no es solo economia de codigo: garantiza que la cadena sintetica y el
        pricer del proyecto no puedan desincronizarse.
        """
        undiscounted_price = math.fsum(
            component.weight
            * black_scholes_call_price(
                forward_price=self._forward_price * component.forward_ratio,
                strike_price=strike_price,
                time_to_expiry_years=self._time_to_expiry_years,
                volatility=component.annualized_volatility,
                discount_factor=1.0,
            )
            for component in self._components
        )
        return self._discount_factor * undiscounted_price

    # ------------------------------------------------------------------ momentos
    def _component_log_return_parameters(
        self, component: LognormalMixtureComponent
    ) -> tuple[float, float]:
        """Media y desviacion estandar del log-retorno dentro de un regimen.

        La desviacion es la volatilidad anualizada escalada por la raiz del
        tiempo. La media sale de exigir que la esperanza del precio terminal
        del componente sea su propio forward, lo que introduce la correccion
        de convexidad de un medio de la varianza.
        """
        standard_deviation = component.annualized_volatility * math.sqrt(self._time_to_expiry_years)
        component_forward = self._forward_price * component.forward_ratio
        mean = math.log(component_forward / self._spot_price) - 0.5 * standard_deviation**2
        return mean, standard_deviation

    def log_return_moments(self) -> LogReturnMoments:
        """Momentos exactos del log-retorno `ln(S_T / S_0)` bajo la medida Q.

        Se calculan pasando por los momentos crudos de la mezcla. Para una
        normal de media `m` y desviacion `s`, los cuatro primeros momentos
        crudos son conocidos:

            E[X]   = m
            E[X^2] = m^2 + s^2
            E[X^3] = m^3 + 3 m s^2
            E[X^4] = m^4 + 6 m^2 s^2 + 3 s^4

        Y los momentos crudos de una mezcla son la mezcla de los momentos
        crudos, porque la esperanza es lineal. Los centrales se obtienen
        despues con las identidades habituales, y de ahi los estandarizados.

        Estos son los valores de referencia del checkpoint V1: son los
        numeros que la implementacion de BKM tiene que reproducir.
        """
        first_raw = 0.0
        second_raw = 0.0
        third_raw = 0.0
        fourth_raw = 0.0

        for component in self._components:
            mean, deviation = self._component_log_return_parameters(component)
            variance = deviation**2
            weight = component.weight
            first_raw += weight * mean
            second_raw += weight * (mean**2 + variance)
            third_raw += weight * (mean**3 + 3.0 * mean * variance)
            fourth_raw += weight * (mean**4 + 6.0 * mean**2 * variance + 3.0 * variance**2)

        central_second = second_raw - first_raw**2
        central_third = third_raw - 3.0 * first_raw * second_raw + 2.0 * first_raw**3
        central_fourth = (
            fourth_raw
            - 4.0 * first_raw * third_raw
            + 6.0 * first_raw**2 * second_raw
            - 3.0 * first_raw**4
        )

        return LogReturnMoments(
            mean=first_raw,
            variance=central_second,
            skewness=central_third / central_second**1.5,
            kurtosis=central_fourth / central_second**2,
        )

    # ------------------------------------------------------------- diagnosticos
    def martingale_pricing_error(self) -> float:
        """Desviacion de la condicion de martingala. Debe ser cero salvo redondeo."""
        weighted_forward_ratio = math.fsum(
            component.weight * component.forward_ratio for component in self._components
        )
        return weighted_forward_ratio - 1.0

    def sample_terminal_prices(self, generator: Generator, sample_size: int) -> np.ndarray:
        """Simula precios terminales. Solo para contrastar los momentos analiticos.

        Recibe el generador de forma explicita, nunca lo crea: la
        reproducibilidad del proyecto depende de que ninguna funcion tenga
        una fuente de aleatoriedad propia.
        """
        import numpy

        weights = numpy.array([component.weight for component in self._components])
        chosen = generator.choice(len(self._components), size=sample_size, p=weights)
        log_returns = numpy.empty(sample_size, dtype=float)
        for index, component in enumerate(self._components):
            mask = chosen == index
            mean, deviation = self._component_log_return_parameters(component)
            log_returns[mask] = generator.normal(mean, deviation, size=int(mask.sum()))
        return self._spot_price * numpy.exp(log_returns)
