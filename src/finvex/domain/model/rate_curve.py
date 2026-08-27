"""`RateCurve`: la curva cero, con interpolacion explicita.

BKM descuenta con `exp(rT)` y la metodologia del VIX usa la tasa del plazo de
cada vencimiento. Eso obliga a tener una curva y no una tasa unica, y obliga
a decidir **como se interpola**, porque los vencimientos de las opciones casi
nunca coinciden con los plazos publicados de la curva.

La decision del proyecto es interpolar linealmente en la tasa continua sobre
el plazo. Es la convencion mas simple defendible, y por debajo de un ano la
diferencia frente a alternativas mas elaboradas es de fracciones de punto
base: irrelevante frente al error de la superficie de volatilidad. Se
documenta aqui para que quede constancia de que fue una decision y no un
descuido, y para poder cambiarla en un solo lugar si alguna vez importa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finvex.domain.errors import DomainValidationError


@dataclass(frozen=True, slots=True)
class RateCurve:
    """Curva de tasas cero continuamente compuestas.

    Atributos:
        tenors_in_years: plazos de los puntos observados, en anos y en orden
            creciente.
        continuously_compounded_rates: tasas cero en cada plazo, en decimal
            anualizado y en composicion continua.
    """

    tenors_in_years: tuple[float, ...]
    continuously_compounded_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tenors_in_years) != len(self.continuously_compounded_rates):
            raise DomainValidationError(
                f"La curva tiene {len(self.tenors_in_years)} plazos y "
                f"{len(self.continuously_compounded_rates)} tasas; deben coincidir."
            )
        if not self.tenors_in_years:
            raise DomainValidationError("La curva necesita al menos un punto.")
        if any(tenor <= 0.0 for tenor in self.tenors_in_years):
            raise DomainValidationError("Todos los plazos de la curva deben ser positivos.")
        if list(self.tenors_in_years) != sorted(self.tenors_in_years):
            raise DomainValidationError(
                "Los plazos de la curva deben venir en orden creciente. "
                "Ordenarlos silenciosamente escondería un error del proveedor."
            )

    @classmethod
    def flat(cls, continuously_compounded_rate: float) -> RateCurve:
        """Curva plana. Util en pruebas y en datos sinteticos."""
        return cls(
            tenors_in_years=(1.0 / 365.0, 30.0),
            continuously_compounded_rates=(continuously_compounded_rate,) * 2,
        )

    def rate_at(self, time_to_expiry_years: float) -> float:
        """Tasa cero continua para ese plazo.

        Fuera del rango observado se extiende de forma plana con el punto
        extremo mas cercano. Extrapolar linealmente una curva de tasas
        produce valores absurdos en plazos cortos, que es donde estan la
        mayoria de las opciones que usa este proyecto.
        """
        if time_to_expiry_years <= 0.0:
            raise DomainValidationError(
                f"El plazo debe ser positivo, se recibio {time_to_expiry_years}."
            )
        return float(
            np.interp(
                time_to_expiry_years,
                np.asarray(self.tenors_in_years),
                np.asarray(self.continuously_compounded_rates),
            )
        )

    def discount_factor_at(self, time_to_expiry_years: float) -> float:
        """Valor presente de una unidad pagada en ese plazo."""
        return float(np.exp(-self.rate_at(time_to_expiry_years) * time_to_expiry_years))
