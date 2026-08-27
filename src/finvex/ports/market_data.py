"""Puerto de datos de mercado: el contrato que el nucleo le pide al exterior.

Este archivo es la frontera de la arquitectura hexagonal. El nucleo de
investigacion sabe que existe *algo* capaz de entregarle una cadena de
opciones y una curva de tasas para una fecha dada; no sabe si ese algo es
OptionMetrics, Cboe, Refinitiv o un generador sintetico, y no debe saberlo.

Se define como `Protocol` y no como clase base abstracta a proposito. Un
`Protocol` da tipado estructural: un adaptador cumple el contrato por tener
los metodos correctos, sin heredar de nada. Eso evita que el nucleo tenga que
exportar una clase base que los adaptadores importen, lo que crearia una
dependencia en el sentido contrario al que manda la regla de la arquitectura.

Las tres reglas que todo adaptador debe respetar:

1. **Devolver solo tipos de dominio.** Nada de DataFrames del proveedor,
   nada de diccionarios crudos, nada de objetos de la libreria del vendor.
   La traduccion es responsabilidad del adaptador.
2. **No inventar campos.** Si el proveedor no entrega open interest, el
   campo va en `None`. Rellenarlo con un valor plausible convierte una
   ausencia de dato en un dato falso, y aguas abajo nadie puede distinguirlo.
3. **Declarar disponibilidad honestamente.** `is_available()` debe hacer una
   comprobacion real. Un adaptador que dice estar disponible y falla a mitad
   de una corrida de dos horas es peor que uno que dice no estarlo.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

from finvex.domain.model.option_chain import OptionChain
from finvex.domain.model.rate_curve import RateCurve
from finvex.domain.values.as_of import AsOf


@runtime_checkable
class OptionsDataProvider(Protocol):
    """Fuente de cadenas de opciones y curvas de tasas."""

    @property
    def provider_name(self) -> str:
        """Nombre y version del proveedor. Queda registrado en la procedencia."""
        ...

    def is_available(self) -> bool:
        """True si el proveedor puede servir datos ahora mismo.

        Debe hacer una comprobacion efectiva: credenciales presentes,
        entitlements activos, archivos accesibles. No basta con devolver True.
        """
        ...

    def available_underlying_symbols(self) -> tuple[str, ...]:
        """Subyacentes que este proveedor puede servir, en simbolos canonicos."""
        ...

    def available_observation_dates(self, underlying_symbol: str) -> tuple[dt.date, ...]:
        """Fechas de observacion disponibles, en orden creciente."""
        ...

    def option_chain(self, as_of: AsOf, underlying_symbol: str) -> OptionChain:
        """Cadena completa de un subyacente en una fecha de observacion."""
        ...

    def rate_curve(self, as_of: AsOf) -> RateCurve:
        """Curva cero vigente en esa fecha de observacion."""
        ...
