"""FINVEX-FIC: momentos implicitos de orden superior y riesgo de cola.

La arquitectura es hexagonal. La regla de dependencia es unidireccional:

    adapters  ->  ports  ->  domain
    application -> ports, domain

`domain` no importa nada del proyecto fuera de si mismo. Nadie importa
`adapters` salvo el composition root (`finvex.bootstrap`).
Ver docs/decisions/0001-arquitectura-hexagonal.md
"""

__version__ = "0.0.1"
