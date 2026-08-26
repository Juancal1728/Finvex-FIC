"""Errores de dominio.

Cada error nombra un problema del negocio, no una condicion tecnica. Un
`ValueError` generico obliga a leer el traceback para entender que paso;
`ArbitrageViolationError` se entiende en el mensaje.
"""

from __future__ import annotations


class FinvexError(Exception):
    """Raiz de todos los errores del proyecto."""


class DomainValidationError(FinvexError):
    """Un objeto de dominio se intento construir en un estado invalido."""


class LookAheadError(FinvexError):
    """Se intento usar informacion posterior a la fecha de formacion.

    Es un error, nunca una advertencia: un backtest que continua tras esto
    produce un resultado sin significado.
    """


class ArbitrageViolationError(FinvexError):
    """La superficie o la cadena viola una condicion de no-arbitraje estatico."""


class ExerciseStyleError(FinvexError):
    """Se intento aplicar BKM sobre opciones americanas sin de-americanizar.

    BKM se apoya en el spanning estatico de payoffs europeos. Aplicarlo sobre
    primas americanas incorpora el premio de ejercicio anticipado a los
    momentos extraidos, con sesgo sistematico.
    """


class ProviderUnavailableError(FinvexError):
    """El proveedor de datos existe pero no puede servir lo que se le pide."""


class ConfigurationError(FinvexError):
    """La configuracion es invalida o incompleta."""
