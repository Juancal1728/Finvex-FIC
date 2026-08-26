# ADR 0001 — Arquitectura hexagonal con nucleo puro

- **Fecha:** 2026-08-26
- **Estado:** aceptada
- **Contexto:** fase 0 del blueprint tecnico

## Contexto

FINVEX-FIC tiene una dependencia externa sin resolver: la fuente de datos de
opciones. Al momento de esta decision no hay acceso confirmado a
OptionMetrics/WRDS ni a Cboe DataShop; existe la posibilidad de conectar
Refinitiv/LSEG Workspace, cuya cobertura historica de cadenas de opciones
esta por verificar.

Ademas el proyecto atraviesa cuatro literaturas (extraccion libre de modelo,
econometria predictiva, portafolio bayesiano, optimizacion de cola) y debe
producir resultados trazables y defendibles academicamente durante varios
anos, con un solo investigador.

## Decision

Arquitectura hexagonal (puertos y adaptadores) con regla de dependencia
unidireccional hacia el dominio:

```
adapters -> ports -> domain
application -> ports, domain
```

- **`domain/`** — nucleo puro, sin I/O y sin dependencias de infraestructura.
  Modelos y value objects en orientacion a objetos, con invariantes en el
  constructor. Algoritmos en estilo funcional sobre arreglos numericos.
- **`ports/`** — interfaces (`Protocol`) de lo que el nucleo necesita del
  exterior: datos, persistencia, solvers.
- **`application/`** — casos de uso; componen dominio y puertos.
- **`adapters/`** — implementaciones concretas, intercambiables.
- **`bootstrap.py`** — composition root; el unico modulo que conoce las
  implementaciones concretas.

## Consecuencias

**A favor**

- La fuente de datos deja de ser una decision bloqueante: se implementa
  `SyntheticProvider` primero y las fases 1 a 4, incluido el checkpoint de
  validacion de BKM, se construyen y validan sin ningun dato real.
- Cambiar de OptionMetrics a Cboe, Refinitiv o Massive toca un adaptador y
  un archivo de configuracion, nunca el nucleo.
- El dominio se prueba sin montar datasets: los tests corren en milisegundos
  y pueden ejecutarse en CI, donde los datos con licencia no pueden estar.
- Los algoritmos quedan aislados de la representacion tabular, que es lo que
  permite compararlos entre si de forma limpia (el analisis de sensibilidad
  del Eje 1 depende de esto).

**En contra**

- Mas indireccion que un script. Se acepta porque el proyecto vive anos y
  debe defenderse, no ejecutarse una vez.
- Riesgo de sobreingenieria. Se mitiga con la regla del presupuesto de
  patrones (abajo).

## Patrones de diseno adoptados, y presupuesto

Se aplican deliberadamente y con limite. Un patron entra cuando resuelve un
problema **ya presente**, no por anticipacion.

| Patron | Donde | Problema que resuelve |
|---|---|---|
| Puertos y adaptadores | estructura general | la fuente de datos no esta decidida |
| Strategy | interpolacion, integracion, escenarios, optimizador | comparar metodologias es el objeto de la tesis |
| Registry / Factory | resolucion nombre-en-YAML -> clase | el barrido de sensibilidad se ejecuta por configuracion |
| Repository | `PointInTimeStore` | una sola puerta de lectura hace estructural el anti-look-ahead |
| Specification | reglas de calidad de datos | cada exclusion debe ser auditable por separado |
| Builder | `ScenarioSet` | construccion con restricciones que deben verificarse al final |
| Value Object | `AsOf`, volatilidad, moneyness, tenor | las unidades son la fuente de error mas comun del area |
| Null Object | providers no disponibles | evita comprobaciones de `None` regadas por el codigo |

**Explicitamente descartados:** Singleton (rompe testabilidad y
determinismo), herencia profunda (se prefiere composicion), bus de eventos
(no hay un problema que lo pida), y crear una interfaz por clase — una
interfaz se justifica con dos implementaciones reales o con la necesidad de
un doble de prueba.

## Verificacion

La regla no es documental. `tests/architecture/` la comprueba:

- `test_dependency_rules.py` — la regla de dependencia y la prohibicion de
  infraestructura dentro del dominio.
- `test_no_hidden_state.py` — sin reloj del sistema en codigo de
  investigacion, sin aleatoriedad global.

## Alternativas consideradas

- **Paquete plano por tema** (`data/`, `surfaces/`, `moments/`, ...): mas
  simple, pero acopla investigacion a proveedor. Descartada por la
  incertidumbre de la fuente de datos.
- **Arquitectura en capas clasica sin puertos**: no permite sustituir el
  proveedor sin tocar consumidores.
- **Notebooks + libreria de utilidades**: es el punto de partida que el
  diagnostico identifico como no defendible.
