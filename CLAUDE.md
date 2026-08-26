# CLAUDE.md — FINVEX-FIC

Instrucciones operativas. La metodologia esta en `docs/methodology/`; las
decisiones de arquitectura en `docs/decisions/`. Este archivo se mantiene
corto a proposito: un CLAUDE.md largo se ignora.

## Que es esto

Investigacion cuantitativa: momentos implicitos de orden superior (MFIV,
MFIS, MFIK) extraidos de opciones, calibrados de Q a P, e integrados como
views en una extension de Black-Litterman orientada a riesgo de cola.

## Comandos

```
make setup       # uv sync --group dev + pre-commit install
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy sobre src/
make test        # pytest -m "not needs_data"
make test-all    # incluye tests que requieren datos con licencia
make doctor      # entorno y disponibilidad de dependencias
```

## Arquitectura: hexagonal, regla de dependencia unidireccional

```
adapters  ->  ports  ->  domain
application  ->  ports, domain
config  ->  domain
```

- `domain/` no importa nada del proyecto fuera de si mismo, ni pandas de
  disco, ni duckdb, ni clientes de datos. Modelos en OOP, algoritmos en
  estilo funcional sobre arreglos.
- `ports/` son solo `Protocol` y tipos de dominio.
- `application/` compone dominio y puertos. Sin estado global.
- `adapters/` implementa los puertos. **Nadie importa desde `adapters/`**
  salvo `bootstrap.py` y la CLI.

`tests/architecture/` verifica esto. Si tu import rompe la regla, el test
falla: la regla no es una sugerencia.

## Reglas anti-look-ahead

- Toda funcion que lea datos recibe `as_of: AsOf`. Nunca `date.today()`,
  nunca `datetime.now()` en `src/` fuera de la CLI.
- La unica puerta de lectura es el store point-in-time. No se llama a un
  provider directamente desde investigacion.
- `SignalPanel` lleva `information_cutoff`; el backtest verifica
  `information_cutoff <= as_of` en cada rebalanceo y **aborta** si falla.
- El prior de equilibrio recibe `as_of` como cualquier otro dato. Usar la
  media muestral in-sample o capitalizaciones futuras es el error del
  notebook original.

## Convenciones numericas (no negociables)

| Magnitud | Convencion |
|---|---|
| Volatilidad implicita | decimal anualizado (0.185, no 18.5) |
| Tiempo a vencimiento | minutos hasta la liquidacion / 525600 |
| Hora de liquidacion | AM = 9:30 ET, PM = 16:00 ET |
| Tasas | continuas, interpoladas en la curva cero |
| Moneyness | log(K/F) |
| Precios | puntos de indice, sin multiplicador |
| Retornos | simples para portafolio; log solo para momentos realizados |
| Timestamps | tz-aware, UTC en disco, America/New_York al presentar |
| Momentos | MFIS y MFIK estandarizados; MFIK es curtosis, no exceso |

## Restricciones metodologicas

- BKM solo sobre opciones **europeas**. Una cadena americana sin
  de-americanizacion explicita levanta `ExerciseStyleError`.
- La posterior de retornos de Black-Litterman es `Sigma + M`, nunca `M`.
- Omega viene de `domain/bayesian/confidence`, estimada como covarianza del
  error de pronostico. Nunca un numero escrito a mano.
- `moments/vix.py` es un testigo externo: implementa la metodologia de Cboe
  literalmente y no comparte codigo con BKM.

## Reproducibilidad

- Sin `np.random` global: toda aleatoriedad recibe un `Generator` explicito.
- Ningun parametro de investigacion en codigo. Umbrales, ventanas,
  horizontes, niveles y costos van en `configs/*.yaml`.
- Todo artefacto en disco lleva su `.meta.json` con config, commit y hashes.
- El runner de experimentos se niega a correr con el arbol de git sucio.

## Estilo

- Nombres del dominio: `as_of`, `mfis`, `expiry_context`. Nunca `df1`, `tmp`.
- Sin parametros booleanos que cambien el comportamiento: usa Strategy.
- Errores de dominio con nombre propio (`ArbitrageViolationError`), no
  `ValueError` generico ni codigos de retorno.
- Los comentarios explican **por que**, nunca **que**.
- Nada de logica en `notebooks/`: si aparece un `def`, va a `src/`.

## Definicion de terminado

Un modulo no esta listo porque corre. Esta listo cuando su test de
validacion pasa y su decision quedo documentada. Ver los Definition of Done
por fase en `docs/decisions/0001-arquitectura-hexagonal.md`.

## Datos

Ningun proveedor activo todavia. `SyntheticProvider` es la fuente de la fase
1 y de casi todos los tests. **No se inventan campos ni endpoints de ningun
proveedor**: si un campo requerido no existe, el provider declara que no
puede servir esa puerta.
