# FINVEX-FIC

Contenido informacional de los momentos de orden superior implicitos del
mercado de opciones y su integracion en la optimizacion bayesiana de
portafolios mediante una extension de Black-Litterman.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Repositorio, entorno, CI, guards de arquitectura | **en curso** |
| 1 | Contratos de datos, provider sintetico, store point-in-time | pendiente |
| 2 | Capa de opciones y motor de calidad | pendiente |
| 3 | Motor de superficie de volatilidad | pendiente |
| 4 | BKM + checkpoint V1 (momentos sinteticos exactos) | pendiente |
| 5 | Datos reales + V2 (VIX) + V3 (qmoms) | bloqueada por decision de datos |
| 6 | Econometria Q->P y senales | pendiente |
| 7 | Black-Litterman baseline y motor de Omega | pendiente |
| 8 | Escenarios y optimizador CVaR | pendiente |
| 9 | Backtest y benchmarks | pendiente |
| 10 | Evaluacion, inferencia y robustez | pendiente |

## Puesta en marcha

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11 o superior.

```bash
git clone https://github.com/Juancal1728/Finvex-FIC.git
cd Finvex-FIC
cp .env.example .env      # y ajustar FINVEX_DATA_ROOT
make setup
make test
make doctor
```

En PyCharm: apuntar el interprete del proyecto al `.venv/` que crea `uv`,
no a un SDK de otro proyecto.

## Arquitectura

Hexagonal, con la regla de dependencia verificada por test:

```
src/finvex/
  domain/       nucleo puro. Modelos en OOP, algoritmos funcionales. Sin I/O.
  ports/        interfaces (Protocol) que el nucleo necesita del exterior.
  application/  casos de uso que componen dominio y puertos.
  adapters/     implementaciones concretas: datos, persistencia, solvers, CLI.
  config/       carga y validacion de configuracion.
```

`tests/architecture/` falla si un import rompe la regla, si aparece el reloj
del sistema en codigo de investigacion, o si se usa aleatoriedad global.

## Documentacion

- `docs/decisions/` — decisiones de arquitectura y metodologia (ADR)
- `docs/methodology/` — la matematica de cada modulo
- `docs/data_dictionary/` — cada campo: tipo, unidad, convencion, rango
- `docs/rejected/` — metodologias descartadas y por que
- `CLAUDE.md` — reglas operativas del repositorio

## Datos

Los datos **no** viven en el repositorio. `FINVEX_DATA_ROOT` apunta a un
directorio fuera de carpetas sincronizadas con la nube. `data/CATALOG.md`
registra que dataset hay, de donde vino, con que licencia y con que hash.
