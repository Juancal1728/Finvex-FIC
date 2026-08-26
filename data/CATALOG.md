# Catalogo de datos

Los datos **no** se versionan en git. Este archivo es el registro de que
existe, de donde vino y como reproducirlo.

Cada entrada debe llevar: nombre, proveedor, licencia, cobertura temporal,
fecha de descarga, hash del contenido y una nota de que se hizo con el.

| Dataset | Proveedor | Licencia | Cobertura | Descargado | Hash | Uso |
|---|---|---|---|---|---|---|
| _(vacio)_ | | | | | | |

## Ubicacion

La raiz del lake se define en `FINVEX_DATA_ROOT` (ver `.env.example`). Debe
estar **fuera** del repositorio y fuera de carpetas sincronizadas con la
nube: el churn de sincronizacion sobre parquet de varios GB y los archivos
que quedan solo en la nube rompen las lecturas.
