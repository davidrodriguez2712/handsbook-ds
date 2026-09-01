# EDA

Etapa de **Análisis Exploratorio de Datos** del pipeline de `handsbook-ds`, orientada a problemas de **clasificación binaria** (riesgo / scoring).

```
EDA/
├── examples/                     <- notebooks de ejemplo + datasets de muestra
│   └── classification_example.ipynb
└── utils/                        <- todo el código de la etapa (este README lo documenta)
    ├── classifications.py        <- clase EDAClasificacion  ← RUTA ESTABLE, la usan los notebooks
    ├── overview.py               <- refactor funcional (data overview, con soporte temporal)
    ├── missing.py                <- refactor funcional (análisis de missings)
    ├── outliers.py               <- refactor funcional (detección de outliers)
    ├── distribution.py           <- refactor funcional (distribuciones + target)
    ├── correlation.py            <- refactor funcional (IV + correlaciones/MI/VIF)
    └── __init__.py               <- vacío; importa siempre por módulo
```

> **Estado del código.** `classifications.py` (clase `EDAClasificacion`) es la implementación **completa y probada**; es lo que importan los notebooks de los proyectos. Los módulos sueltos (`overview.py`, `missing.py`, `outliers.py`, `distribution.py`, `correlation.py`) son una **refactorización en curso** que parte esa clase-monolito en funciones independientes: hoy están **incompletos y con algunos bugs conocidos** (ver notas al final).

---

## Cómo importar

Los notebooks resuelven la raíz del monorepo y la agregan a `sys.path` antes de importar (ver `CLAUDE.md` de la raíz):

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent.parent))   # cwd == .../<proyecto>/notebooks/

from EDA.utils.classifications import EDAClasificacion
```

---

## `classifications.py` — clase `EDAClasificacion`

Cubre **toda la etapa EDA** en una sola clase. Cada método devuelve un `DataFrame` (tabla para inspección/guardado) o una `Figure` de matplotlib (ya renderizada con `plt.close()`, lista para `display()` o `fig.savefig(...)`).

### Construcción

```python
eda = EDAClasificacion(
    data=df,                              # DataFrame con features + target
    target_name="SeriousDlqin2yrs",       # nombre de la columna target (binaria 0/1)
    tipo_problema="Clasificación Binaria" # solo etiqueta informativa
)
```

### Métodos

| Método | Devuelve | Qué hace |
|---|---|---|
| `data_overview()` | `DataFrame` (1 col, transpuesto) | Resumen global: nº filas/columnas, conteo de features por tipo (numérica / categórica / datetime / booleana), missing global, features con missing y con >20% missing, duplicados, constantes y casi-constantes, alta/baja cardinalidad, target rate, no-event rate y ratio de desbalanceo. |
| `inventario_features()` | `DataFrame` (una fila por feature) | Ficha por columna: `dtype`, `semantic_dtype` (`Conteo` / `Continua` / `String` / `Datetime`), `missing_pct`, `unique_values`, `cardinality_group` (Bajo/Medio/Alto), `is_constant`, `is_near_constant`. Ordenado por % missing. |
| `missings_analisis()` | `Figure` | 3 paneles: heatmap de nulos por fila, correlación de nulidad entre variables (`missingno`), barras de cantidad de missing por feature. |
| `outlier_detect_iqr()` | `DataFrame` | Límites por regla **IQR** (Q1−1.5·IQR, Q3+1.5·IQR) y nº/% de registros fuera de rango por feature numérica. **Excluye** las features `semantic_dtype == 'Conteo'`. |
| `outlier_detect_percentil()` | `DataFrame` | Igual pero con corte por **percentiles adaptativos al tamaño**: <5 000 filas → P5/P95; <100 000 → P1/P99; resto → P0.5/P99.5. También excluye `Conteo`. |
| `numeric_distributions()` | `Figure` | Por cada feature numérica: histograma, event-rate del target por decil (`qcut` q=10), y boxplot por clase del target. |
| `categorical_distributions()` | `Figure` \| `None` | Por cada feature categórica: target rate por categoría y conteo de cada categoría. Devuelve `None` e imprime aviso si no hay categóricas. |
| `analisis_target()` | `(Figure, DataFrame)` | Gráfico de conteo + pie del target, y tabla con positive/negative rate, imbalance ratio, baseline accuracy y "nivel de desbalance" (Muy Desbalanceado / Desbalanceado / Moderado / Balanceado). |
| `information_value()` | `(DataFrame, list)` | **IV** de cada feature numérica (binning `qcut` q=5, WOE con `eps` anti-log(0)). Devuelve la tabla de IV ordenada + la lista de arrays de bins usados por feature. |
| `eda_relaciones_feats_numericas()` | `Figure` | 5 paneles: heatmap Pearson, heatmap Spearman, Mutual Information (`mutual_info_classif`), VIF (línea de corte en 5) e IV (línea de corte en 0.1). |
| `run_all(data_comparacion=None, include_psi=False, show=True)` | `dict` | Ejecuta todos los métodos anteriores y devuelve `{titulo: salida}`. Con `show=True` los imprime con `display()`. Con `include_psi=True` añade el bloque PSI y **requiere** pasar `data_comparacion` (un segundo DataFrame, p. ej. test/OOT). |

### Submódulo PSI (drift entre dos muestras)

Pensado para comparar train vs test / vs out-of-time. `self.data` actúa como muestra base ("expected").

| Método | Devuelve | Qué hace |
|---|---|---|
| `calculate_psi(expected, actual)` | `float` | PSI entre dos vectores de proporciones (con `EPSILON` anti-log(0)). |
| `interpret_psi(psi)` | `str` | `"Sin drift"` (<0.10), `"Drift moderado"` (<0.25), `"Drift severo"` (≥0.25). |
| `psi_numeric(train, test, bins=10)` | `float` | PSI de una variable numérica usando cuantiles calculados sobre `train`. |
| `psi_categorical(train, test)` | `float` | PSI de una variable categórica (los nulos se agrupan como `"MISSING"`). |
| `dataframe_psi(test_df)` | `DataFrame` | PSI de **todas** las columnas: `self.data` vs `test_df`, con tipo detectado, PSI e interpretación, ordenado desc. |

### Uso típico

```python
eda = EDAClasificacion(df_train, target_name="SeriousDlqin2yrs")

eda.data_overview()
eda.inventario_features()
fig = eda.missings_analisis();            fig.savefig("artifacts/eda/figures/missings.png", bbox_inches="tight")
eda.outlier_detect_iqr()
df_iv, bins = eda.information_value()

# todo de una vez, incluyendo PSI contra el test
resultados = eda.run_all(data_comparacion=df_test, include_psi=True)
```

---

## Módulos funcionales (refactor en curso)

Misma lógica que los métodos homónimos de `EDAClasificacion`, pero como **funciones puras** que reciben el `DataFrame` por argumento. Útiles cuando no quieres instanciar la clase o quieres componerlos sueltos.

### `overview.py`

| Función | Firma | Devuelve | Notas |
|---|---|---|---|
| `get_first_datetime(data)` | `(DataFrame) -> Series \| None` | primera columna datetime o `None` | helper interno |
| `data_overview(data, target_name, tipo_problema)` | | `DataFrame` transpuesto | como `EDAClasificacion.data_overview()` **más** `Fecha Min`, `Fecha Max` y `N° Períodos` (meses) si hay columna datetime. |

### `missing.py`

| Función | Firma | Devuelve |
|---|---|---|
| `missings_analisis(data)` | `(DataFrame) -> Figure` | idéntico al método de la clase (3 paneles). |

### `outliers.py`

| Función | Firma | Devuelve |
|---|---|---|
| `outlier_detect_iqr(data, excluir_conteo=False)` | | `DataFrame` de límites IQR y outliers |
| `outlier_detect_percentil(data, excluir_conteo)` | | `DataFrame` de límites por percentil |

Con `excluir_conteo=True` descarta las features de tipo semántico `Conteo`. **Ojo:** ese camino hoy está roto (llama `inventario_features()` sin pasarle `data`); úsalo con `excluir_conteo=False` hasta que se corrija.

### `distribution.py`

| Función | Firma | Devuelve |
|---|---|---|
| `numeric_distributions(data, target_name)` | | `None` (dibuja con `plt.show()`) |
| `categorical_distributions(data, target_name, exclude=[])` | | `None` (dibuja con `plt.show()`); `exclude` = lista de columnas a saltar |
| `analisis_target(data, target_name)` | | `(Figure, DataFrame)` |

### `correlation.py`

| Función | Firma | Devuelve |
|---|---|---|
| `information_value(data, target_name)` | | `(DataFrame IV, list de bins)` |
| `eda_relaciones_feats_numericas(data, target_name)` | | `Figure` (Pearson, Spearman, MI, VIF, IV) |

---

## Notas y limitaciones conocidas

- **Target binario 0/1 obligatorio.** `analisis_target` y varios gráficos asumen exactamente dos clases codificadas `0`/`1`; el pie usa labels fijas `"No default"/"Default"`.
- **Hardcodes pendientes de limpiar:** `EDAClasificacion.numeric_distributions` y `distribution.numeric_distributions` filtran literalmente `feat != 'churn'` en vez de `target_name`; varios `figsize` están fijos y se deforman con muchas features.
- **IV / distribuciones numéricas** aplican `qcut` a todas las numéricas: features con muchos valores repetidos pueden colapsar bins (`duplicates='drop'`) y dar IV poco fiable.
- **`outliers.py` (funcional):** la firma `outlier_detect_percentil(data, excluir_conteo)` no tiene default real para el segundo argumento; hay que pasarlo siempre.
- Al editar, respeta la convención del repo: comentarios, docstrings y nombres de variables en **español**.
