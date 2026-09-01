# Feature Engineering

Etapa de **ingeniería y selección de variables** del pipeline de `handsbook-ds`, orientada a **clasificación binaria** (riesgo / scoring). El objetivo de la carpeta es alimentar un `Pipeline` / `ColumnTransformer` de scikit-learn con transformers reutilizables y funciones de selección de features.

```
Feature_Engineering/
├── conceptos.md                 <- notas teóricas (vacío por ahora)
├── examples/                    <- notebooks de ejemplo (vacío; carpeta ignorada por git)
└── utils/
    ├── transformers/            <- transformers compatibles con sklearn (fit / transform)
    │   ├── woe.py               <- WOEClassic, Logit_Smoothing_Rolling
    │   ├── binning.py           <- Binning
    │   ├── missing.py           <- AutoMissingHandler, AutoInteractiveImputer
    │   ├── outliers.py          <- AutoOutlierHandler
    │   ├── scaling.py           <- ScalingMethods
    │   ├── tranformations.py    <- TransformationMethods  (sic: "tranformations")
    │   ├── encoders.py          <- (vacío)
    │   └── domain.py            <- (vacío)
    ├── selection/               <- funciones de selección + un transformer final
    │   ├── correlation.py       <- correlation()
    │   ├── vif.py               <- vif()
    │   ├── iv.py                <- information_value()
    │   ├── gini.py              <- gini_univariate_features(), gini_missing_rule(), gini_rule_selection()
    │   ├── psi.py               <- calculate_psi(), interpret_psi(), psi_numeric(), psi_categorical(), dataframe_psi()
    │   ├── mutual_info.py       <- (solo imports, vacío)
    │   └── feature_selector.py  <- FeatureSelector
    ├── reports/                 <- feature_report.py, woe_report.py  (vacíos)
    ├── binary_classification.py <- (vacío)
    └── __init__.py              <- vacío; importa siempre por módulo
```

> **Estado.** El bloque de `selection/` está operativo y es lo más maduro. En `transformers/`, `AutoMissingHandler`, `AutoOutlierHandler`, `ScalingMethods`, `TransformationMethods` y `Logit_Smoothing_Rolling` funcionan; **`WOEClassic` y `Binning` están a medio construir** (ver notas al final). `domain.py`, `encoders.py`, `mutual_info.py`, `reports/` y `binary_classification.py` son stubs vacíos.

---

## Cómo importar

Igual que el resto del monorepo: se agrega la raíz a `sys.path` (asume cwd = `.../<proyecto>/notebooks/`, 3 niveles bajo la raíz). Varios módulos hacen internamente `from CORE.metadata import inventario_features`, así que esa suposición de cwd debe cumplirse.

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent.parent))

from Feature_Engineering.utils.transformers.woe import Logit_Smoothing_Rolling
from Feature_Engineering.utils.transformers.outliers import AutoOutlierHandler
from Feature_Engineering.utils.selection.gini import gini_rule_selection
from Feature_Engineering.utils.selection.feature_selector import FeatureSelector
```

---

## `utils/transformers/` — transformers sklearn

Todos siguen el patrón `BaseEstimator, TransformerMixin` (fit/transform) para poder componerse en `Pipeline` / `ColumnTransformer`. **La mayoría no tiene defaults en `__init__`**: hay que pasar todos los argumentos.

### `missing.py`

| Clase | `__init__` | Qué hace |
|---|---|---|
| `AutoMissingHandler` | `strategy_numeric, strategy_categorical, fill_value, missing_indicator, return_dataframe` | Imputa numéricas y categóricas por separado con `SimpleImputer` (o `StandardScaler`+`KNNImputer` si `strategy_numeric='knn'`). Con `missing_indicator=True` añade columnas `<col>_missing_indicator`. Devuelve DataFrame o `np.ndarray` según `return_dataframe`. |
| `AutoInteractiveImputer` | `algorithm, max_iter, random_state, initial_strategy, return_dataframe` | Imputación multivariante iterativa (`IterativeImputer`) sobre las features numéricas, con `algorithm` como estimador base. |

### `outliers.py`

| Clase | `__init__` | Qué hace |
|---|---|---|
| `AutoOutlierHandler` | `method` (`'iqr'` \| `'percentile'`), `strategy=None` (`'windsorization'`), `return_dataframe=True`, `exclude_continuous=False` | En `fit` aprende límites por feature (IQR = Q±1.5·IQR; percentil adaptativo al nº de filas). En `transform`, con `strategy='windsorization'`, hace `np.clip` a esos límites. Usa `inventario_features` para elegir features `Continua`/`Conteo` (con `exclude_continuous=True` solo `Continua`). |

### `scaling.py`

| Clase | `__init__` | Qué hace |
|---|---|---|
| `ScalingMethods` | `strategy` (`'standarscaler'` \| `'robustscaler'` \| `'minmaxscaler'`, sic), `return_dataframe`, `feats_num` (lista de columnas a escalar) | Aplica el scaler de sklearn correspondiente a `feats_num`. `summary(X)` devuelve una figura antes/después por feature. |

### `tranformations.py`

| Clase | `__init__` | Qué hace |
|---|---|---|
| `TransformationMethods` | `strategy` (`'yeo-johnson'` \| `'box-cox'` \| `'log1p'`), `features_num` (lista), `return_dataframe=True` | Transformaciones de potencia / logarítmica sobre `features_num` (`PowerTransformer` o `FunctionTransformer(log1p)`). `summary(X)` → figura antes/después. **Bug de import** en la línea 7 (`from matplotlib.pyplot import pyplot as plt`); corregir a `import matplotlib.pyplot as plt` antes de usar. |

### `woe.py`

| Clase | `__init__` | Qué hace |
|---|---|---|
| `WOEClassic` | `strategy, return_dataframe` | WOE clásico por binning (numéricas: `pd.cut`; categóricas: por nivel). **Incompleto / con bugs** — no usar todavía (ver notas). |
| `Logit_Smoothing_Rolling` | `return_dataframe, show_details=False` | Alternativa a WOE: para cada feature numérica calcula el *log-odds* del target con media móvil ordenada por la feature, ajusta funciones candidatas (polinomios grado 1-3 y logarítmica), elige la de mejor AUC en validación y la usa como transformación `feature → log_odds`. En `fit` hace su propio `train_test_split` interno (30% validación). `transform` devuelve columnas `<feat>_log_odds`. Maneja NaN con el log-odds del subgrupo faltante. |

### `binning.py`

`Binning(strategy='width'|'frecuency', show_dataframe=True)` — **no es un transformer estándar**: `transform(X, y, operation='iv')` requiere `y` y hoy solo sabe calcular IV (delega en `selection.iv.information_value`). Sirve como explorador de binning + IV, y lo reutiliza `Monitoring`. Tiene un bug conocido (`labels.append(...)` sobre el resultado de `.append`, líneas 54-55).

---

## `utils/selection/` — selección de features

Funciones puras (reciben `X` / `y`), devuelven `DataFrame` ordenado por relevancia. No modifican los datos.

| Módulo | Función | Firma | Devuelve |
|---|---|---|---|
| `correlation.py` | `correlation(X, method='pearson')` | | DataFrame de **pares** de features numéricas y su correlación, ordenado desc. |
| `vif.py` | `vif(X)` | | DataFrame `feature, VIF` (agrega constante, `dropna` de filas). VIF alto ⇒ multicolinealidad. |
| `iv.py` | `information_value(X, y, feat_bins, feat_numeric, feat_cat)` | | DataFrame `feature, iv`. `feat_bins` es el dict de cortes que produce `Binning`; agrupa NaN como `MISSING` y niveles no vistos como `UNKNOWN`. |
| `gini.py` | `gini_univariate_features(X, y_true)` | | DataFrame `feature, roc_auc, gini` univariado (numéricas: score directo; categóricas: vía target mean → log-odds). |
| `gini.py` | `gini_missing_rule(gini, missing)` | | `str`: `'Mantener'` / `'Revisar'` / `'Eliminar'` según umbrales de gini y % missing. |
| `gini.py` | `gini_rule_selection(X, y_true)` | | Cruza `gini_univariate_features` con `inventario_features` y aplica `gini_missing_rule` por feature → DataFrame con columna `regla`. |
| `psi.py` | `calculate_psi(expected, actual)` | arrays de proporciones | `float` PSI. |
| `psi.py` | `interpret_psi(psi)` | | `str`: `"Sin drift"` (<0.10) / `"Drift moderado"` (<0.25) / `"Drift severo"`. |
| `psi.py` | `psi_numeric(train, test, bins=10)` / `psi_categorical(train, test)` | | `float` PSI de una variable (bins por cuantiles del train / por categorías). |
| `psi.py` | `dataframe_psi(train_df, test_df, period_name=None, return_dict=False)` | | PSI de todas las columnas: DataFrame ordenado por PSI, o lista de dicts si `return_dict=True` (lo usa `Monitoring`). |

### `feature_selector.py`

```python
FeatureSelector(features_selected: list)
```
Transformer final del pipeline: en `transform` devuelve `X[features_selected]`. Se usa como último paso para congelar la lista de features que entra al modelo.

---

## Uso típico (esquema de pipeline)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ("missing",  AutoMissingHandler("median", "most_frequent", None, True, True)),
    ("outliers", AutoOutlierHandler(method="iqr", strategy="windsorization")),
    ("woe",      Logit_Smoothing_Rolling(return_dataframe=True)),
    ("select",   FeatureSelector(features_selected=["f1_log_odds", "f2_log_odds", ...])),
])
pipe.fit(X_train, y_train)
X_train_fe = pipe.transform(X_train)
# joblib.dump(pipe, "artifacts/feature_engineering/preprocessing.pkl")
```

Las funciones de `selection/` se usan **fuera** del pipeline, para decidir esa lista de features:

```python
gini_rule_selection(X_train, y_train)                 # regla negocio por gini + missing
vif(X_train)                                          # multicolinealidad
dataframe_psi(X_train, X_test)                         # estabilidad train vs test
```

---

## Notas y limitaciones conocidas

- **`WOEClassic` no está terminado**: en `fit` el bucle de "continuas" itera sobre `FEATS_NUM_CONTEO` (copy-paste), `FEATS_NUM` es un DataFrame y se usa como lista, el mapping categórico se guarda como `set` en vez de `dict`, y usa una sintaxis de `groupby().agg((nombre, func))` que no es la API actual de pandas. Usar `Logit_Smoothing_Rolling` mientras tanto.
- **`Binning.transform` rompe el contrato de sklearn** (pide `y`, solo hace IV) — no lo pongas dentro de un `Pipeline`.
- **`tranformations.py`**: import inválido de matplotlib (línea 7).
- `Logit_Smoothing_Rolling` hace un `train_test_split` **interno** en `fit`: el modelo se entrena solo con el 70% de lo que le pases. Tenlo en cuenta si ya venías con tu propio split.
- `correlation()` ordena por correlación con signo (no por valor absoluto): las correlaciones negativas fuertes quedan al final de la tabla.
- Convención del repo: comentarios, docstrings y nombres de variables en **español**.
