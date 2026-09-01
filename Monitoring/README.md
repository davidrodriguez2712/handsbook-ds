# Monitoring

Etapa de **monitoreo del modelo en producción** del pipeline de `handsbook-ds`. Compara un **baseline** (`expected`, normalmente train/desarrollo) contra los datos observados en producción (`actual`), típicamente **desagregados por período** (una columna tipo `periodo` = `YYYYMM`).

```
Monitoring/
└── utils/
    └── monitor.py     <- todas las funciones de monitoreo (una por dimensión)
```

No hay `__init__.py`; se importa por módulo. `monitor.py` se autoresuelve la raíz del repo vía `Path(__file__)` (funciona desde cualquier cwd); el notebook solo necesita la raíz en el `sys.path`.

## Cómo importar

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[2]))   # ajusta el índice según dónde esté tu notebook

from Monitoring.utils.monitor import (
    monitor_performance_fix_baseline, monitor_missing,
    monitoring_psi_baseline_fix, monitoring_outliers_univariado,
    monitor_iv_baseline_fixed, curves_roc_pr,
    monitor_bad_rate_fix_baseline,
    shap_variacion_global, shap_variacion_global_periodica,
)
```

**Dependencias:** `shap`, `scikit-learn`, `matplotlib`, y el paquete `Feature_Engineering` del monorepo — `monitor.py` reutiliza `AutoOutlierHandler`, `Binning` y `dataframe_psi` de esa carpeta.

## Convenciones

- **`expected` vs `actual`**: `expected` = muestra de referencia (una sola); `actual` = producción, con una columna de período (`col_period` / `period_col` / `period_name`, según la función).
- **Datos ya preprocesados**: `expected` y `actual` deben tener las mismas columnas que vio el modelo (las funciones de performance/SHAP llaman a `predict_proba` / `TreeExplainer` directamente).
- **`dict_transform` / `dict_period`** (opcional): mapeo para mostrar los períodos de forma legible. La función toma los 2 últimos dígitos del período (mes) y los pasa por el dict, p. ej. `{'01': 'Ene', '02': 'Feb', ...}` → etiquetas tipo `Ene'24`.
- **`show_graph` / `show_curve`**: `False` (default) devuelve solo el `DataFrame`; `True` devuelve `(DataFrame, Figure)`.
- **SHAP**: usa `shap.TreeExplainer` ⇒ solo modelos de árbol (XGBoost, LightGBM, CatBoost, RandomForest).

## Funciones

| Función | Dimensión | Firma (resumen) | Devuelve |
|---|---|---|---|
| `monitor_performance_fix_baseline` | Discriminación | `(pipeline_model, expected_x, actual_x, expected_y, actual_y, col_period, show_graph=False)` | DataFrame AUC/Gini: expected global, actual global y actual por período (+ fig de doble eje). |
| `curves_roc_pr` | Discriminación | `(y_true_expected, y_score_expected, y_true_actual, y_score_actual)` | `Figure` con curvas ROC y PR superpuestas (baseline vs actual) y Δ% de AUC/Gini. |
| `monitor_bad_rate_fix_baseline` | Calibración | `(expected_y, expected_pd, actual_y, actual_pd, period_serie, show_graph=False)` | DataFrame bad rate vs PD promedio por período + Brier global (+ fig). |
| `monitoring_psi_baseline_fix` | Estabilidad (input) | `(expected, actual, period_col, show_graph, dict_transform=None)` | DataFrame PSI por feature y período (bins fijados en `expected`, vía `dataframe_psi`); fig con líneas de alerta en 0.10 y 0.25. |
| `monitor_missing` | Calidad de datos | `(X, period_name, dict_period=None)` | `Figure`: evolutivo del % de missing por feature (solo las que tienen missing). |
| `monitoring_outliers_univariado` | Calidad de datos | `(expected, actual, period_col, show_graph=False, dict_transform=None)` | DataFrame: variación (abs.) del % de outliers IQR de cada feature vs el baseline, por período. Los límites IQR se aprenden con `AutoOutlierHandler` sobre `expected` (+ fig). |
| `monitor_iv_baseline_fixed` | Poder predictivo | `(expected, actual, y_actual, period_col, show_graph=False, dict_transform=None)` | DataFrame IV por feature y período, con el binning fijado en `expected` (vía `Binning`); fig con umbrales (+ requiere el target `y_actual`). |
| `shap_variacion_global` | Explicabilidad | `(model, expected, actual, n_sample=5000, show_graph=False, random_state=42)` | DataFrame: contribución SHAP media \|·\| por feature en expected vs actual, diferencia y variación % (+ fig de barras). |
| `shap_variacion_global_periodica` | Explicabilidad | `(model, expected, actual, period_col, n_sample=5000, show_graph=False, random_state=42)` | DataFrame: SHAP medio por feature y período (Baseline + cada período de `actual`); fig evolutiva. |

Helper interno: `threshold_f1(precision, recall)`.

## Uso típico

```python
model = joblib.load("artifacts/training_model/pipeline_final.pkl")

# discriminación por período
df_perf, fig = monitor_performance_fix_baseline(
    model, X_dev, X_prod, y_dev, y_prod, col_period="periodo", show_graph=True)

# estabilidad de las variables
df_psi, fig = monitoring_psi_baseline_fix(
    X_dev, X_prod, period_col="periodo", show_graph=True,
    dict_transform={"01":"Ene","02":"Feb", ...})

# calibración: bad rate observado vs PD
df_br, fig = monitor_bad_rate_fix_baseline(
    y_dev, pd_dev, y_prod, pd_prod, period_serie=X_prod["periodo"], show_graph=True)

# drift de explicabilidad
df_shap, fig = shap_variacion_global(model, X_dev, X_prod, show_graph=True)
```

## Notas

- Las funciones `*_fix_baseline` / `*_baseline_fix(ed)` congelan las reglas (bins, límites, explainer background) en `expected` y solo las **aplican** a `actual` — es intencional: se mide el drift contra un patrón fijo.
- `shap_variacion_global_periodica` submuestrea a 400 filas por período (además del `n_sample` del background); para carteras pequeñas puede ser ruidoso.
- Convención del repo: comentarios, docstrings y nombres de variables en **español**.
