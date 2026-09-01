# Training Model

Etapa de **entrenamiento y calibración de modelos** del pipeline de `handsbook-ds` (clasificación binaria de riesgo / scoring). Cubre: baseline de algoritmos, búsqueda de hiperparámetros con Optuna, métricas de discriminación / negocio, y calibración de la probabilidad de default (PD).

```
Training_Model/
└── utils/
    ├── algorithm.py        <- initial_models()        (baseline CV de varios algoritmos)
    ├── optuna_tunning.py   <- class OptunaTunning     (tuning de hiperparámetros)  (sic: "tunning")
    ├── metrics.py          <- métricas de discriminación, deciles, KS, lift y negocio
    └── calibration.py      <- curvas de calibración, Platt scaling, intervalos de Vasicek
```

No hay `__init__.py`; se importa por módulo. Igual que el resto del repo, los imports asumen cwd = `.../<proyecto>/notebooks/` (3 niveles bajo la raíz del monorepo).

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent.parent))

from Training_Model.utils.algorithm import initial_models
from Training_Model.utils.optuna_tunning import OptunaTunning
from Training_Model.utils import metrics, calibration
```

**Dependencias:** `xgboost`, `lightgbm`, `catboost`, `optuna`, `scikit-learn`, `scipy`, `joblib`.

**Convención de scores:** en casi todo el módulo `y_score` es la **PD** (probabilidad de default / de la clase positiva = "malo"). Las tablas de discriminación (`ks_table`, `tabla_deciles_completa`, ...) ordenan por `-y_score`, así que **decil 1 = mayor riesgo**. Ojo: `bad_rate_decil`, `lift_bandas`, `seleccion_deciles_agrupamiento` ordenan por `y_score` ascendente (decil 1 = menor riesgo). Revisa el sentido del decil en cada función antes de comparar tablas.

---

## `algorithm.py`

```python
initial_models(
    X_linear, X_trees, y,
    algorithms_linear = [LogisticRegression],
    algorithms_trees  = [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier],
    pipeline_linear = False,
    random_seed = 42,
) -> pd.DataFrame
```

Baseline rápido: para cada algoritmo corre `cross_val_score` (5-fold estratificado, `scoring='roc_auc'`) y devuelve un DataFrame con `AUC (cv)`, `Std AUC`, `Max AUC`, `Min AUC`, ordenado por AUC desc.

- `X_linear` / `X_trees`: permite pasar dos representaciones distintas de las features (p. ej. WOE para el lineal, crudas para los árboles).
- `pipeline_linear=True`: envuelve el modelo lineal en imputación + escalado (+ OHE si hay categóricas).
- CatBoost recibe las categóricas vía `cat_features`; el resto de árboles usan OHE si hay categóricas.
- **Bug conocido:** la variable `pipeline_linear` se reutiliza como flag y como objeto pipeline dentro del bucle; con `pipeline_linear=True` y varios algoritmos lineales el comportamiento a partir del 2º es inconsistente. Úsalo con `pipeline_linear=False` (modelos con params por defecto) o con un solo algoritmo lineal.

---

## `optuna_tunning.py`

```python
OptunaTunning(
    nombre_proyecto, X, y, algoritmos: list, params: dict,
    preprocessor_linear,
    activate_preprocessor_trees = False, preprocessor_trees = None,
    default_params = False, random_state = 42, n_trials = 50,
)
```

Búsqueda de hiperparámetros con Optuna, un `study` por algoritmo (`direction='maximize'`, objetivo = AUC media en CV 5-fold estratificado, `TPESampler`).

| Método | Devuelve | Qué hace |
|---|---|---|
| `optimize()` | `dict` | `{model_name: {'best_params': ..., 'best_value': ...}}`. Es el método que se llama. |
| `objective(trial, model)` | `float` | AUC CV del pipeline construido para ese trial. |
| `build_model(trial, model)` | estimador/Pipeline | Arma el pipeline por algoritmo: el lineal con `preprocessor_linear`; los árboles con `preprocessor_trees` solo si `activate_preprocessor_trees=True` (si no, el modelo suelto). |
| `_fix_params(trial, model)` | `dict` | Traduce `params[model_name]` a llamadas `trial.suggest_*`. |

**Esquema de `params`** (uno por algoritmo, por nombre de clase):

```python
params = {
  "XGBClassifier": {
      "max_depth":     {"type": "int",         "min": 3,   "max": 10},
      "learning_rate": {"type": "float",       "min": 0.01, "max": 0.2, "log": True},
      "booster":       {"type": "categorical", "choices": ["gbtree", "dart"]},
      "objective":     {"type": "fixed",       "value": "binary:logistic"},
  },
  ...
}
```
Tipos soportados: `int`, `float` (con `log` opcional), `categorical` (con `choices`), `fixed` (con `value`).

**Limitaciones conocidas:**
- `default_params=True` está a medio hacer (`_default_params` usa un esquema con `low`/`high` en vez de `min`/`max` y devuelve el esquema, no valores). Usa `default_params=False` y pasa tu propio `params`.
- `_fix_params` llama `trial.suggest_category` — la API correcta es `trial.suggest_categorical`; ajústalo si usas tipos `categorical`.
- El pipeline no expone directamente el modelo re-entrenado con `best_params`: hay que reconstruirlo tú con el dict que devuelve `optimize()`.

---

## `metrics.py`

Métricas de discriminación, tablas de deciles y simulación de negocio. Las que dibujan devuelven la `Figure` ya cerrada (`plt.close`).

| Función | Firma (resumen) | Devuelve | Para qué |
|---|---|---|---|
| `curves_roc_pr` | `(y_true, y_score)` | `Figure` | Curvas ROC y PR, con Gini y el umbral de mejor F1. |
| `calibration_curve` | `(y_true, y_score, method_bin='qcut', show_curve=True)` | `(table, fig)` o `table` | Reliability plot por decil de score + Brier score + desvío nominal/relativo por bin. |
| `ks_table` | `(y_true, y_score, method_bin='qcut', show_curve=False)` | `table` (+`fig`) | Malos/Buenos acumulados y **KS** por decil de riesgo. |
| `bad_rate_decil` | `(y_true, y_score, method_bin='qcut')` | `table` | Bad rate por decil + chequeo de **monotonicidad**. |
| `bad_rate_acumulado` | idem | `table` | Bad rate de la cartera aprobada según hasta qué decil se aprueba. |
| `lift_acumulado` | `(y_true, y_score, method_bin='qcut', show_curve=False)` | `table` (+`fig`) | Gain / Lift acumulado + **curva CAP**. |
| `lift_bandas` | `(y_true, y_score, method_bin='qcut')` | `table` | Lift de cada banda vs bad rate global. |
| `tabla_deciles_completa` | `(y_true, y_score, method_bin='qcut')` | `table` | Tabla "todo en uno" por decil: bad rate, KS, Gain, Lift. |
| `top_decile_capture` | `(y_true, y_score, method_bin='qcut')` | `DataFrame` (1 fila) | % del total de default concentrado en el peor decil. |
| `tabla_deciles_completa_saldos` | `(y_true, y_score, series_saldo, method_bin='qcut')` | `table` | Como la anterior pero **ponderada por saldo** (captura de saldo malo, lift de saldo, montos en millones `S/`). |
| `seleccion_deciles_agrupamiento` | `(y_true, y_score, method_bin='qcut')` | `table` | Bad rate / lift / KS por intervalo de score (para agrupar deciles en bandas). |
| `tanteo_deciles` | `(y_true, y_score, bins, labels)` | `table` | Igual pero con cortes de score **manuales** (`bins`, `labels`). |
| `gestion_portafolio` | ver abajo | `table` (+ `table_2` de escenarios) | Simulación de P&L por segmento de riesgo. |

`gestion_portafolio(y_true, y_score, saldo, garantia, categoria_cliente, bins, labels, tasa_activa_por_segmento, costo_fondeo, reglas_categorias_cliente, gastos_operativos, impuesto_renta, mostrar_escenarios=False)` — segmenta por score (`bins`/`labels`), calcula provisiones (parte cubierta por garantía vs no cubierta, tasa por `categoria_cliente`), margen financiero, gastos, utilidad neta y ROA por segmento. Con `mostrar_escenarios=True` añade una tabla acumulada "aprobar hasta el segmento X".

`method_bin`: `'qcut'` (deciles por cuantiles) o `'cut'` (ancho fijo).

---

## `calibration.py`

Calibración de la PD y backtesting regulatorio.

| Función | Firma (resumen) | Devuelve | Para qué |
|---|---|---|---|
| `calibration_curve` | `(y_true, y_score, method_bin='qcut', show_curve=True)` | `(table, fig)` | Igual que la de `metrics.py` (duplicada). |
| `calibration_curve_fix_baseline` | `(y_true_expected, y_score_expected, y_true_actual, y_score_actual, period_serie, method_bin='qcut', show_curve=True)` | `(agg_actual, fig)` | Curva de calibración por período, usando los bins fijados en el **baseline** (expected). |
| `logit` | `(p, eps=1e-6)` | array | Logit con clipping. |
| `calibracion_platt_scaling` | `(score_raw_expected, y_true_expected, score_raw_actual, y_true_actual, show_graph=False, show_coef=False)` | `df_calibration` (+`fig`, +`dict_detalles`) | **Platt scaling**: ajusta una regresión logística sobre el logit del score en el baseline y la aplica al actual; compara Brier pre/post. Espera arrays de numpy (usa `.reshape`). |
| `rho_basel_retail_other` | `(pd)` | float | Correlación de activos (fórmula Basilea II, "other retail"). |
| `vasicek_quantile` | `(pd, rho, alpha)` | float | Cuantil `alpha` de la tasa de default bajo el modelo de Vasicek de un factor. |
| `vasicek_interval` | `(pd, rho, confidence=0.95)` | `(lower, upper)` | Intervalo bilateral para la tasa de default observada dada la PD. |
| `intervalos_vasicek` | `(df_calibration, confidence=0.95, figsize=(9,6))` | `(df, fig)` | Backtesting: sobre la tabla de `calibration_curve`, dibuja la banda de Vasicek por bin y marca en rojo los bins cuyo default observado cae fuera. |

---

## Flujo típico

```python
# 1. Baseline
initial_models(X_woe, X_trees, y_train)

# 2. Tuning
tuner = OptunaTunning("mi_proyecto", X_train, y_train,
                      algoritmos=[LGBMClassifier, CatBoostClassifier],
                      params=params, preprocessor_linear=None, n_trials=50)
best = tuner.optimize()

# 3. Métricas sobre test (con el modelo re-entrenado con best[...] )
pd_test = model.predict_proba(X_test)[:, 1]
metrics.ks_table(y_test, pd_test, show_curve=True)
metrics.tabla_deciles_completa_saldos(y_test, pd_test, saldo_test)

# 4. Calibración
calibration.calibracion_platt_scaling(pd_train, y_train, pd_test, y_test, show_graph=True)
tabla, _ = calibration.calibration_curve(y_test, pd_test)
calibration.intervalos_vasicek(tabla)
```

Convención del repo: comentarios, docstrings y nombres de variables en **español**.
