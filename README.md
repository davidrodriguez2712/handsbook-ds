# handsbook-ds

Toolkit para **automatizar el pipeline de Machine Learning** en problemas de **clasificación binaria de riesgo / scoring** (probabilidad de default, admisión de crédito, churn).

Cada etapa del pipeline es una carpeta con un paquete `utils/` de código reutilizable (funciones y transformers compatibles con scikit-learn) y su propio `README.md` con el detalle de cada script.

## Etapas

| Etapa | Carpeta | Estado | Qué contiene |
|---|---|---|---|
| 1. Análisis exploratorio | [`EDA/`](EDA/README.md) | Terminado | `EDAClasificacion`: overview, missings, outliers, distribuciones, análisis del target, Information Value, PSI. |
| 2. Ingeniería de variables | [`Feature_Engineering/`](Feature_Engineering/README.md) | Terminado | Transformers sklearn (imputación, outliers, escalado, transformaciones, WOE / log-odds) + selección por correlación / VIF / IV / Gini / PSI. |
| 3. Entrenamiento | [`Training_Model/`](Training_Model/README.md) | Terminado | Baseline de algoritmos, tuning con Optuna, métricas de discriminación / deciles / KS / lift / negocio, calibración (Platt, Vasicek). |
| 4. Evaluación económica | [`Model_Evaluation/`](Model_Evaluation/README.md) | Terminado | Tablas de deciles ponderadas por saldo y simulación de P&L / ROA por segmento de riesgo. |
| 5. Monitoreo | [`Monitoring/`](Monitoring/README.md) | Terminado | Comparación baseline vs producción por período: performance, PSI, missings, outliers, IV, bad rate vs PD, drift de SHAP. |

Las cinco etapas están operativas de punta a punta. El proyecto sigue en evolución: se irán añadiendo nuevos transformers, métricas y utilidades sobre esta misma estructura.

`CORE/metadata.py` — helper compartido entre todas las etapas: `inventario_features(data)` devuelve una ficha por columna (dtype, tipo semántico `Conteo`/`Continua`/`String`/`Datetime`, % missing, cardinalidad, constantes).

## Convención de imports

Los módulos de cada `utils/` que necesitan llegar a `CORE/` u otra etapa **se autoresuelven**: anclan la raíz del repo con `Path(__file__).resolve().parents[N]` y la agregan a `sys.path`. Funcionan desde cualquier directorio de trabajo.

Lo único que sigue necesitando ayuda es el **notebook** (no es un archivo de paquete, no tiene `__file__`): hay que ponerle la raíz del repo en el path una vez, al inicio.

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[2]))   # ajusta el índice a dónde esté tu notebook

from EDA.utils.classifications import EDAClasificacion
from Feature_Engineering.utils.selection.gini import gini_rule_selection
from Training_Model.utils.optuna_tunning import OptunaTunning
```

(Al empaquetar el repo como librería, este último paso también desaparece.)

## Entorno

- Python + conda (ver `.vscode/settings.json`).
- Dependencias principales: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `missingno`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `joblib`.
- No hay build, lint ni tests configurados; el flujo de trabajo es notebook-first.

## Convenciones de código

- Comentarios, docstrings y nombres de variables en **español**.
- Los transformers custom siguen el patrón `BaseEstimator, TransformerMixin` (fit/transform) para componerse dentro de un `Pipeline` / `ColumnTransformer`.
- Convención de score en las etapas 3-5: `y_score` es la **PD** (probabilidad de la clase positiva = "malo"). Revisa el sentido del decil en cada función (unas ordenan por mayor riesgo, otras por menor).
