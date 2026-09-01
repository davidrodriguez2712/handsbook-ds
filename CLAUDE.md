# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Responde siempre en español. Los comentarios, docstrings y nombres de variables del código de este repo están en español; respeta esa convención al editar.

## Qué es esto

`handsbook-ds` es un monorepo que automatiza el pipeline de Machine Learning para problemas de riesgo/scoring (clasificación binaria bancaria). Tiene dos capas:

1. **Paquetes de utilidades reutilizables por etapa del pipeline**, en la raíz: `EDA/`, `Feature_Engineering/`, `Training_Model/`, `Model_Evaluation/`, `Monitoring/`, más `CORE/` (helpers compartidos entre etapas).
2. **Proyectos concretos** bajo `11_Project_Templates/<proyecto>/`, que son scaffolds notebook-first (`notebooks/`, `data/`, `artifacts/`, `backend/`, `frontend/`) e importan la lógica real desde los paquetes de la raíz. Cada proyecto tiene su propio `CLAUDE.md` con contexto específico (ver `11_Project_Templates/2-give-me-some-credit/CLAUDE.md`).

Estado (README.md): EDA terminado, Feature Engineering en progreso, Training/Evaluation/Monitoring en construcción — los paquetes existen y son funcionales aunque los notebooks de esas etapas estén incompletos.

**No hay build, lint, tests, ni CI.** No hay `requirements.txt`/`environment.yml` en la raíz. El entorno es conda (ver `.vscode/settings.json`). El flujo de trabajo es ejecutar notebooks en Jupyter.

## Resolución de imports (crítico)

Los notebooks y varios módulos compartidos agregan la raíz del monorepo a `sys.path` en runtime así:

```python
PARENT_DIRECTORY_1 = Path.cwd().parent.parent.parent
sys.path.append(f"{str(PARENT_DIRECTORY_1)}")
from EDA.utils.classifications import EDAClasificacion
from Feature_Engineering.utils.transformers.woe import WOEClassic
```

Esto **asume que el cwd está 3 niveles bajo la raíz** — es decir, `11_Project_Templates/<proyecto>/notebooks/`. Módulos como `Feature_Engineering/utils/selection/feature_selector.py` y `Training_Model/utils/algorithm.py` repiten internamente el mismo `Path.cwd().parent.parent.parent` para llegar a `CORE.metadata`. Si ejecutas ese código con otro cwd (script suelto, `python -c`, tests, notebook en `EDA/examples/`), los imports fallan. Al ejecutar código de estos paquetes fuera de un notebook de proyecto, ajusta el cwd o el `sys.path` manualmente.

## Paquetes de la raíz

- **`CORE/metadata.py`** — `inventario_features(data)`: DataFrame de auditoría por feature (dtype, tipo semántico Conteo/Continua/String/Datetime, % missing, cardinalidad, constantes). Usado por todas las etapas.
- **`EDA/utils/classifications.py`** — `EDAClasificacion(data, target_name, tipo_problema)`: clase única que cubre toda la etapa EDA (overview, missings, outliers IQR/percentil, distribuciones, análisis de target, information value, PSI train-vs-test) con un `run_all()` de conveniencia. Los archivos `overview.py`, `missing.py`, `outliers.py`, `distribution.py`, `correlation.py` son una extracción en curso de esa clase-monolito hacia funciones sueltas — al tocar EDA, revisa si el cambio va en la clase o en el módulo extraído para no divergir.
- **`Feature_Engineering/utils/transformers/`** — transformers sklearn (`BaseEstimator, TransformerMixin`, fit/transform): `WOEClassic` y `Logit_Smoothing_Rolling` (woe.py), `Binning`, `AutoOutlierHandler`, `AutoMissingHandler`/`AutoInteractiveImputer`, `ScalingMethods`, `TransformationMethods`. Se componen dentro de un `ColumnTransformer`/`Pipeline`.
- **`Feature_Engineering/utils/selection/`** — funciones de selección: `correlation`, `vif`, `information_value` (iv.py), `gini_rule_selection` (gini.py), `dataframe_psi` (psi.py); más `FeatureSelector` (transformer que subsetea a una lista fija de features, típicamente el último paso del pipeline).
- **`Training_Model/utils/`** — `initial_models(...)` (algorithm.py) hace baseline de LogisticRegression/DecisionTree/RandomForest/XGBoost/LightGBM/CatBoost; `OptunaTunning` (optuna_tunning.py) búsqueda de hiperparámetros por algoritmo; `metrics.py` (KS, lift, tablas de deciles, gestión de portafolio); `calibration.py` (curva de calibración, Platt scaling, intervalos de Vasicek/Basel).
- **`Model_Evaluation/economic_metrics.py`** — tablas de deciles ponderadas por saldo y simulación de gestión de portafolio/pricing. **Es casi una copia literal del tramo final de `Training_Model/utils/metrics.py`** — antes de modificar cualquiera de las dos, evalúa consolidarlas.
- **`Monitoring/utils/monitor.py`** — compara un baseline ("expected") contra periodos actuales ("actual"): performance (AUC/Gini), missings, PSI, outliers, IV, bad rate vs PD, ROC/PR, variación de contribución SHAP.

Nota: los archivos `*/utils/binary_classification.py` y varios `reports/*.py` son stubs vacíos. Los directorios de la raíz `AI_PROMPTS/`, `BUSINESS/`, `CHEATSHEETS/`, `CHECKLIST/`, `FRAMEWORKS/`, `SQL/` y `11_Project_Templates/ejemplo-2/` están vacíos (placeholders).

## Estructura de un proyecto (`11_Project_Templates/<proyecto>/`)

`notebooks/` numerados por etapa (`01_eda` … `05_monitoring`) · `data/{raw,external,interim,processed}` (raw = solo lectura) · `artifacts/{eda,feature_engineering,training_model,model_evaluation,monitoring}` (salidas: `.pkl`, figuras, tablas) · `backend/` FastAPI que sirve predicciones · `frontend/` Streamlit.

**El `backend/` vendoriza copias físicas** de `CORE/`, `Training_Model/` y partes de `Feature_Engineering/utils/` dentro de sí mismo (el build context de Docker es `backend/`). No hay symlink ni submódulo — si cambias un paquete compartido de la raíz del que dependa un backend, propaga el cambio manualmente.
