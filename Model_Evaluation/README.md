# Model Evaluation

Etapa de **evaluación económica del modelo** del pipeline de `handsbook-ds`: traduce el score de riesgo (PD) a impacto de negocio — captura de saldo malo, segmentación por bandas y simulación de P&L / rentabilidad de la cartera.

```
Model_Evaluation/
└── economic_metrics.py     <- 4 funciones (sin utils/, sin __init__.py)
```

> **Duplicación intencional a vigilar.** Las 4 funciones de `economic_metrics.py` son **una copia casi literal** del tramo final de `Training_Model/utils/metrics.py` (`tabla_deciles_completa_saldos`, `seleccion_deciles_agrupamiento`, `tanteo_deciles`, `gestion_portafolio`). Antes de modificar cualquiera de las dos, evalúa consolidarlas en un solo módulo en vez de mantener el duplicado.

## Cómo importar

`economic_metrics.py` se autoresuelve la raíz del repo vía `Path(__file__)` (funciona desde cualquier cwd); el notebook solo necesita la raíz en el `sys.path`:

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[2]))   # ajusta el índice según dónde esté tu notebook

from Model_Evaluation.economic_metrics import (
    tabla_deciles_completa_saldos, seleccion_deciles_agrupamiento,
    tanteo_deciles, gestion_portafolio,
)
```

**Convención de score:** `y_score` = PD (probabilidad de default). `tabla_deciles_completa_saldos` ordena por `-y_score` (decil 1 = mayor riesgo); las otras tres segmentan por el valor del score directamente. `method_bin`: `'qcut'` (cuantiles) o `'cut'` (ancho fijo).

## Funciones

| Función | Firma | Devuelve | Qué hace |
|---|---|---|---|
| `tabla_deciles_completa_saldos` | `(y_true, y_score, series_saldo: pd.Series, method_bin='qcut')` | `DataFrame` | Tabla de deciles **ponderada por saldo**: clientes, saldo total / malo / bueno (en millones `S/…M`), bad rate por nº y por saldo, captura de saldo malo acumulada, lift de saldo, KS, Gain, Lift. |
| `seleccion_deciles_agrupamiento` | `(y_true, y_score, method_bin='qcut')` | `DataFrame` | Bad rate %, lift por banda y KS % por intervalo de score. Sirve para decidir cómo **agrupar deciles en bandas de riesgo**. |
| `tanteo_deciles` | `(y_true, y_score, bins, labels)` | `DataFrame` | Igual que la anterior pero con **cortes manuales** de score (`bins` = lista de bordes, `labels` = nombres de banda). Para tantear puntos de corte concretos. |
| `gestion_portafolio` | `(y_true, y_score, saldo, garantia, categoria_cliente, bins, labels, tasa_activa_por_segmento: dict, costo_fondeo, reglas_categorias_cliente: dict, gastos_operativos, impuesto_renta, mostrar_escenarios=False)` | `DataFrame` (+ `DataFrame` de escenarios) | Simulación de **P&L por segmento de riesgo**: provisiones (parte cubierta por garantía a media tasa vs parte no cubierta a tasa según `categoria_cliente`), margen financiero (tasa activa − `costo_fondeo`), gastos operativos prorrateados por participación, utilidad neta después de `impuesto_renta` y ROA. Con `mostrar_escenarios=True` añade una segunda tabla acumulada tipo "aprobar hasta el segmento X". |

### Parámetros de `gestion_portafolio`

- `saldo`, `garantia`, `categoria_cliente`: series alineadas con `y_true` (exposición, valor de garantía y segmento comercial de cada cliente).
- `bins` / `labels`: cortes de score que definen los segmentos de riesgo.
- `tasa_activa_por_segmento`: `dict` `{label_segmento: tasa}` — tasa de colocación por banda.
- `reglas_categorias_cliente`: `dict` `{categoria_cliente: tasa_provision_no_cubierta}`.
- `costo_fondeo`, `gastos_operativos`, `impuesto_renta`: escalares.

Montos en la salida vienen formateados como texto `S/{x}M` (millones de soles).

## Uso típico

```python
pd_test = model.predict_proba(X_test)[:, 1]

tabla_deciles_completa_saldos(y_test, pd_test, saldo_test)

bins   = [0, 0.03, 0.08, 0.20, 1.0]
labels = ["Bajo", "Medio", "Alto", "Muy Alto"]
tanteo_deciles(y_test, pd_test, bins, labels)

tabla, escenarios = gestion_portafolio(
    y_test, pd_test, saldo_test, garantia_test, categoria_test,
    bins=bins, labels=labels,
    tasa_activa_por_segmento={"Bajo": 0.18, "Medio": 0.24, "Alto": 0.32, "Muy Alto": 0.40},
    costo_fondeo=0.06,
    reglas_categorias_cliente={"A": 0.05, "B": 0.10, "C": 0.20},
    gastos_operativos=5_000_000, impuesto_renta=0.295,
    mostrar_escenarios=True,
)
```

Convención del repo: comentarios, docstrings y nombres de variables en **español**.
