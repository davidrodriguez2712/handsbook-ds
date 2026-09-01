import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve, brier_score_loss
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import joblib

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

def calibration_curve(y_true, y_score, method_bin = 'qcut', show_curve= True):
    # crear los deciles
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    brier_score = brier_score_loss(
        y_true= y_true,
        y_proba = y_score
    )
    if method_bin == 'qcut':
        df_mix['bins'] = pd.qcut(
            x= df_mix['y_score'],
            q= 10
        )
    elif method_bin == 'cut':
        df_mix['bins'] = pd.cut(
            x= df_mix['y_score'],
            bins= 10
        )
    table = df_mix.groupby('bins', observed= True).agg(
        materialidad = ('y_score', 'count'),
        y_real_mean = ('y_true', 'mean'),
        y_score_mean = ('y_score', 'mean')
    ).reset_index()
    table['dif_nom'] = table['y_real_mean'] - table['y_score_mean']
    table['dif_var_%'] = round(((table['y_real_mean'] - table['y_score_mean'])/ table['y_score_mean'])*100, 2)

    # gráfico de curva
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.plot(
        table['y_score_mean'],
        table['y_real_mean'],
        marker = 'o'
    )

    ax.plot(
        [0, max( table['y_real_mean'].max(), table['y_score_mean'].max() ) + 0.05 ],
        [0, max( table['y_real_mean'].max(), table['y_score_mean'].max() ) + 0.05 ],
        '--',
        color= 'tomato'
    )
    ax.set_title(f'Curva de Calibración - Brier Score: {brier_score:.4f}')
    ax.set_ylabel('Promedio Default Rate')
    ax.set_xlabel('Promedio del Score Predicho')
    plt.close(fig)
    if show_curve:
        return table, fig
    return table

def calibration_curve_fix_baseline(y_true_expected, y_score_expected, y_true_actual, y_score_actual, period_serie,  method_bin = 'qcut', show_curve= True):
    # crear los deciles
    df_mix_expected = pd.DataFrame({
        'y_true': y_true_expected,
        'y_score': y_score_expected
    })
    if method_bin == 'qcut':
        df_mix_expected['bins'], bins_expected = pd.qcut(
            x= df_mix_expected['y_score'],
            q= 10,
            retbins= True
        )
    elif method_bin == 'cut':
        df_mix_expected['bins'], bins_expected = pd.cut(
            x= df_mix_expected['y_score'],
            bins= 10,
            retbins= True
        )
    df_mix_actual = pd.DataFrame({
        'y_true': y_true_actual,
        'y_score': y_score_actual,
        'period': period_serie
    })
    df_mix_actual['bins'] = pd.cut(
        df_mix_actual['y_score'],
        bins= bins_expected
    )
    agg_actual = df_mix_actual.groupby(['period','bins'], observed= False).agg(
        materialidad = ('y_score', 'count'),
        y_real_mean = ('y_true', 'mean'),
        y_score_mean = ('y_score', 'mean')
    ).reset_index()

    # gráfico de curva
    period_actual = sorted(agg_actual['period'].unique())
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    for period in period_actual:
        eje_x = agg_actual.loc[agg_actual['period'] == period, 'y_score_mean']
        eje_y = agg_actual.loc[agg_actual['period'] == period, 'y_real_mean']
        ax.plot(
            eje_x,
            eje_y,
            #marker = 'o',
            label = period
        )
    max_score = agg_actual['y_score_mean'].max()
    max_real = agg_actual['y_real_mean'].max()
    max_ambos = max(max_real, max_score)
    ax.plot(
        [0, max_ambos],
        [0, max_ambos],
        '--',
        color = 'red'
    )
    ax.set_title('Calibration Curve por decil en Test/OOT')
    ax.set_ylabel('Promedio Default Rate')
    ax.set_xlabel('Promedio PD')
    ax.legend()
    plt.close(fig)
    if show_curve:
        return agg_actual, fig
    return agg_actual


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def calibracion_platt_scaling(score_raw_expected, y_true_expected, score_raw_actual, y_true_actual, show_graph= False, show_coef = False):
    score_logit_expected = logit(p = score_raw_expected)
    score_logit_actual = logit(p = score_raw_actual)
    brier_score_actual_pre_calibrado = brier_score_loss(
        y_true= y_true_actual,
        y_proba = score_raw_actual
    )
    platt_model = LogisticRegression()
    platt_model.fit(score_logit_expected.reshape(-1, 1), y_true_expected)

    dict_detalles = {
        'coeficientes': platt_model.coef_,
        'intercepto': platt_model.intercept_
    }

    pd_calibrada_actual = platt_model.predict_proba(score_logit_actual.reshape(-1, 1))[:, 1]

    brier_score_calibrado_actual = brier_score_loss(
        y_true= y_true_actual,
        y_proba= pd_calibrada_actual
    )
    dict_detalles['brier_score'] = {
        'brier_score_pre_calibrado': brier_score_actual_pre_calibrado,
        'brier_score_calibrado': brier_score_calibrado_actual
    }

    df_calibration, fig = calibration_curve(y_true= y_true_actual, y_score= pd_calibrada_actual, method_bin= 'qcut', show_curve= True)
    if show_graph:
        if show_coef:
            return df_calibration, fig, dict_detalles
        else:
            return df_calibration, fig
    if show_coef:
        return df_calibration, dict_detalles
    return df_calibration


def rho_basel_retail_other(pd):
    """Correlación de activos, fórmula Basel II para 'other retail' exposures"""
    term = (1 - np.exp(-35 * pd)) / (1 - np.exp(-35))
    return 0.03 * term + 0.16 * (1 - term)

def vasicek_quantile(pd, rho, alpha):
    """
    pd:    PD esperada (ej. 0.05)
    rho:   correlación de activos (0 a 1)
    alpha: nivel de confianza (ej. 0.999 para WCDR Basel, 0.975 para límite sup. de un IC 95%)
    """
    return norm.cdf(
        (norm.ppf(pd) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho)
    )

def vasicek_interval(pd, rho, confidence=0.95):
    """
    Intervalo de confianza bilateral para la tasa de default observada, dado PD y rho.
    Devuelve (limite_inferior, limite_superior)
    """
    alpha_lower = (1 - confidence) / 2
    alpha_upper = 1 - alpha_lower
    lower = vasicek_quantile(pd, rho, alpha_lower)
    upper = vasicek_quantile(pd, rho, alpha_upper)
    return lower, upper

def intervalos_vasicek(df_calibration, confidence=0.95, figsize=(9, 6)):
    """
    df_calibration: tu tabla con y_score_mean (PD esperada) y y_real_mean (default observado) por bin
    """
    df = df_calibration.copy().sort_values('y_score_mean').reset_index(drop=True)
    
    # calcular rho y bandas por bin
    df['rho'] = df['y_score_mean'].apply(rho_basel_retail_other)
    bounds = df.apply(
        lambda row: vasicek_interval(row['y_score_mean'], row['rho'], confidence),
        axis=1
    )
    df['ic_inferior'] = bounds.apply(lambda x: x[0])
    df['ic_superior'] = bounds.apply(lambda x: x[1])
    df['fuera_de_rango'] = (df['y_real_mean'] < df['ic_inferior']) | (df['y_real_mean'] > df['ic_superior'])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))  # eje x = índice de bin (más limpio que valores continuos)

    # banda de confianza Vasicek
    ax.fill_between(
        x, df['ic_inferior'], df['ic_superior'],
        color='steelblue', alpha=0.2,
        label=f'Intervalo Vasicek ({int(confidence*100)}%)'
    )
    ax.plot(x, df['ic_inferior'], color='steelblue', linestyle='--', linewidth=1)
    ax.plot(x, df['ic_superior'], color='steelblue', linestyle='--', linewidth=1)

    # PD esperada (línea central del modelo)
    ax.plot(x, df['y_score_mean'], color='steelblue', linewidth=1.5,
            label='PD esperada (score)')

    # default rate observado, resaltando los que caen fuera de rango
    colors = np.where(df['fuera_de_rango'], 'red', 'black')
    ax.scatter(x, df['y_real_mean'], color=colors, zorder=5, s=50,
               label='Default rate observado')
    ax.plot(x, df['y_real_mean'], color='gray', linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df['bins'].astype(str), rotation=45, ha='right')
    ax.set_xlabel('Bin (decil de score)')
    ax.set_ylabel('Tasa de default')
    ax.set_title('Backtesting de calibración con intervalos de Vasicek')
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    return df, fig