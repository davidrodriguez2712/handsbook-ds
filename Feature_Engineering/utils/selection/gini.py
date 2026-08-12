# Sumar aqui una función que traiga los valores de gini para cada feature
# considerando un modelo lineal o no lineal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

def gini_univariate_features(X: pd.DataFrame, y_true:pd.Series):
    """Calcula el gini univariado de una feature vs el target de referencia"""
    X = X.copy()
    features_num = X.select_dtypes(include= 'number').columns.tolist()
    features_cat = X.select_dtypes(include= ['string', 'category', 'object']).columns.tolist()
    EPS = 1e-6
    features_list = []

    for feat_num in features_num:
        mask = X[feat_num].notna()
        y_score = X.loc[mask, feat_num]
        roc_auc = roc_auc_score(y_true= y_true[mask], y_score= y_score)
        gini = (2 * roc_auc) - 1
        features_list.append({
            'feature': feat_num,
            'roc_auc': round(roc_auc, 3),
            'gini': round(gini, 3)
        })

    if features_cat:
        for feat_cat in features_cat:
            p_target_mean = y_true.groupby(X[feat_cat]).mean()
            p_target_mean = p_target_mean.clip(lower= EPS, upper= 1 - EPS)
            output = X[feat_cat].map(p_target_mean)
            log_odds = np.log(
                output / (1 - output)
            )
            mask = output.notna()
            roc_auc = roc_auc_score(y_true= y_true[mask], y_score= output[mask])
            gini = (2 * roc_auc) - 1

            features_list.append({
                'feature': feat_cat,
                'roc_auc': round(roc_auc, 3),
                'gini': round(gini, 3)
            })

    df_final = pd.DataFrame(features_list)

    return df_final


def gini_missing_rule(gini, missing):
    gini_bajo = 0.03
    gini_medio = 0.08
    missing_bajo = 0.1
    missing_medio = 0.3
    if gini < gini_bajo:
        if missing < missing_bajo:
            return 'Revisar'
        elif missing >= missing_bajo and missing <= missing_medio:
            return 'Eliminar'
        elif missing > missing_medio:
            return 'Eliminar'
    elif gini >= gini_bajo and gini < gini_medio:
        if missing < missing_bajo:
            return 'Mantener'
        elif missing >= missing_bajo and missing <= missing_medio:
            return 'Revisar'
        elif missing > missing_medio:
            return 'Eliminar'
    elif gini > gini_medio:
        if missing < missing_bajo:
            return 'Mantener'
        elif missing >= missing_bajo and missing <= missing_medio:
            return 'Mantener'
        elif missing > missing_medio:
            return 'Revisar'  

def gini_rule_selection(X:pd.DataFrame, y_true: pd.Series):
    """Devuelve un dataframe con la respuesta de negocio configuradas para considerar eliminar o mantener una variable segun su Gini y % missing"""
    X = X.copy()
    y = y_true.copy()
    df_inventario = inventario_features(X)
    df_auc_gini = gini_univariate_features(X, y)
    df_merge = df_auc_gini.merge(
        right= df_inventario[['feature', 'missing_pct']],
        how='inner',
        on= 'feature'
    ).sort_values(by= ['roc_auc', 'missing_pct'], ascending= [False, False])

    df_merge['regla'] = (
        df_merge.apply(
            lambda x: gini_missing_rule(x['gini'], x['missing_pct']),
            axis= 1
        )
    )

    return df_merge







