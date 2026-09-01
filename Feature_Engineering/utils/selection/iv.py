import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

def information_value(X: pd.DataFrame, y: pd.Series, feat_bins: dict, feat_numeric: list, feat_cat: list):
    X = X.copy()
    y = y.copy()
    print(type(y))
    print(y)
    target_name = y.name
    X_y = pd.concat([X, y], axis= 1)
    eps = 1e-5
    
    feats_iv = []

    for feat in feat_numeric:
        X_y['bins'] = pd.cut(
            x = X_y[feat],
            bins= feat_bins[feat]['bins'],
            duplicates= 'drop'
        )

        X_y['bins'] = X_y['bins'].astype('object')
        X_y.loc[X_y[feat].isna(), 'bins'] = "MISSING"

        table = X_y.groupby(
            ['bins'], observed=False
        ).agg(
            total_obs = (target_name, "count"),
            total_event = (target_name, "sum"),
            event_rate = (target_name, "mean")
        )
        table['total_no_event'] = table["total_obs"] - table["total_event"]
        table['no_event_rate'] = table['total_no_event'] / table['total_obs']
        table['dist_event'] = table['total_event'] / table['total_event'].sum()
        table['dist_no_event'] = table['total_no_event'] / table['total_no_event'].sum()
        table['woe'] = (
            np.log(
                (table['dist_no_event'] + eps) / (table['dist_event'] + eps)
            )
        )
        table = table.reset_index(level= 0, drop= False)
        table['information_value_calculation'] = ((table['dist_no_event'] - table['dist_event']) * (table['woe']))
        iv = (table['information_value_calculation'].sum())

        feats_iv.append({
            'feature': feat,
            'iv': round(iv, 4)
        })

    for feat in feat_cat:

        labels = feat_bins[feat]['bins']
        X_y.loc[X_y[feat].isna(), feat] = "MISSING"
        X_y.loc[~X_y[feat].isin(labels), feat] = "UNKNOWN"

        table = X_y.groupby(feat, observed= False).agg(
            total_obs = (target_name, "count"),
            total_event = (target_name, "sum"),
            event_rate = (target_name, "mean")
        )
        table['total_no_event'] = table["total_obs"] - table["total_event"]
        table['no_event_rate'] = table['total_no_event'] / table['total_obs']
        table['dist_event'] = table['total_event'] / table['total_event'].sum()
        table['dist_no_event'] = table['total_no_event'] / table['total_no_event'].sum()
        table['woe'] = (
            np.log(
                (table['dist_no_event'] + eps) / (table['dist_event'] + eps)
            )
        )
        table = table.reset_index(level= 0, drop= False)
        table['information_value_calculation'] = ((table['dist_no_event'] - table['dist_event']) * (table['woe']))
        iv = (table['information_value_calculation'].sum())

        feats_iv.append({
            'feature': feat,
            'iv': round(iv, 4)
        })
        
    df_iv = pd.DataFrame(feats_iv).sort_values(by= 'iv', ascending= False)
    return df_iv

