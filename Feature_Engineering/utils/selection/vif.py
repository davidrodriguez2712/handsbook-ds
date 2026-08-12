import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
from matplotlib.pyplot import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features


def vif(X: pd.DataFrame):
    X = X.copy()
    df_feats_numerics = X.select_dtypes(include= 'number')
    df_feats_numerics = df_feats_numerics.dropna()

    vif_df = pd.DataFrame({
        'feature': df_feats_numerics.columns,
        'VIF': [
            variance_inflation_factor(df_feats_numerics.values, i)
            for i in range(df_feats_numerics.shape[1])
        ]
    }).sort_values(by= 'VIF', ascending = False)

    return vif_df








