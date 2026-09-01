import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

def vif(X: pd.DataFrame):

    X = X.copy()

    X_numeric = X.select_dtypes(include='number').dropna()

    X_const = add_constant(X_numeric)

    vif_df = pd.DataFrame({
        'feature': X_numeric.columns,
        'VIF': [
            variance_inflation_factor(
                X_const.values,
                i + 1
            )
            for i in range(X_numeric.shape[1])
        ]
    })

    return vif_df.sort_values(
        by='VIF',
        ascending=False
    )








