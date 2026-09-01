import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features


def calculate_psi(expected, actual):
        """
        Calcula el PSI a partir de dos distribuciones.
        expected y actual deben ser arrays de proporciones.
        """
        EPSILON = 1e-6
        expected = np.clip(expected, EPSILON, None)
        actual = np.clip(actual, EPSILON, None)

        return np.sum((actual - expected) * np.log(actual / expected))

def interpret_psi(psi):

    if psi < 0.10:
        return "Sin drift"

    elif psi < 0.25:
        return "Drift moderado"

    else:
        return "Drift severo"

def psi_numeric(train, test, bins=10):
    """
    PSI para variables numéricas usando cuantiles del Train.
    """

    # bins usando el Train
    cut_points = np.unique(
        np.quantile(train.dropna(), np.linspace(0, 1, bins + 1))
    )

    train_bins = pd.cut(
        train,
        bins=cut_points,
        include_lowest=True,
        duplicates="drop"
    )

    test_bins = pd.cut(
        test,
        bins=cut_points,
        include_lowest=True,
        duplicates="drop"
    )

    train_dist = (
        train_bins.value_counts(normalize=True)
        .sort_index()
    )

    test_dist = (
        test_bins.value_counts(normalize=True)
        .reindex(train_dist.index, fill_value=0)
    )

    psi = calculate_psi(
        train_dist.values,
        test_dist.values
    )

    return psi


def psi_categorical(train, test):
    """
    PSI para variables categóricas.
    """

    train_dist = (
        train.fillna("MISSING")
        .value_counts(normalize=True)
    )

    test_dist = (
        test.fillna("MISSING")
        .value_counts(normalize=True)
    )

    categories = train_dist.index.union(test_dist.index)

    train_dist = train_dist.reindex(categories, fill_value=0)
    test_dist = test_dist.reindex(categories, fill_value=0)

    psi = calculate_psi(
        train_dist.values,
        test_dist.values
    )

    return psi


def dataframe_psi(train_df, test_df, period_name = None, return_dict = False):
    """
    Calcula PSI para todas las columnas.
    """
    train_df = train_df.copy()
    #X_train, X_test, y_train, y_test = train_test_split()
    results = []
    for col in train_df.columns:

        if pd.api.types.is_numeric_dtype(train_df[col]):
            psi = psi_numeric(
                train_df[col],
                test_df[col]
            ).round(4)
            var_type = "numeric"
            interpretacion = interpret_psi(psi= psi)

        else:
            psi = psi_categorical(
                train_df[col],
                test_df[col]
            ).round(4)
            var_type = "categorical"
            interpretacion = interpret_psi(psi= psi)

        if period_name is None:
            results.append({
                "feature": col,
                "type": var_type,
                "psi": psi,
                "interpretacion": interpretacion
            })
        else:
            results.append({
                "feature": col,
                "type": var_type,
                "psi": psi,
                "interpretacion": interpretacion,
                "period": period_name
            })
    print(results)
    if return_dict:
        return results
    results = (
        pd.DataFrame(results)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )

    return results