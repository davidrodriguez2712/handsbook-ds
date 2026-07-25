import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features



class AutoOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method, strategy=None, return_dataframe = True):
        self.method = method # iqr, percentile
        self.strategy = strategy # windsorization
        self.return_dataframe = return_dataframe

    def fit(self, X: pd.DataFrame, y = None):
        X = X.copy()
        # Entrenar por IQR
        df_features = inventario_features(data= X)
        self.numeric_features = df_features.loc[df_features['semantic_dtype'] == 'Continua'].columns.tolist()
        resultados = []
        if self.method == 'iqr':
            for feat in self.numeric_features:
                iqr = X[feat].quantile(0.75) - X[feat].quantile(0.25)
                upper_limit = X[feat].quantile(0.75) + 1.5*(iqr)
                lower_limit = X[feat].quantile(0.25) - 1.5*(iqr)
                iqr_dict = {
                    'feature': feat,
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit
                }
                resultados.append(iqr_dict)
        elif self.method == 'percentile':
            for feat in self.numeric_features:
                if X.shape[0] < 5000:
                    upper_limit = X[feat].quantile(0.95)
                    lower_limit = X[feat].quantile(0.05)
                if X.shape[0] < 50000:
                    upper_limit = X[feat].quantile(0.975)
                    lower_limit = X[feat].quantile(0.025)
                if X.shape[0] > 50000:
                    upper_limit = X[feat].quantile(0.99)
                    lower_limit = X[feat].quantile(0.01)
                percentile_dict = {
                    'feature': feat,
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit
                }
                resultados.append(percentile_dict)
            
        self.resultados_list = resultados

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        if self.strategy == 'windsorization':
            for item in self.resultados_list:
                X[item['feature']] = np.clip(X[item['feature']], min= item['lower_limit'], max= item['upper_limit'])

        if self.return_dataframe:
            return X

        return X.values
        

    def summary(self):
        # aca colocar lo que daría en el EDA un dataframe básicamente
        pass



























