import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class AutoOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method, strategy=None, return_dataframe = True, exclude_continuous = False):
        self.method = method # iqr, percentile
        self.strategy = strategy # windsorization
        self.return_dataframe = return_dataframe
        self.exclude_continuous = exclude_continuous

    def fit(self, X: pd.DataFrame, y = None):
        X = X.copy()
        # Entrenar por IQR
        df_features = inventario_features(data= X)
       
        if self.exclude_continuous:
            self.numeric_features_ = df_features.loc[df_features['semantic_dtype'] == 'Continua', 'feature'].values.tolist()
        else:
            self.numeric_features_ = df_features.loc[(df_features['semantic_dtype'] == 'Continua') | (df_features['semantic_dtype'] == 'Conteo'), 'feature'].values.tolist()
        
        self.resultados_ = {}
        if self.method == 'iqr':
            for feat in self.numeric_features_:
                iqr = X[feat].quantile(0.75) - X[feat].quantile(0.25)
                upper_limit = X[feat].quantile(0.75) + 1.5*(iqr)
                lower_limit = X[feat].quantile(0.25) - 1.5*(iqr)
                iqr_dict = {
                    'feature': feat,
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit
                }
                self.resultados_[feat] = iqr_dict
        elif self.method == 'percentile':
            for feat in self.numeric_features_:
                
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
                self.resultados_[feat] = percentile_dict

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        #resultados_dict = {self.resultados_list['feature']: d for d in self.resultados_list}
        X_transformed = {}
        if self.strategy == 'windsorization':
            for feat in self.numeric_features_:
                #print(self.resultados_)
                min_value = self.resultados_[feat]['lower_limit']
                max_value = self.resultados_[feat]['upper_limit']
                X_transformed[feat] = np.clip(X[feat], a_min= min_value, a_max= max_value)

        X_transformed_df = pd.DataFrame(X_transformed, index= X.index)

        if self.return_dataframe:
            return X_transformed_df

        return X_transformed_df.values
        

    def summary(self):
        # aca colocar lo que daría en el EDA un dataframe básicamente
        pass



























