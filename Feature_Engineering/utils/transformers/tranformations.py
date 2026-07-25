import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
import sys
from pathlib import Path
from matplotlib.pyplot import pyplot as plt

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class TransformationMethods(BaseEstimator, TransformerMixin):
    # Incluirá: Yeo-Johnson, Box-Cox, Log1p
    def __init__(self, strategy, features_num, return_dataframe= True):
        self.strategy = strategy
        self.features_num = features_num
        self.return_dataframe = return_dataframe

    def log_transform(self, X):
        return np.log1p(X)
    
    def inverse_log_transform(self, X):
        return np.expm1(X)
    
    def fit(self, X, y=None):
        X = X.copy()
        if self.strategy == 'yeo-johnson':
            self.transformer = PowerTransformer(method= 'yeo-johnson', standardize = False)
        elif self.strategy == 'box-cox':
            self.transformer = PowerTransformer(method = 'box-cox', standardize = False)
        elif self.strategy == 'log1p':
            self.transformer = FunctionTransformer(
                func = self.log_transform,
                inverse_func = self.inverse_log_transform
            )

        if self.features_num:
            self.transformer.fit(X[self.features_num])

        return self

    def transform(self, X, y=None):
        X = X.copy()
        self.X_num = pd.DataFrame(index = X.index)

        if self.features_num:
            self.X_num = pd.DataFrame(
                self.transformer.transform(X[self.features_num]),
                columns = self.features_num,
                index = X.index
            )
        
        if self.return_dataframe:
            return self.X_num
        
        return self.X_num.values

    def summary(self, X):
        X = X.copy()

        n_rows = len(self.features_num)
        fig, ax = plt.subplots(nrows=n_rows, ncols=2, figsize = (12, n_rows*4), squeeze= False)
        
        for i, feat in enumerate(self.features_num):
            ax[i, 0].hist(
                X[feat].dropna(),
                bins = 30,
                color = 'skyblue',
                label = 'Antes de la transformación'
            )

            ax[i, 0].set_title(f'Previamente: {feat}')
            ax[i, 0].set_ylabel('Frecuencia')
            ax[i, 0].set_xlabel(f'{feat}')
            ax[i, 0].legend()


            ax[i, 1].hist(
                self.X_num[feat].dropna(),
                bins = 30,
                color = 'lightcoral',
                label = 'Después de la transformación'
            )

            ax[i, 1].set_title(f'Posteriormente: {feat}')
            ax[i, 1].legend()
            ax[i, 1].set_ylabel('Frecuencia')
            ax[i, 1].set_xlabel(f'{feat}')

        plt.tight_layout()
        plt.close(fig)

        return fig

    









