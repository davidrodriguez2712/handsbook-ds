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


class ScalingMethods(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, return_dataframe, feats_num):
        self.strategy = strategy
        self.return_dataframe = return_dataframe
        self.feats_num = feats_num

    def fit(self, X, y=None):
        X = X.copy()

        if self.strategy == 'standarscaler':
            self.transformer = StandardScaler()
        
        elif self.strategy == 'robustscaler':
            self.transformer = RobustScaler()

        elif self.strategy == 'minmaxscaler':
            self.transformer = MinMaxScaler()

        return self

    def transform(self, X):
        X = X.copy()
        X_num = pd.DataFrame(index= X.index)

        if self.feats_num:
            X_num = pd.DataFrame(
                self.transformer.transform(X[self.feats_num]),
                columns= self.feats_num,
                index= X.index
            )
        
        if self.return_dataframe:
            return X_num
        
        return X_num.values


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

        






























