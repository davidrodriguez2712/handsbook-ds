import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from Feature_Engineering.utils.selection.iv import information_value

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, strategy = 'width', show_dataframe = True):
        self.strategy = strategy
        self.show_dataframe = show_dataframe

    def fit(self, X: pd.DataFrame):
        """Retorna una tabla resumen del IV de cada feature numérica"""
        X = X.copy()
        self.feats_numeric = X.select_dtypes(include= 'number').columns.tolist()
        self.feats_cat = X.select_dtypes(include= ['string', 'object', 'category']).columns.tolist()
        self.feat_bins = {}

        for feat in self.feats_numeric:
            if self.strategy == 'width':
                X['bins'], bins = pd.cut(
                    x = X[feat],
                    bins= 10,
                    retbins= True,
                    duplicates= 'drop'
                )

            elif self.strategy == 'frecuency':
                X['bins'], bins = pd.qcut(
                    x = X[feat],
                    q= 10,
                    retbins= True,
                    duplicates= 'drop'
                )

            X['bins'] = X['bins'].astype('object')
            X.loc[X[feat].isna(), 'bins'] = 'MISSING'

            self.feat_bins[feat] = {
                'feature': feat,
                'bins': bins
            }

        for feat in self.feats_cat:
            labels = X[feat].dropna().unique().tolist()
            labels = labels.append('MISSING')
            labels = labels.append('UNKNOWN')

            self.feat_bins[feat] = {
                'feature': feat,
                'bins': labels
            }

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series, operation = 'iv'):
        X_base = X.copy()
        y_base = y.copy()

        if operation == 'iv':
            print(type(y_base))
            df_iv = information_value(
                X= X_base,
                y= y_base,
                feat_bins= self.feat_bins,
                feat_numeric= self.feats_numeric,
                feat_cat= self.feats_cat
            )

            return df_iv

        return 'Operación no identificada'

        
        
        






























