import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
from matplotlib.pyplot import pyplot as plt
from Feature_Engineering.utils.selection.iv import information_value

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class Binning(MinMaxScaler, RobustScaler):
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

            X.loc[X['feat'].isna(), 'bins'] = 'MISSING'

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

    def transform(self, X_base: pd.DataFrame, y_base: pd.Series, operation = 'iv', X_comparative_1: pd.DataFrame = None, y_comparative_1: pd.Series = None):

        if operation == 'iv':
            df_iv = information_value(
                feat_bins= self.feat_bins,
                X= X_base,
                y= y_base,
                feat_numeric= self.feats_numeric,
                feat_cat= self.feats_cat
            )

            return df_iv

        return 'Operación no identificada'

class EqualFrecuencyBinning:
    def __init__(self):
        pass

class OptimalBinning:
    def __init__(self):
        pass
        
        
        






























