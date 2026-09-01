import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features_selected: list):
        self.features_selected = features_selected

    def fit(self, X, y=None):
        self.features_selected_ = self.features_selected
        return self

    def transform(self, X):
        X = X.copy()
        return X[self.features_selected_]
        






