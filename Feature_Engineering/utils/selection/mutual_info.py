import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PARENT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PARENT_DIR))

from CORE.metadata import inventario_features



