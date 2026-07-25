import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor


def data_overview(data: pd.DataFrame, target_name, tipo_problema: str):
    filas = data.shape[0] # filas
    columnas = data.shape[1] # columnas
    # Features numéricas
    feat_num = data.select_dtypes(
        include= 'number'
    ).shape[1]
    # Features categóricas
    feat_cat = data.select_dtypes(
        include= ['string', 'category', 'object']
    ).shape[1]
    # Features Datetime
    feat_datetime = data.select_dtypes(
        include= 'datetime'
    ).shape[1]
    # Features booleanas
    feat_boolean = data.select_dtypes(
        include= 'bool'
    ).shape[1]
    # Missing Global
    missing_global = data.isnull().sum().sum()
    missing_ratio = (
        missing_global / data.size
    ) * 100
    # Feature con Missing
    feat_w_miss = (data.isnull().sum()>0).sum()
    # Feature con Missing > 20%
    feat_w_miss_0_20 = (data.isnull().sum()>0.2).sum()
    # Filas Duplicadas
    duplicados = data.duplicated().sum()
    # Features Constantes
    feat_constantes = (data.nunique()==1).sum()
    # Features casi constantes
    feat_casi_constantes = (data.nunique()==2).sum()
    # Features alta cardinalidad
    feat_alta_cardinalidad = (
        data.select_dtypes(include= ['string', 'category', 'object'])
        .nunique()>20).sum()
    # Features baja cardinalidad
    feat_baja_cardinalidad = (data.select_dtypes(
        include= ['object', 'string', 'category']
    ).nunique()<20).sum()
    # Tipo de Problema
    tipo_problema = tipo_problema
    # Target
    target = target_name
    df_data_overview = pd.DataFrame({
        'Filas': filas,
        'Columnas': columnas,
        'Features Numéricas': feat_num,
        'Features Categóricas': feat_cat,
        'Features Datetime': feat_datetime,
        'Features Booleanas': feat_boolean,
        'Missing Global (%)': missing_global,
        'Features con Missing': feat_w_miss,
        'Features > 20% Missing': feat_w_miss_0_20,
        'Filas Duplicadas': duplicados,
        'Features Constantes': feat_constantes,
        'Features Casi Constantes': feat_casi_constantes,
        'Features Alta Cardinalidad': feat_alta_cardinalidad,
        'Features Baja Cardinalidad': feat_baja_cardinalidad,
        'Tipo Problema': tipo_problema,
        'Target': target_name
    }, index= ['Data Overview'])
    return df_data_overview.T































