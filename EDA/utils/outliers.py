import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
PARENT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PARENT_DIR))

from CORE.metadata import inventario_features

def outlier_detect_iqr(data, excluir_conteo = False):
        data = data.copy()
        data_numeric = data.select_dtypes(include= 'number')
        semantic_feats = inventario_features()
        if excluir_conteo:
            excluir_feat_conteo = semantic_feats.loc[semantic_feats['semantic_dtype'] == 'Conteo']['feature'].to_list()
            data_numeric = data_numeric.drop(excluir_feat_conteo, axis= 1)
        feats = []
        outliers_num_list = []
        outliers_pct_list = []
        lim_sup_list = []
        lim_inf_list = []
        for feat in data_numeric.columns.to_list():
            IQR = data_numeric[feat].quantile(0.75) - data_numeric[feat].quantile(0.25)
            limite_superior = data_numeric[feat].quantile(0.75) + (1.5 * IQR)
            limite_inferior = data_numeric[feat].quantile(0.25) - (1.5 * IQR)
            feats.append(feat)
            lim_inf_list.append(limite_inferior)
            lim_sup_list.append(limite_superior)
            ## Cantidad outliers y %
            outliers_n = (data[feat] > limite_superior).sum() +  (data[feat] < limite_inferior).sum()
            outliers_pct = outliers_n / (data[feat].shape[0])
            outliers_num_list.append(outliers_n)
            outliers_pct_list.append(f'{(outliers_pct * 100):.2f}%')
        
        outliers_iqr = pd.DataFrame({
            'Feature': feats,
            'Límite Superior': lim_sup_list,
            'Límite Inferior': lim_inf_list,
            'Reg. Fuera del Rango': outliers_num_list,
            '% Reg. Fuera del Rango': outliers_pct_list
        }).sort_values(by= 'Reg. Fuera del Rango', ascending= False)

        return outliers_iqr

def outlier_detect_percentil(data, excluir_conteo: False):
    data = data.copy()
    data_numeric = data.select_dtypes(include= 'number')
    semantic_feats = inventario_features()
    if excluir_conteo:
        excluir_feat_conteo = semantic_feats.loc[semantic_feats['semantic_dtype'] == 'Conteo']['feature'].to_list()
        data_numeric = data_numeric.drop(excluir_feat_conteo, axis= 1)
    rows_data = data_numeric.shape[0]
    feats = []
    p_min_list = []
    p_max_list = []
    registros_fuera_list = []
    registros_fuera_pct_list = []
    p_min_nom_list = []
    p_max_nom_list = []
    if rows_data < 5000:
        for feat in data_numeric.columns.tolist():
            p_min = data_numeric[feat].quantile(0.05)
            p_max = data_numeric[feat].quantile(0.95)
            p_min_nom = 'P5'
            p_max_nom = 'P95'
            feats.append(feat)
            p_min_list.append(round(p_min, 2))
            p_max_list.append(round(p_max, 2))
            registros_fuera = (data_numeric[feat] < p_min).sum() + (data_numeric[feat] > p_max).sum()
            registros_fuera_pct = registros_fuera / (data_numeric[feat].shape[0])
            registros_fuera_list.append(registros_fuera)
            registros_fuera_pct_list.append(f'{(registros_fuera_pct*100):.2f}%')
            p_min_nom_list.append(p_min_nom)
            p_max_nom_list.append(p_max_nom)
    elif rows_data < 100000:
        for feat in data_numeric.columns.tolist():
            p_min = data_numeric[feat].quantile(0.01)
            p_max = data_numeric[feat].quantile(0.99)
            p_min_nom = 'P1'
            p_max_nom = 'P99'
            feats.append(feat)
            p_min_list.append(round(p_min,2))
            p_max_list.append(round(p_max, 2))
            registros_fuera = (data_numeric[feat] < p_min).sum() + (data_numeric[feat] > p_max).sum()
            registros_fuera_pct = registros_fuera / (data_numeric[feat].shape[0])
            registros_fuera_list.append(registros_fuera)
            registros_fuera_pct_list.append(f'{(registros_fuera_pct*100):.2f}%')
            p_min_nom_list.append(p_min_nom)
            p_max_nom_list.append(p_max_nom)
    else:
        for feat in data_numeric.columns.tolist():
            p_min = data_numeric[feat].quantile(0.005)
            p_max = data_numeric[feat].quantile(0.995)
            p_min_nom = 'P0.5'
            p_max_nom = 'P99.5'
            feats.append(feat)
            p_min_list.append(round(p_min,2))
            p_max_list.append(round(p_max,2))
            registros_fuera = (data_numeric[feat] < p_min).sum() + (data_numeric[feat] > p_max).sum()
            registros_fuera_pct = registros_fuera / (data_numeric[feat].shape[0])
            registros_fuera_list.append(registros_fuera)
            registros_fuera_pct_list.append(f'{(registros_fuera_pct*100):.2f}%')
            p_min_nom_list.append(p_min_nom)
            p_max_nom_list.append(p_max_nom)

    df_outlier_percentil = pd.DataFrame({
        'Feature': feats,
        'PCT Min': p_min_nom,
        'Valor PCT minimo': p_min_list,
        'PCT Max': p_max_nom,
        'Valor PCT maximo': p_max_list,
        'Registros fuera': registros_fuera_list,
        '% Registros fuera': registros_fuera_pct_list
    }).sort_values(by= 'Registros fuera', ascending= False)
    return df_outlier_percentil