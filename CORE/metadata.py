import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype

def inventario_features(data):
    data = data.copy()
    feats = []
    dtypes = []
    missing_pct = []
    unique_values = []
    cardinality_group = []
    is_constant = []
    is_near_constant = []
    semantic_type = []

    for i in data.columns:
        feats.append(i)
        dtypes.append(data[i].dtype.name)
        pct_missing = float(data[i].isnull().sum() / data[i].shape[0])
        missing_pct.append(pct_missing)
        #unique_value = 
        unique_values.append(data[i].nunique())
        if data[i].dtype.name in ['object', 'category', 'str', 'bool']:
            if data[i].nunique() <= 20:
                cardinality_group.append('Bajo')
            elif data[i].nunique() <= 50:
                cardinality_group.append('Medio')
            else:
                cardinality_group.append('Alto')
        else:
            cardinality_group.append('---')
        if data[i].nunique() == 1:
            is_constant.append(True)
        else:
            is_constant.append(False)
        if data[i].nunique() == 2:
            is_near_constant.append(True)
        else:
            is_near_constant.append(False)
        ## Semantic Type
        if is_numeric_dtype(data[i]):
            value = (
                (data[i] % 1 == 0).all() # todos enteros
                and (data[i].min() >= 0)
                and (data[i].nunique() <= 30)
                and (data[i].quantile(0.99) <= 100)
                and (data[i] <= 2).mean() >= 0.4
            )
            if value:
                semantic_type.append('Conteo')
            else:
                semantic_type.append('Continua')
        elif is_string_dtype(data[i]):
            semantic_type.append('String')
        elif is_datetime64_any_dtype(data[i]):
            semantic_type.append('Datetime')


    df_inventario_features = pd.DataFrame({
        'feature': feats,
        'dtype': dtypes,
        'semantic_dtype': semantic_type,
        'missing_pct': missing_pct,
        'unique_values': unique_values,
        'cardinality_group': cardinality_group,
        'is_constant': is_constant,
        'is_near_constant': is_near_constant
    }).sort_values(by= ['missing_pct', 'dtype'], ascending= [False, False])

    #self.inventario_features_df = df_inventario_features

    return df_inventario_features



























