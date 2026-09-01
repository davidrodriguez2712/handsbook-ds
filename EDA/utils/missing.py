import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

def missings_analisis(data):
    data = data.copy()
    # Missings
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize = (10, 12))
    axes = ax.flatten()

    sns.heatmap(
        data= data.isnull().transpose(),
        cmap= 'YlGnBu',
        cbar_kws = {'label': 'Valores perdidos'},
        ax= axes[0]
    )
    axes[0].set_title('Distribución de valores perdidos')

    msno.heatmap(
        df= data,
        cmap= 'RdYlGn',
        figsize= (10,5),
        fontsize= 12,
        ax = axes[1]
    )
    axes[1].set_title('Correlación de nulidad entre variables')

    missing_list = data.columns.tolist()
    cant_missing = [data[feat].isnull().sum() for feat in data.columns.tolist()]
    data_missing = pd.DataFrame({
        'feat': missing_list,
        'cant_missing': cant_missing
    })
    data_missing = (
    data_missing[data_missing["cant_missing"] > 0]
    .sort_values("cant_missing", ascending=False)
    )
    axes[2].bar(
        data_missing['feat'],
        data_missing['cant_missing'],
    )
    axes[2].tick_params(axis = 'x', rotation = 45, labelsize = 7)
    axes[2].set_title('Cantidad de Missing por Feature')
    plt.tight_layout()
    plt.close(fig)

    return fig