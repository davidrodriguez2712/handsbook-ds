import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

def correlation(X: pd.DataFrame, method= 'pearson'):
    """Devuelve un dataframe priorizado por correlación mayor a menor"""
    X= X.copy()
    feats_numericas = X.select_dtypes(include= 'number')
    combinaciones = list(combinations(feats_numericas, 2))
    dict_correlations = []

    for feat1, feat2 in combinaciones:
        tmp_var = f'{feat1}_{feat2}'
        corr = X[feat1].corr(X[feat2], method= method)

        dict_correlations.append({
            'Features': tmp_var,
            'Correlación': round(corr, 4)
        })

    df_correlations = pd.DataFrame(dict_correlations).sort_values(by= 'Correlación', ascending= False)

    return df_correlations



"""

ef eda_relaciones_feats_numericas(self):
        data = self.data.copy()
        target_name = self.target_name
        corr = data.corr('pearson', numeric_only= True)
        mask = np.triu(
            np.ones_like(corr, dtype= bool)
        )
        fig, ax = plt.subplots(nrows= 5, ncols= 1, figsize = (6, 16))
        # Gráfico 1: Correlación Pearson
        sns.heatmap(
            data= corr,
            mask= mask,
            ax= ax[0],
            annot= True,
            cmap= 'coolwarm',
            square= False,
            fmt = '.2f',
            annot_kws= {
                'fontsize': 8
            }
        )
        ax[0].set_title('Correlación de Pearson', fontsize = 10)
        ax[0].tick_params(axis = 'x', labelsize = 7)
        ax[0].tick_params(axis = 'y', labelsize = 7)

        ## Gráfico 2: Correlación Spearman

        corr_spearman = data.corr('spearman', numeric_only= True)
        mask_spearman = np.triu(
            np.ones_like(corr_spearman, dtype= bool)
        )

        sns.heatmap(
            data= corr_spearman,
            mask= mask_spearman,
            cmap= 'coolwarm',
            annot = True,
            annot_kws= {
                'fontsize': 8
            },
            fmt= '.2f',
            ax= ax[1]
        )
        ax[1].set_title('Correlación de Spearman', fontsize = 10)
        ax[1].tick_params(axis = 'x', labelsize = 7)
        ax[1].tick_params(axis = 'y', labelsize = 7)

        ## Gráfico 3: Mutual information

        ## Entrenamiento de MI
        data_numeric = data.select_dtypes(include='number').drop([target_name], axis= 1)
        mi_scores = mutual_info_classif(
        X= data_numeric,
        y= data[target_name],
        discrete_features= 'auto',
        n_neighbors= 3,
        random_state= 42
        )
        mi_df = pd.DataFrame({
            'feature': data_numeric.columns,
            'mutual_info': mi_scores
        }).sort_values(by= 'mutual_info', ascending = False).reset_index(level=0, drop= False)


        ax[2].barh(
            mi_df['feature'][::-1],
            mi_df['mutual_info'][::-1]
        )
        ax[2].set_title('Mutual Information', fontsize = 10)
        ax[2].tick_params(axis = 'x', labelsize = 7)
        ax[2].tick_params(axis = 'y', labelsize = 7)

        ## Gráfico 4: VIF

        ## Entrenamiento del VIF
        vif_df = pd.DataFrame({
        'feature': data_numeric.columns,
        'VIF': [
            variance_inflation_factor(data_numeric.values, i)
            for i in range(data_numeric.shape[1])
        ]
        }).sort_values(by= 'VIF', ascending = False)

        ax[3].barh(
            vif_df['feature'][::-1],
            vif_df['VIF'][::-1]
        )
        ax[3].set_title('Análisis del VIF', fontsize = 10)
        ax[3].tick_params(axis = 'x', labelsize = 7)
        ax[3].tick_params(axis = 'y', labelsize = 7)
        ax[3].axvline(
            x = 5,
            linestyle = '--',
            color = 'tomato',
            linewidth = 2
        )

        ## Gráfico 5: IV
        df_iv, lista_bins = self.information_value()
        ax[4].barh(
            df_iv['Feature'][::-1],
            df_iv['Information Value'][::-1]
        )
        ax[4].set_title('Information Value de las Features', fontsize = 10)
        ax[4].tick_params(axis = 'x', labelsize = 7)
        ax[4].tick_params(axis = 'y', labelsize = 7)
        ax[4].axvline(
            x = 0.1,
            color = 'tomato',
            linestyle = '--',
            linewidth = 2
        )

        plt.tight_layout()
        plt.close(fig)
        return fig
"""




















