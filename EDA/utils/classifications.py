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
from sklearn.model_selection import train_test_split

class EDAClasificacion:
    def __init__(self, data, target_name, tipo_problema = "Clasificación Binaria"):
        self.target_name = target_name
        self.data = data
        self.tipo_problema = tipo_problema

    def data_overview(self):
        data = self.data.copy()
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
        tipo_problema = self.tipo_problema
        # Target
        target = self.target_name
        # Target Rate (%)
        target_rate = f'{(data[self.target_name].mean()*100):.2f}%'
        # No event rate (%)
        no_event_rate = f'{((data[self.target_name] == 0).mean()*100):.2f}%'
        # Ratio Desbalanceo
        ###
        ratio_desbalanceo = round(((data[self.target_name] == 0).mean()) / (data[self.target_name].mean()), 0)

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
            'Target': self.target_name,
            'Target Rate (%)': target_rate,
            'No event Rate (%)': no_event_rate,
            'Ratio Desbalanceo': ratio_desbalanceo
        }, index= ['Data Overview'])
        return df_data_overview.T
    
    def inventario_features(self):
        data = self.data.copy()
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
            missing_pct.append(f'{((data[i].isnull().sum() / data[i].shape[0])*100):.2f}%')
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


    def missings_analisis(self):
        data = self.data.copy()
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
    
    def outlier_detect_iqr(self):
        data = self.data.copy()
        data_numeric = data.select_dtypes(include= 'number')
        semantic_feats = self.inventario_features()
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

    def outlier_detect_percentil(self):
        data = self.data.copy()
        data_numeric = data.select_dtypes(include= 'number')
        semantic_feats = self.inventario_features()
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
    
    def numeric_distributions(self):
        data = self.data.copy()
        target_name = self.target_name
        ## Distribución y BoxPlot
        data_numeric = data.select_dtypes(include= 'number')
        data_numeric_wo_target = [feat for feat in data_numeric.columns.to_list() if feat != 'churn']
        n_feats = len(data_numeric_wo_target)
        fig, ax = plt.subplots(nrows=len(data_numeric_wo_target) , ncols=3, figsize = (10, n_feats * 3.5))
        for i, var in enumerate(data_numeric_wo_target):
            ax[i,0].hist(
                data_numeric[var],
                bins = 30
                #color='tab:blue'
            )
            ax[i, 0].set_title(f'Distribución de {var}', fontsize = 9)
            ax[i, 0].tick_params(axis= 'x', labelsize = 9)
            ax[i, 0].tick_params(axis = 'y', labelsize = 9)

            sns.boxplot(
                data = data_numeric,
                x = target_name,
                y = var,
                ax= ax[i,2]
                #orient = 'v'
                
            )
            ax[i,2].set_title(f'Boxplot {var}', fontsize = 9)
            ax[i,2].tick_params(axis='x', labelsize = 9)
            ax[i,2].tick_params(axis='y', labelsize = 9)
            ax[i,2].set_xlabel("")
            ax[i,2].set_ylabel("")

            df_tmp = data_numeric.copy()
            df_tmp['bin'] = pd.qcut(
                x= data_numeric[var],
                q= 10,
                duplicates= 'drop'
            )
            agg = df_tmp.groupby(
                ['bin']
            ).agg(
                event_rate = (target_name, 'mean')
            ).reset_index(level= 0, drop= False)
            agg['bin'] = agg['bin'].astype(str)
            
            sns.lineplot(
                data= agg,
                ax= ax[i,1],
                x= 'bin',
                y= 'event_rate',
                markers= 'o',
                
            )
            ax[i,1].set_title(f'Relación {var} vs Target Rate', fontsize = 9)
            ax[i,1].tick_params(axis = 'x', labelsize = 9, labelrotation= 34)
            ax[i,1].tick_params(axis = 'y', labelsize = 9)
            ax[i,1].set_xlabel("")
            ax[i,1].set_ylabel("Target Rate", fontsize = 9)
            
        plt.suptitle('Distribución, BoxPlot y Relación con el Target', y= 1.0001)
        plt.tight_layout()
        plt.close(fig)

        return fig
    
    # Distribuciones de los features categóricas
    def categorical_distributions(self):
        data = self.data.copy()
        target_name = self.target_name
        exclude = ['customer_id']
        data_categorical = data.select_dtypes(include= ['string', 'object', 'category'])
        data_categorical_w_target = pd.concat([data_categorical, data[target_name]], axis= 1)
        feats_categoricas = [feat for feat in data_categorical.columns.to_list() if feat not in exclude]
        n_feats = len(feats_categoricas)
        fig, ax = plt.subplots(nrows= len(feats_categoricas) , ncols=2, figsize = (10, n_feats * 3.5))
        for i, var in enumerate(feats_categoricas):
            agg = data_categorical_w_target.groupby(
                [var]
            ).agg(
                mean_target = (target_name, 'mean'),
                count_var = (var, 'count')
            ).reset_index(level=0, drop= False).sort_values(by= 'mean_target', ascending= True)
            
            ax[i,0].plot(
                agg[var],
                agg['mean_target'],
                marker = 'o'
            )
            ax[i,0].tick_params(axis = 'x', labelsize = 7, rotation = 45)
            ax[i,0].tick_params(axis = 'y', labelsize = 7)
            ax[i,0].set_title(f'Feature {var} vs Target Rate', fontsize = 11)

            agg = agg.sort_values(by= 'count_var', ascending = False)
            bar = ax[i,1].bar(
                agg[var],
                agg['count_var'],
            )
            ax[i,1].bar_label(bar, fontsize = 9)
            ax[i,1].set_title(f'Distribución de {var}', fontsize = 11)
            ax[i,1].tick_params(axis = 'x', labelsize = 9, rotation = 45)
            ax[i,1].tick_params(axis = 'y', labelsize = 9)
        plt.tight_layout()
        plt.close(fig)

        return fig
    
    def analisis_target(self):
        data_df = self.data.copy()
        target_name = self.target_name
        fig, ax = plt.subplots(nrows= 1, ncols= 2, figsize = (7, 4))
        bars = ax[0].bar(
            x = [0, 1],
            height= data_df[target_name].value_counts(),
            color = ["steelblue", "tomato"]
        )

        ax[0].bar_label(bars, fmt = "%d")
        ax[0].set(title = 'Conteo del Target', ylabel = 'Cantidad', xlabel = 'Target')
        ax[0].set_xticks([0, 1])

        ax[1].pie(
            x = data_df[target_name].value_counts().reindex([0, 1], fill_value= 0),
            labels = ["No default", "Default"],
            colors = ["steelblue", "tomato"],
            autopct = "%.0f%%",
            startangle = 90
        )
        ax[1].set_title("Distribución del target")

        plt.tight_layout()
        plt.close(fig)

        # Registros
        n_registros = data_df[target_name].count()
        
        # Missing target
        missing_target = data_df[target_name].isnull().sum()

        # Clases únicas
        number_of_class = data_df[target_name].nunique()

        # Positive Rate
        positive_rate = data_df[target_name].mean()

        # Negative Rate
        negative_rate = (data_df[target_name] == 0).mean()

        # Imbalance ratio
        imbalance_ratio = f'{(negative_rate / positive_rate):.2f}:1'                         

        # Baseline accuracy
        baseline_accuracy = f'{(negative_rate*100):.2f}%'

        # Nivel de desbalance
        if positive_rate < 0.05:
            level_imbalance = 'Muy Desbalanceado'
        elif positive_rate < 0.20:
            level_imbalance = 'Desbalanceado'
        elif positive_rate < 0.45:
            level_imbalance = 'Moderadamente desbalanceado'
        else:
            level_imbalance = 'Balanceado'

        target_analisis_df = pd.DataFrame([{
            'Registros': n_registros,
            'Missing target': missing_target,
            'Clases únicas': number_of_class,
            'Positive rate': positive_rate,
            'Negative rate': negative_rate,
            'Imbalance rate': imbalance_ratio,
            'Baseline accuracy': baseline_accuracy,
            'Nivel de desbalance': level_imbalance
        }])

        return fig, target_analisis_df
    
    def information_value(self):
        """Retorna una tabla resumen del IV de cada feature numérica"""
        df = self.data.copy()
        target_name = self.target_name
        data_numeric = df.select_dtypes(include= 'number').drop([target_name], axis= 1)
        lista_iv = []
        lista_bins = []

        for i in data_numeric.columns:
            df['bins'], bins = pd.qcut(
                x = df[i],
                q = 5,
                retbins= True,
                duplicates= 'drop'
            )
            eps = 1e-5
            table = df.groupby(
                ['bins'], observed=False
            ).agg(
                total_obs = (target_name, "count"),
                total_event = (target_name, "sum"),
                event_rate = (target_name, "mean")
            )
            table['total_no_event'] = table["total_obs"] - table["total_event"]
            table['no_event_rate'] = table['total_no_event'] / table['total_obs']
            table['dist_event'] = table['total_event'] / table['total_event'].sum()
            table['dist_no_event'] = table['total_no_event'] / table['total_no_event'].sum()
            table['woe'] = (
                np.log(
                    (table['dist_no_event'] + eps) / (table['dist_event'] + eps)
                )
            )
            table = table.reset_index(level= 0, drop= False)
            table['information_value_calculation'] = ((table['dist_no_event'] - table['dist_event']) * (table['woe']))
            iv = (table['information_value_calculation'].sum())
            #print(table['woe'].count())
            lista_iv.append(iv)
            lista_bins.append(bins)
        
        df_iv = pd.DataFrame({
            'Feature': data_numeric.columns.to_list(),
            'Information Value': lista_iv
        }).sort_values(by= 'Information Value', ascending= False)

        return df_iv, lista_bins
    

    def eda_relaciones_feats_numericas(self):
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


    def calculate_psi(self, expected, actual):
        """
        Calcula el PSI a partir de dos distribuciones.
        expected y actual deben ser arrays de proporciones.
        """
        EPSILON = 1e-6
        expected = np.clip(expected, EPSILON, None)
        actual = np.clip(actual, EPSILON, None)

        return np.sum((actual - expected) * np.log(actual / expected))

    def interpret_psi(self, psi):

        if psi < 0.10:
            return "Sin drift"

        elif psi < 0.25:
            return "Drift moderado"

        else:
            return "Drift severo"

    def psi_numeric(self, train, test, bins=10):
        """
        PSI para variables numéricas usando cuantiles del Train.
        """

        # bins usando el Train
        cut_points = np.unique(
            np.quantile(train.dropna(), np.linspace(0, 1, bins + 1))
        )

        train_bins = pd.cut(
            train,
            bins=cut_points,
            include_lowest=True,
            duplicates="drop"
        )

        test_bins = pd.cut(
            test,
            bins=cut_points,
            include_lowest=True,
            duplicates="drop"
        )

        train_dist = (
            train_bins.value_counts(normalize=True)
            .sort_index()
        )

        test_dist = (
            test_bins.value_counts(normalize=True)
            .reindex(train_dist.index, fill_value=0)
        )

        psi = self.calculate_psi(
            train_dist.values,
            test_dist.values
        )

        return psi


    def psi_categorical(self, train, test):
        """
        PSI para variables categóricas.
        """

        train_dist = (
            train.fillna("MISSING")
            .value_counts(normalize=True)
        )

        test_dist = (
            test.fillna("MISSING")
            .value_counts(normalize=True)
        )

        categories = train_dist.index.union(test_dist.index)

        train_dist = train_dist.reindex(categories, fill_value=0)
        test_dist = test_dist.reindex(categories, fill_value=0)

        psi = self.calculate_psi(
            train_dist.values,
            test_dist.values
        )

        return psi


    def dataframe_psi(self, test_df):
        """
        Calcula PSI para todas las columnas.
        """
        train_df = self.data.copy()
        #X_train, X_test, y_train, y_test = train_test_split()
        results = []

        for col in train_df.columns:

            if pd.api.types.is_numeric_dtype(train_df[col]):
                psi = self.psi_numeric(
                    train_df[col],
                    test_df[col]
                ).round(4)
                var_type = "numeric"
                interpretacion = self.interpret_psi(psi= psi)

            else:
                psi = self.psi_categorical(
                    train_df[col],
                    test_df[col]
                ).round(4)
                var_type = "categorical"
                interpretacion = self.interpret_psi(psi= psi)

            results.append({
                "feature": col,
                "type": var_type,
                "psi": psi,
                "interpretacion": interpretacion
            })

        results = (
            pd.DataFrame(results)
            .sort_values("psi", ascending=False)
            .reset_index(drop=True)
        )

        return results

    def run_all(self, data_comparacion: pd.DataFrame | None = None, include_psi: bool = False, show = True):
        """Corre todo y retorna todo el EDA, además guarda los artefactos en sus carpetas respectivas"""

        if include_psi:
            resultados = {
                'Data Overview': self.data_overview(),
                'Inventario de Features': self.inventario_features(),
                'Analisis de Missing': self.missings_analisis(),
                'Detector de Outliers - IQR': self.outlier_detect_iqr(),
                'Detector de Outlier - Percentiles': self.outlier_detect_percentil(),
                'Distribucion Feats Numericas': self.numeric_distributions(),
                'Distribucion Feats Categoricas': self.categorical_distributions(),
                'Analisis del Target - Gráficos': self.analisis_target()[0],
                'Analisis del Target - Resumen Tabla': self.analisis_target()[1],
                'Information Value': self.information_value()[0],
                'Relaciones entre Variables': self.eda_relaciones_feats_numericas(),
                'PSI': self.dataframe_psi(data_comparacion)
            }
        else:
            resultados = {
                'Data Overview': self.data_overview(),
                'Inventario de Features': self.inventario_features(),
                'Analisis de Missing': self.missings_analisis(),
                'Detector de Outliers - IQR': self.outlier_detect_iqr(),
                'Detector de Outlier - Percentiles': self.outlier_detect_percentil(),
                'Distribucion Feats Numericas': self.numeric_distributions(),
                'Distribucion Feats Categoricas': self.categorical_distributions(),
                'Analisis del Target': self.analisis_target(),
                'Information Value': self.information_value(),
                'Relaciones entre Variables': self.eda_relaciones_feats_numericas()
            }
        
        if show:
            for title, output in resultados.items():
                print("")
                print(f'{"="*50}')
                print(f'{title}')
                print(f'{"="*50}')
                print("")

                display(output)
        
        return resultados


        # Data Overview
        # data_overview(self): -> Retorna: DataFrame
        
        # Inventario de Feautres
        # def inventario_features(self): -> Retorna: DataFrame


        # Missing Analysis
        # def missings_analisis(self): -> Retorna: fig



        # Detector de outlier por IQR
        # def outlier_detect_iqr(self): -> Retorna: DataFrame

        # Detector de outlier por percentiles
        # def outlier_detect_percentil(self): -> Retorna: DataFrame

        # Analiza distribuciones numéricas -> Retorna: fig
        # def numeric_distributions(self):

        # Analiza distribuciones de las categóricas -> Retorna: fig
        # def categorical_distributions(self):

        # Analiza el target -> Retorna: fig, DataFrame
        # def analisis_target(self):

        # Muestra el IV de cada feature -> Retorna: DataFrame, lista[de los bins]
        # def information_value(self):

        # Análisis de relaciones entre features  -> Retorna: fig
        # def eda_relaciones_feats_numericas(self):

        # PSI de cada variable base vs comparación -> Retorna: DataFrame
        # def dataframe_psi(self, train_df, test_df):


















    



























































