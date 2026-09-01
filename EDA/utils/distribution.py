import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def numeric_distributions(data: pd.DataFrame, target_name: str):
    
    ## Distribución y BoxPlot
    data_numeric = data.select_dtypes(include= 'number')
    data_numeric_wo_target = [feat for feat in data_numeric.columns.to_list() if feat != 'churn']
    fig, ax = plt.subplots(nrows=len(data_numeric_wo_target) , ncols=3, figsize = (10, 25))
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

    return plt.show()


# Distribuciones de los features categóricas
def categorical_distributions(data: pd.DataFrame, target_name: str, exclude: list = []):
    data_categorical = data.copy()
    data_categorical = data_categorical.select_dtypes(include= ['string', 'object', 'category'])
    data_categorical_w_target = pd.concat([data_categorical, data[target_name]], axis= 1)
    feats_categoricas = [feat for feat in data_categorical.columns.to_list() if feat not in exclude]
    fig, ax = plt.subplots(nrows= len(feats_categoricas) , ncols=2, figsize = (8, 13))
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

    return plt.show()


def analisis_target(data: pd.DataFrame, target_name: str):
    data_df = data.copy()
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
