import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve, roc_curve, average_precision_score, brier_score_loss
import joblib
PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')


def threshold_f1(precision, recall):
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def curves_roc_pr(y_true, y_score):
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true= y_true, y_score= y_score)
    pr_auc = average_precision_score(y_true= y_true, y_score = y_score)
    t = np.argmax(threshold_f1(precision, recall))
    
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].plot(
        fpr,
        tpr
    )

    ax[0].set_title('Curva ROC')
    ax[0].set_ylabel('TPR')
    ax[0].set_xlabel('FPR')
    ax[0].plot([0,1],[0,1], linestyle='--')
    ax[0].text(
        0.05, 0.98,
        f'ROC AUC: {round(roc_auc, 3)} | Gini: {round(roc_auc*2 - 1, 3)}',
        transform = ax[0].transAxes,
        ha = 'left',
        va = 'top',
        fontsize = 9
        #fontweight = 'bold'
    )

    ax[1].plot(
        recall,
        precision
    )
    ax[1].set_title('Curva PR')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Recall')
    ax[1].axhline(precision[t], color = 'red', linestyle = '--', alpha = 0.5)
    ax[1].axvline(recall[t], color = 'red', linestyle = '--', alpha = 0.5)
    ax[1].text(
        0.10, 0.98,
        f'PR AUC: {round(pr_auc, 3)} | Best F1 Threshold: {round(thresholds_pr[t], 3)}',
        transform = ax[1].transAxes,
        ha = 'left',
        va = 'top',
        fontsize = 9
        #fontweight = 'bold'
    )
    plt.close(fig)
    
    return fig

def calibration_curve(y_true, y_score, method_bin = 'qcut', show_curve= True):
    # crear los deciles
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    brier_score = brier_score_loss(
        y_true= y_true,
        y_proba = y_score
    )
    if method_bin == 'qcut':
        df_mix['bins'] = pd.qcut(
            x= df_mix['y_score'],
            q= 10
        )
    elif method_bin == 'cut':
        df_mix['bins'] = pd.cut(
            x= df_mix['y_score'],
            bins= 10
        )
    table = df_mix.groupby('bins', observed= True).agg(
        materialidad = ('y_score', 'count'),
        y_real_mean = ('y_true', 'mean'),
        y_score_mean = ('y_score', 'mean')
    ).reset_index()
    table['dif_nom'] = table['y_real_mean'] - table['y_score_mean']
    table['dif_var_%'] = round(((table['y_real_mean'] - table['y_score_mean'])/ table['y_score_mean'])*100, 2)

    # gráfico de curva
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.plot(
        table['y_score_mean'],
        table['y_real_mean'],
        marker = 'o'
    )

    ax.plot(
        [0, max( table['y_real_mean'].max(), table['y_score_mean'].max() ) + 0.05 ],
        [0, max( table['y_real_mean'].max(), table['y_score_mean'].max() ) + 0.05 ],
        '--',
        color= 'tomato'
    )
    ax.set_title(f'Curva de Calibración - Brier Score: {brier_score:.4f}')
    ax.set_ylabel('Promedio Default Rate')
    ax.set_xlabel('Promedio del Score Predicho')
    plt.close(fig)
    if show_curve:
        return table, fig
    return table



def ks_table(y_true, y_score, method_bin = 'qcut', show_curve= False):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['decil'] = pd.qcut(
            x= -df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['decil'] = pd.cut(
            x= -df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('decil').agg(
        materialidad = ('y_true', 'count'),
        y_real_event_count = ('y_true', 'sum')
    ).reset_index()
    table['y_real_no_event_count'] = table['materialidad'] - table['y_real_event_count']
    table['malos_decil_%'] = ((table['y_real_event_count'] / table['y_real_event_count'].sum()) * 100)
    table['buenos_decil_%'] = ((table['y_real_no_event_count'] / table['y_real_no_event_count'].sum())*100)
    table['Malos_Acumulados_%'] = (table['malos_decil_%'].cumsum()).round(0).astype(int)
    table['Buenos_Acumulados_%'] = round(table['buenos_decil_%'].cumsum()).round(0).astype(int)
    table['KS'] = table['Malos_Acumulados_%'] - table['Buenos_Acumulados_%']
    max_ks = table['KS'].max()
    decil_max = (table.loc[table['KS'] == max_ks, 'decil'].iloc[0]).astype(int)
    #print(type(decil_max))
    FEATS_A_MOSTRAR = ['decil', 'Malos_Acumulados_%', 'Buenos_Acumulados_%', 'KS']

    if show_curve:
        # Mostrando la curva de malos acumulados y buenos acumulados
        
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.plot(
            table['decil'],
            table['Malos_Acumulados_%'],
            label = 'Malos Acumulados %',
            color = 'red',
        )
        ax.set_xlabel('Deciles')
        ax.set_ylabel('Malos Acumulados %')
        ax.set_xticks(table['decil'])
        ax.set_ylim(0, 105)

        ax2 = ax.twinx()
        ax2.plot(
            table['decil'],
            table['Buenos_Acumulados_%'],
            label = 'Buenos Acumulados %',
            color = 'green'
        )
        ax2.set_ylabel('Buenos Acumulados %')
        ax2.set_ylim(0,105)
        plt.tight_layout()
        plt.suptitle('Distribución Buenos y Malos Acumulados % por Decil', y = 1.03)
        #print(f'prueba 1: {}')
        line_ks_v = ax.axvline(table['decil'][decil_max-1], linestyle = '--', color = 'black', alpha = 0.5, label = 'KS Máximo')
        ax.axhline(table['Malos_Acumulados_%'][decil_max-1], linestyle = '--', color = 'black', alpha = 0.5)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='lower right'
        )
        #ax2.legend()
        plt.close(fig)
        return table[FEATS_A_MOSTRAR], fig

    return table[FEATS_A_MOSTRAR]


def bad_rate_decil(y_true, y_score, method_bin = 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Bad_Rate = ('y_true', 'mean')
    ).reset_index()
    table['Bad_Rate'] = (table['Bad_Rate'] * 100).round(2)
    table = table.rename(columns = {'Bad_Rate': 'Bad_Rate_%'})
    table['diff'] = table['Bad_Rate_%'].diff(1)
    table['Monotonicidad'] = np.where(table['diff'] <= 0, 'Viola Monotonicidad', 'No viola Monotonicidad')
    table = table.drop(['diff'], axis = 1)
    return table


def bad_rate_acumulado(y_true, y_score, method_bin = 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Bad_Rate = ('y_true', 'mean')
    ).reset_index()
    table['Corte'] = table['Decil'].apply(
        lambda x: f'Hasta decil {x}'
    )
    table['Clientes_Aprobados'] = table['Clientes'].cumsum()
    table['Malos_Aprobados'] = table['Malos'].cumsum()
    table['Bad_rate_aprobado_%'] = ((table['Malos_Aprobados'] / table['Clientes_Aprobados']) * 100).round(1)
    FEATS_A_MOSTRAR = ['Corte', 'Clientes_Aprobados', 'Malos_Aprobados', 'Bad_rate_aprobado_%']
    return table[FEATS_A_MOSTRAR]

def lift_acumulado(y_true, y_score, method_bin= 'qcut', show_curve= False):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= -df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= -df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum')
    ).reset_index()
    table['Poblacion_Acumulada_%'] = ((table['Clientes'].cumsum() / table['Clientes'].sum())*100).round(0).astype(int)
    table['Malos_Acumulados'] = table['Malos'].cumsum()
    table['Total_Malos'] = table['Malos'].sum()
    table['Gain_%'] = ((table['Malos'].cumsum() / table['Malos'].sum())*100).round(0).astype(int)
    table['Lift_Acumulado'] = (table['Gain_%'] / table['Poblacion_Acumulada_%']).round(2)
    FEATS_ELEGIDAS = ['Poblacion_Acumulada_%', 'Malos_Acumulados', 'Total_Malos', 'Gain_%', 'Lift_Acumulado']

    if show_curve:
        fig, ax = plt.subplots(1, 1, figsize = (8, 4))
        ax.plot(
            table['Poblacion_Acumulada_%'],
            table['Gain_%'],
            color= 'red'
        )
        ax.set_ylabel('Malos Acumulados Capturados % (Gain)')
        ax.set_xlabel('Población Acumulada %')
        ax.set_title('Curva CAP (Captura Acumulada de Malos)')
        ax.set_xticks(table['Poblacion_Acumulada_%'])
        ax.set_ylim(0, 105)
        plt.tight_layout()
        plt.close(fig)

        return table[FEATS_ELEGIDAS], fig
    return table[FEATS_ELEGIDAS]


def lift_bandas(y_true, y_score, method_bin= 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Bad_Rate = ('y_true', 'mean')
    ).reset_index()
    bad_rate_total = table['Malos'].sum() / table['Clientes'].sum()
    table = table.rename(columns= {'Bad_Rate': 'Bad_Rate_%'})
    table['Lift_por_Banda'] = (table['Bad_Rate_%'] / bad_rate_total).round(1)
    table['Bad_Rate_%'] = (table['Bad_Rate_%']*100).round(1)
    FEATS_ELEGIDAS = ['Decil', 'Bad_Rate_%', 'Lift_por_Banda']

    return table[FEATS_ELEGIDAS]
    


def tabla_deciles_completa(y_true, y_score, method_bin = 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= -df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= -df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum')
    ).reset_index()
    table['Buenos'] = table['Clientes'] - table['Malos']
    table['Bad_Rate'] = table['Malos'] / table['Clientes']
    table['Malos_Acumulados_%'] = ((table['Malos'].cumsum() / table['Malos'].sum())*100).round(0).astype(int)
    table['Buenos_Acumulados_%'] = ((table['Buenos'].cumsum() / table['Buenos'].sum())*100).round(0).astype(int)
    table['KS_%'] = table['Malos_Acumulados_%'] - table['Buenos_Acumulados_%']
    table['Gain_%'] = ((table['Malos'].cumsum() / table['Malos'].sum())*100).round(0).astype(int)
    table['Lift'] = (table['Gain_%'] / (table['Decil'].astype(int) * 10)).round(1)
    table['Bad_Rate_%'] = (table['Bad_Rate'] * 100).round(1)
    FEATS_A_MOSTRAR = ['Decil', 'Clientes', 'Malos', 'Buenos', 'Bad_Rate_%', 'Malos_Acumulados_%','KS_%', 'Gain_%', 'Lift']

    return table[FEATS_A_MOSTRAR]



def top_decile_capture(y_true, y_score, method_bin= 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= -df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= -df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Bad_Rate = ('y_true', 'mean')
    ).reset_index()
    malos_totales = table['Malos'].sum()
    max_malos = table['Malos'].max()
    decil_max_malos = (table.loc[table['Malos'] == max_malos, 'Decil'].iloc[0]).astype(int)
    top_decile_captura = ((max_malos / malos_totales) * 100).round(0).astype(int)
    df_final = pd.DataFrame([{
        'Decil': decil_max_malos,
        'Malos en el decil': max_malos,
        'Malos Totales': malos_totales,
        'Top Decil Capture_%': top_decile_captura,
        'Interpretación': f'El peor 10% de clientes concentra el {top_decile_captura}% de todos los default de la cartera'
    }])
    return df_final


## Métricas ponderadas por saldo

def tabla_deciles_completa_saldos(y_true, y_score, series_saldo: pd.Series, method_bin = 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score,
        'saldo': series_saldo
    })
    if method_bin == 'qcut':
        df_mix['Decil'] = pd.qcut(
            x= -df_mix['y_score'],
            q= 10,
            labels = False,
            duplicates= 'drop'
        ) + 1
    elif method_bin == 'cut':
        df_mix['Decil'] = pd.cut(
            x= -df_mix['y_score'],
            labels = False,
            duplicates= 'drop',
            bins= 10
        ) + 1
    df_mix['saldo_malo'] = df_mix['saldo'] * df_mix['y_true']
    df_mix['saldo_bueno'] = df_mix['saldo'] * (1 - df_mix['y_true'])

    table = df_mix.groupby('Decil').agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Saldo = ('saldo', 'sum'),
        Saldo_Bueno = ('saldo_bueno', 'sum'),
        Saldo_Malo = ('saldo_malo', 'sum')
    ).reset_index()
    table['Buenos'] = table['Clientes'] - table['Malos']
    table['Bad_Rate'] = table['Malos'] / table['Clientes']
    table['Malos_Acumulados_%'] = ((table['Malos'].cumsum() / table['Malos'].sum())*100).round(0).astype(int)
    table['Buenos_Acumulados_%'] = ((table['Buenos'].cumsum() / table['Buenos'].sum())*100).round(0).astype(int)
    table['KS_%'] = table['Malos_Acumulados_%'] - table['Buenos_Acumulados_%']
    table['Gain_%'] = ((table['Malos'].cumsum() / table['Malos'].sum())*100).round(0).astype(int)
    table['Lift'] = (table['Gain_%'] / (table['Decil'].astype(int) * 10)).round(1)
    table['Bad_Rate_%'] = (table['Bad_Rate'] * 100).round(1)
    table['Bad_Rate_%_Saldo'] = ((table['Saldo_Malo'] / table['Saldo'])*100).round(0).astype(int)
    table['Captura_Saldo_Malo_%'] = ((table['Saldo_Malo'].cumsum() / table['Saldo_Malo'].sum())*100).round(0).astype(int)
    table['Saldo_Acumulado_%'] = ((table['Saldo'].cumsum() / table['Saldo'].sum())*100).round(0).astype(int)
    table['Lift_Saldo_Acumulado'] = ((table['Captura_Saldo_Malo_%'] / table['Saldo_Acumulado_%'])).round(1)
    table['Saldo_Malo_Millones'] = (table['Saldo_Malo'] / 1000000).round(1)
    table['Saldo_Malo_Millones'] = table['Saldo_Malo_Millones'].apply(
        lambda x: f'S/{x}M'
    )
    table['Saldo_Millones'] = (table['Saldo'] / 1000000).round(1)
    table['Saldo_Millones'] = table['Saldo_Millones'].apply(
        lambda x: f'S/{x}M'
    )
    table['Saldo_Bueno_Millones'] = (table['Saldo_Bueno'] / 1000000).round(1)
    table['Saldo_Bueno_Millones'] = table['Saldo_Bueno_Millones'].apply(
        lambda x: f'S/{x}M'
    )
    FEATS_A_MOSTRAR = ['Decil', 'Clientes', 'Saldo_Millones', 'Saldo_Malo_Millones', 'Saldo_Bueno_Millones', 'Bad_Rate_%_Saldo', 'Captura_Saldo_Malo_%', 'Lift_Saldo_Acumulado', 'Malos', 'Bad_Rate_%', 'Malos_Acumulados_%','KS_%', 'Gain_%', 'Lift']

    return table[FEATS_A_MOSTRAR]


def seleccion_deciles_agrupamiento(y_true, y_score, method_bin= 'qcut'):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })
    if method_bin == 'qcut':
        df_mix['Decil'], bins = pd.qcut(
            x= df_mix['y_score'],
            q= 10,
            retbins= True,
            duplicates= 'drop'
        )
    elif method_bin == 'cut':
        df_mix['Decil'], bins = pd.cut(
            x= df_mix['y_score'],
            retbins= True,
            duplicates= 'drop',
            bins= 10
        )
    table = df_mix.groupby('Decil', observed= False).agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Bad_Rate = ('y_true', 'mean')
    ).reset_index()
    bad_rate_total = table['Malos'].sum() / table['Clientes'].sum()
    table = table.rename(columns= {'Bad_Rate': 'Bad_Rate_%'})
    table['Buenos'] = table['Clientes']-table['Malos']
    table['Lift_por_Banda'] = (table['Bad_Rate_%'] / bad_rate_total).round(1)
    table['Bad_Rate_%'] = (table['Bad_Rate_%']*100).round(1)
    table['KS_%'] = (((table['Buenos'].cumsum() / table['Buenos'].sum()) - (table['Malos'].cumsum() / table['Malos'].sum()))*100).round(1)
    FEATS_ELEGIDAS = ['Decil', 'Bad_Rate_%', 'Lift_por_Banda', 'KS_%']

    return table[FEATS_ELEGIDAS]

def tanteo_deciles(y_true, y_score, bins, labels):
    df_mix = pd.DataFrame({
        'y_true': y_true,
        'y_score': y_score
    })

    df_mix['Decil'] = pd.cut(
        x= df_mix['y_score'],
        bins = bins,
        labels= labels,
        duplicates= 'drop'
    )
    table = df_mix.groupby('Decil', observed= False).agg(
        Clientes = ('y_true', 'count'),
        Malos = ('y_true', 'sum'),
        Bad_Rate = ('y_true', 'mean')
    ).reset_index()
    bad_rate_total = table['Malos'].sum() / table['Clientes'].sum()
    table = table.rename(columns= {'Bad_Rate': 'Bad_Rate_%'})
    table['Buenos'] = table['Clientes']-table['Malos']
    table['Lift_por_Banda'] = (table['Bad_Rate_%'] / bad_rate_total).round(1)
    table['Bad_Rate_%'] = (table['Bad_Rate_%']*100).round(1)
    table['KS_%'] = (((table['Buenos'].cumsum() / table['Buenos'].sum()) - (table['Malos'].cumsum() / table['Malos'].sum()))*100).round(1)
    FEATS_ELEGIDAS = ['Decil', 'Bad_Rate_%', 'Lift_por_Banda', 'KS_%']

    return table[FEATS_ELEGIDAS]


def gestion_portafolio(
        y_true,
        y_score,
        saldo,
        garantia,
        categoria_cliente,
        bins,
        labels,
        tasa_activa_por_segmento: dict,
        costo_fondeo,
        reglas_categorias_cliente:dict,
        gastos_operativos,
        impuesto_renta,
        mostrar_escenarios = False
        ):
        df_mix = pd.DataFrame({
                'y_true': y_true,
                'y_score': y_score,
                'saldo': saldo,
                'garantia': garantia,
                'categoria_cliente': categoria_cliente
                })

        df_mix['Segmento'] = pd.cut(
        x= df_mix['y_score'],
        bins= bins,
        labels = labels
        )
        #df_mix['tasa_activa'] = df_mix['Segmento'].map(tasa_activa_por_segmento)
        
        df_mix['saldo_malo'] = df_mix['saldo'] * df_mix['y_true']
        df_mix['saldo_bueno'] = df_mix['saldo'] * (1 - df_mix['y_true'])
        df_mix['tasa_no_cubierta'] = df_mix['categoria_cliente'].apply(
                lambda x: reglas_categorias_cliente[x]
        )
        df_mix['tasa_cubierta'] = df_mix['tasa_no_cubierta'] / 2
        df_mix['porcion_cubierta'] = df_mix[['saldo', 'garantia']].min(axis= 1)
        df_mix['porcion_no_cubierta'] = df_mix['saldo'] - df_mix['porcion_cubierta']
        df_mix['provision_cubierta'] = df_mix['porcion_cubierta'] * df_mix['tasa_cubierta']
        df_mix['provision_no_cubierta'] = df_mix['porcion_no_cubierta'] * df_mix['tasa_no_cubierta']
        df_mix['provision_total'] = df_mix['provision_cubierta'] + df_mix['provision_no_cubierta']

        table = df_mix.groupby('Segmento', observed= True).agg(
                Clientes = ('y_true', 'count'),
                Malos = ('y_true', 'sum'),
                Saldo = ('saldo', 'sum'),
                Saldo_Bueno = ('saldo_bueno', 'sum'),
                Saldo_Malo = ('saldo_malo', 'sum'),
                Provisiones = ('provision_total' ,'sum')
        ).reset_index()

        table['Saldo_M'] = (table['Saldo'] / 1000000).round(1)
        table['Saldo_M'] = table['Saldo_M'].apply(
                lambda x: f'S/{x}M'
        )
        table['Provisiones_M'] = (table['Provisiones'] / 1000000).round(1)
        table['Provisiones_M'] = table['Provisiones_M'].apply(
                lambda x: f'S/{x}M'
        )
        table['Participacion_%'] = ((table['Saldo'] / table['Saldo'].sum())*100).round(1)
        table['Tasa_Activa_%'] =  ( (table['Segmento'].map(tasa_activa_por_segmento)).astype(float) *100).round(3)
        table['Spread_%'] = table['Tasa_Activa_%'] - (costo_fondeo * 100)
        table['Ingreso_Financiero'] = (table['Saldo'] * (table['Tasa_Activa_%'] / 100))
        table['Egreso Financiero'] = (table['Saldo'] * costo_fondeo)
        table['Margen_Financiero_Bruto'] = table['Ingreso_Financiero'] - table['Egreso Financiero']
        table['Provisiones/Margen_%'] = ((table['Provisiones'] / table['Margen_Financiero_Bruto'])*100).round(1)
        table['Gastos'] = (table['Participacion_%'] /100) * gastos_operativos
        table['Gastos_M'] = (table['Gastos'] / 1000000).round(2)
        table['Gastos_M'] = table['Gastos_M'].apply(
                lambda x: f'S/{x}M'
        )
        table['Margen_Neto_de_Riesgo'] = table['Margen_Financiero_Bruto'] - table['Provisiones']
        table['Resultado_antes_Impuestos'] = (table['Margen_Neto_de_Riesgo'] - table['Gastos'])
        table['Utilidad_Neta'] = (table['Resultado_antes_Impuestos'] * (1 - impuesto_renta))
        table['Utilidad_Neta_M'] = (table['Utilidad_Neta'] / 1000000).round(1)
        table['Utilidad_Neta_M'] = table['Utilidad_Neta_M'].apply(
                lambda x: f'S/{x}M'
        )
        table['ROA_%'] = ((table['Utilidad_Neta'] / table['Saldo'])*100).round(1)

        FEATS_A_MOSTRAR = [
                'Segmento', 'Saldo_M', 'Participacion_%', 'Tasa_Activa_%', 'Spread_%',
                'Provisiones_M', 'Provisiones/Margen_%', 'Gastos_M', 'Utilidad_Neta_M', 'ROA_%'
        ]

        # Crear la tabla de escenarios
        table_2 = table.copy()
        table_2['Escenario_de_Corte'] = table_2['Segmento'].apply(
                lambda x: f'Aprobar hasta {x}'
        )
        table_2['Saldo_Acumulado'] = table_2['Saldo'].cumsum()
        table_2['Saldo_Acumulado_M'] = (table_2['Saldo_Acumulado'] / 1000000).round(1)
        table_2['Saldo_Acumulado_M'] = table_2['Saldo_Acumulado_M'].apply(
                lambda x: f'S/{x}M'
        )
        table_2['Participacion_Acumulada_%'] = table_2['Participacion_%'].cumsum()
        table_2['Utilidad_Neta_Acumulada'] = table_2['Utilidad_Neta'].cumsum()
        table_2['Utilidad_Neta_Acumulada_M'] = (table_2['Utilidad_Neta_Acumulada'] / 1000000).round(1)
        table_2['Utilidad_Neta_Acumulada_M'] = table_2['Utilidad_Neta_Acumulada_M'].apply(
                lambda x: f'S/{x}M'
        )
        table_2['ROA_Escenario_%'] = ((table_2['Utilidad_Neta_Acumulada'] / table_2['Saldo_Acumulado'])*100).round(1)

        FEATS_A_MOSTRAR_TABLE_2 = [
                'Escenario_de_Corte', 'Saldo_Acumulado_M', 'Participacion_Acumulada_%',
                'Utilidad_Neta_Acumulada_M', 'ROA_Escenario_%'
        ]
        if mostrar_escenarios:
                return table[FEATS_A_MOSTRAR], table_2[FEATS_A_MOSTRAR_TABLE_2]
        else: 
                return table[FEATS_A_MOSTRAR]
        

