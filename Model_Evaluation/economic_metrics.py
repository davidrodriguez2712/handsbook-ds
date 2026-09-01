import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve, roc_curve, average_precision_score
import joblib
PARENT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PARENT_DIR))

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
        

