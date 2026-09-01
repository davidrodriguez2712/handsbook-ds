import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve, roc_curve, average_precision_score, brier_score_loss
import joblib
from matplotlib.ticker import PercentFormatter
import shap
PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from Feature_Engineering.utils.transformers.outliers import AutoOutlierHandler
from Feature_Engineering.utils.selection.psi import dataframe_psi
from Feature_Engineering.utils.transformers.binning import Binning


def monitor_performance_fix_baseline(pipeline_model, expected_x, actual_x, expected_y, actual_y, col_period, show_graph= False):
    expected_x = expected_x.copy()
    actual_x = actual_x.copy()
    #period_expected = sorted(expected[col_period].unique())
    period_actual = sorted(actual_x[col_period].unique())
    list_performance = []
    proba_expected = pipeline_model.predict_proba(expected_x)[:,1]
    #print(proba_expected.shape)
    #print(proba_expected)
    roc_expected = roc_auc_score(y_true= expected_y, y_score= proba_expected)
    gini_expected = 2 * roc_expected - 1
    list_performance.append({
        'Periodo': 'Expected Global',
        'AUC': round(roc_expected, 2),
        'Gini': round(gini_expected, 2)
    })
    feats_actual = [col for col in actual_x.columns.tolist() if col != col_period]
    proba_actual_global = pipeline_model.predict_proba(actual_x[feats_actual])[:,1]
    roc_actual_global = roc_auc_score(y_true= actual_y, y_score= proba_actual_global)
    gini_actual_global = 2 * roc_actual_global - 1
    list_performance.append({
        'Periodo': 'Actual Global',
        'AUC': round(roc_actual_global, 2),
        'Gini': round(gini_actual_global, 2)
    })
    for period in period_actual:
        mask = (actual_x[col_period] == period)
        #df_period = actual.loc[actual[col_period] == period]
        proba_actual = pipeline_model.predict_proba(actual_x[feats_actual][mask])[:,1]
        roc_score = roc_auc_score(y_true= actual_y[mask], y_score= proba_actual)
        gini = 2 * roc_score - 1
        list_performance.append({
            'Periodo': f'Actual: {period}',
            'AUC': round(roc_score, 2),
            'Gini': round(gini, 2)
        })

    df_monitor_performance = pd.DataFrame(list_performance)

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        ax.plot(
            df_monitor_performance['Periodo'],
            df_monitor_performance['AUC'],
            color = 'gray',
            label = 'AUC',
            marker = 'o'
        )
        ax.set_title('Monitoreo AUC', y= 1.02)
        ax.set_xlabel('Periodos', fontsize = 7)
        ax.tick_params(axis= 'x', rotation = 45)
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(
            PercentFormatter(1)
        )
        ax2 = ax.twinx()
        ax2.plot(
            df_monitor_performance['Periodo'],
            df_monitor_performance['Gini'],
            color = 'tomato',
            label = 'Gini',
            marker = 'o'
        )
        ax2.set_ylabel('Monitoreo Gini')
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(
            PercentFormatter(1)
        )
        object_plot_1, legend_1 = ax.get_legend_handles_labels()
        object_plot_2, legend_2 = ax2.get_legend_handles_labels()
        ax.legend(object_plot_1 + object_plot_2,
                   legend_1 + legend_2,
                   loc= 'lower right')
        plt.tight_layout()
        plt.close(fig)
        return df_monitor_performance, fig
    return df_monitor_performance


def monitor_missing(X: pd.DataFrame, period_name, dict_period = None):
    X = X.copy()
    feats_x = [col for col in X.columns.tolist() if col != period_name]
    agg = (
    X.groupby(period_name)[feats_x]
    .apply(lambda x: x.isna().mean())
    .reset_index()
    .melt(id_vars= period_name, var_name= 'feature', value_name= 'missing_rate')
    ).sort_values(by= period_name, ascending = True) 

    if dict_period is not None:
        agg[f'{period_name}_transform'] = agg[period_name].astype(str).str[-2:].map(dict_period) + "'" +agg[period_name].astype(str).str[2:4]

    feats = (
        agg.loc[agg['missing_rate'].gt(0), 'feature']
        .unique().tolist()
    )
    #feats = sorted(feats)
    fig, ax = plt.subplots(1, 1 , figsize = (10,5))
    for feat in feats:
        if dict_period is not None:
            eje_x = agg.loc[agg['feature'] == feat, f'{period_name}_transform']
        else:
            eje_x = agg.loc[agg['feature'] == feat, period_name]
        #print(eje_x)
        ax.plot(
            eje_x,
            agg.loc[agg['feature'] == feat, 'missing_rate'],
            marker = 'o',
            label = feat
        )
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_xlabel('Periodo')
        ax.set_ylabel('Porcentaje de Missings')
        ax.set_title('Evolutivo de Porcentaje de Missing por Feature', y = 1.02)
        ax.yaxis.set_major_formatter(
            PercentFormatter(1)
        )
        
    plt.tight_layout()
    plt.close(fig)
    return fig


def monitoring_psi_baseline_fix(expected, actual, period_col, show_graph, dict_transform = None):

    periods_actual = sorted(actual[period_col].unique())
    periodos_lista = []
    for period in periods_actual:
        lista_period_tmp = dataframe_psi(
            train_df= expected,
            test_df= actual,
            period_name= period,
            return_dict= True
        )
        periodos_lista.extend(lista_period_tmp)

    df_psi = pd.DataFrame(periodos_lista)
    feats_psi = df_psi['feature'].unique().tolist()

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (10, 7))
        #print(feats_psi)
        for feat in feats_psi:
            feat_df = df_psi.loc[df_psi['feature'] == feat]
            if dict_transform is not None:
                period_str = feat_df['period'].astype(str).str[-2:].map(dict_transform) + "'" + feat_df['period'].astype(str).str[2:4]
            else:
                period_str = feat_df['period'].astype(str)
            ax.plot(
                period_str,
                feat_df['psi'],
                label = feat,
                marker = 'o'
            )
        ax.axhline(y =0.25, linestyle = '--', color= 'tomato')
        ax.text(
            0.5, 0.10, 'Estabilidad',
            transform=ax.get_yaxis_transform(),
            ha='center',
            va='bottom'
        )   
        ax.axhline(y=0.10, linestyle='--', color = 'green')
        ax.text(
            0.5, 0.25, 'Alerta',
            transform=ax.get_yaxis_transform(),
            ha='center',
            va='bottom'
        )
        ax.set_title('Monitoreo PSI por cada Feature', y= 1.02)
        ax.set_xlabel('Periodos', fontsize = 9)
        ax.tick_params(axis= 'x', rotation = 45)
        ax.set_ylabel('PSI')
        ax.set_ylim(0, max((feat_df['psi'].max()+0.001), 0.5))
        #ax.yaxis.set_major_formatter(
        #    PercentFormatter(1)
        #)
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return df_psi, fig
        
    return df_psi


def monitoring_outliers_univariado(expected, actual, period_col, show_graph = False, dict_transform = None):
    objeto_iqr = AutoOutlierHandler(
        exclude_continuous= False,
        method= 'iqr',
        strategy='windsorize',
        return_dataframe= True
    )
    objeto_iqr.fit(X= expected)
    dict_rules = objeto_iqr.resultados_
    feats_train = [col for col in dict_rules.keys()]
    periods_actual = sorted(actual[period_col].unique())
    expected_outliers_n_map = {}
    expected_outliers_var_map = {}

    for feat in feats_train:
        min_value = dict_rules[feat]['lower_limit']
        max_value = dict_rules[feat]['upper_limit']
        outliers_iqr_expected_var = ( ( (expected[feat]< min_value) | (expected[feat]> max_value) ).sum() ) / expected[feat].count()
        outliers_iqr_expected_n = ( ( (expected[feat]< min_value) | (expected[feat]> max_value) ).sum() )
        expected_outliers_n_map[feat] = outliers_iqr_expected_n
        expected_outliers_var_map[feat] = outliers_iqr_expected_var
        
    lista_outliers = []
    for period in periods_actual:
        for feat in feats_train:
            actual_interval = actual.loc[actual[period_col] == period]
            min_value = dict_rules[feat]['lower_limit']
            max_value = dict_rules[feat]['upper_limit']
            outliers_iqr_var =  ( ((actual_interval[feat]< min_value) | (actual_interval[feat]> max_value) ).sum() ) / actual_interval[feat].count()
            outliers_iqr_n =  ( ((actual_interval[feat]< min_value) | (actual_interval[feat]> max_value) ).sum() ) 
            lista_outliers.append({
                'period': period,
                'feature': feat,
                'cantidad_outliers_iqr': outliers_iqr_n,
                'var_outliers_iqr': outliers_iqr_var
            })
    df_outliers = pd.DataFrame(lista_outliers)
    df_outliers['cantidad_out_iqr_expected'] = df_outliers['feature'].map(expected_outliers_n_map)
    df_outliers['variacion_out_iqr_expected'] = df_outliers['feature'].map(expected_outliers_var_map)
    df_outliers['variacion_iqr_final'] = (df_outliers['var_outliers_iqr'] - df_outliers['variacion_out_iqr_expected']).abs()

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (10, 7))
        
        for feat in feats_train:
            feat_df = df_outliers.loc[df_outliers['feature'] == feat]
            if dict_transform is not None:
                period_str = feat_df['period'].astype(str).str[-2:].map(dict_transform) + "'" + feat_df['period'].astype(str).str[2:4]
            else:
                period_str = feat_df['period'].astype(str)
            ax.plot(
                period_str,
                feat_df['variacion_iqr_final'],
                label = feat,
                marker = 'o'
            )
        ax.set_title('Monitoreo Outliers IQR por cada Feature', y= 1.02)
        ax.set_xlabel('Periodos', fontsize = 9)
        ax.tick_params(axis= 'x', rotation = 45)
        ax.set_ylabel('Variación % Outliers IQR')
        ax.set_ylim(0, (df_outliers['variacion_iqr_final'].max()+0.03))
        #print(feat_df['variacion_iqr'].max())
        ax.yaxis.set_major_formatter(
            PercentFormatter(1)
        )
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return df_outliers, fig

    return df_outliers


    
def monitor_iv_baseline_fixed(expected, actual, y_actual, period_col, show_graph = False, dict_transform = None):
    objeto_binning = Binning(
        strategy= 'frecuency',
        show_dataframe= True
    )
    objeto_binning.fit(X= expected)
    dict_rules = objeto_binning.feat_bins
    feats_train = [col for col in objeto_binning.feat_bins.keys()]
    period_list = sorted(actual[period_col].unique())
    iv_list = []
    for period in period_list:
        mask = actual[period_col] == period
        df_iv_tmp = objeto_binning.transform(
            X= actual[mask],
            y= y_actual[mask],
            operation= 'iv'
        )
        df_iv_tmp['period'] = period
        iv_list.append(df_iv_tmp)
        #print(df_iv_tmp)
        #print('==========')
    df_final = pd.concat(iv_list, axis= 0)

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (10, 7))
        
        for feat in feats_train:
            feat_df = df_final.loc[df_final['feature'] == feat]
            if dict_transform is not None:
                period_str = feat_df['period'].astype(str).str[-2:].map(dict_transform) + "'" + feat_df['period'].astype(str).str[2:4]
            else:
                period_str = feat_df['period'].astype(str)
            ax.plot(
                period_str,
                feat_df['iv'],
                label = feat,
                marker = 'o'
            )
        ax.set_title('Monitoreo PSI Evolutivo cada Feature', y= 1.02)
        ax.set_xlabel('Periodos', fontsize = 9)
        ax.tick_params(axis= 'x', rotation = 45)
        ax.set_ylabel('Information Value (IV)')
        ax.set_ylim(0, (df_final['iv'].max()+0.3))
        #print(feat_df['variacion_iqr'].max())
        ax.axhline(y= 0.1, linestyle = '--', color = 'tomato')
        ax.text(
            0.5, 0.10, 'PSI 0.10',
            transform=ax.get_yaxis_transform(),
            ha='center',
            va='bottom'
        )   
        ax.axhline(y = 0.3, linestyle = '--', color = 'green')
        ax.text(
            0.5, 0.30, 'PSI 0.30',
            transform=ax.get_yaxis_transform(),
            ha='center',
            va='bottom'
        )   
        #ax.yaxis.set_major_formatter(
        #    PercentFormatter(1)
        #)
        ax.legend(loc = 'upper left')
        plt.tight_layout()
        plt.close(fig)
        return df_final, fig

    return df_final



def threshold_f1(precision, recall):
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def curves_roc_pr(y_true_expected, y_score_expected, y_true_actual, y_score_actual):
    precision_exp, recall_exp, thresholds_pr_exp = precision_recall_curve(y_true_expected, y_score_expected)
    fpr_exp, tpr_exp, thresholds_roc_exp = roc_curve(y_true_expected, y_score_expected)
    roc_auc_exp = roc_auc_score(y_true= y_true_expected, y_score= y_score_expected)
    gini_exp = 2 * roc_auc_exp - 1
    pr_auc_exp = average_precision_score(y_true= y_true_expected, y_score = y_score_expected)
    t_exp = np.argmax(threshold_f1(precision_exp, recall_exp))

    precision_actual, recall_actual, thresholds_pr_actual = precision_recall_curve(y_true_actual, y_score_actual)
    fpr_actual, tpr_actual, thresholds_roc_actual = roc_curve(y_true_actual, y_score_actual)
    roc_auc_actual = roc_auc_score(y_true= y_true_actual, y_score= y_score_actual)
    gini_actual = 2 * roc_auc_actual - 1
    pr_auc_actual = average_precision_score(y_true= y_true_actual, y_score = y_score_actual)
    t_actual = np.argmax(threshold_f1(precision_actual, recall_actual))

    var_exp_auc = round((((roc_auc_actual - roc_auc_exp) / roc_auc_exp)*100), 1)
    var_exp_gini = round((((gini_actual - gini_exp) / gini_exp)*100), 1)
    
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].plot(
        fpr_exp,
        tpr_exp,
        label = 'Baseline'
    )
    ax[0].plot(
        fpr_actual,
        tpr_actual,
        label = 'Test/OOT'
    )

    ax[0].set_title('Curva ROC')
    ax[0].set_ylabel('TPR')
    ax[0].set_xlabel('FPR')
    ax[0].plot([0,1],[0,1], linestyle='--', color= 'tomato')
    ax[0].text(
        0.05, 0.98,
        f'Δ ROC AUC: {var_exp_auc}% | Δ Gini: {var_exp_gini}%',
        transform = ax[0].transAxes,
        ha = 'left',
        va = 'top',
        fontsize = 9
        #fontweight = 'bold'
    )
    ax[0].legend()

    ax[1].plot(
        recall_exp,
        precision_exp,
        label = 'Baseline'
    )
    ax[1].plot(
        recall_actual,
        precision_actual,
        label = 'Test/OOT'
    )
    ax[1].set_title('Curva PR')
    ax[1].set_ylabel('Precision')
    ax[1].set_xlabel('Recall')
    ax[1].axhline(precision_actual[t_actual], color = 'red', linestyle = '--', alpha = 0.5)
    ax[1].axvline(recall_actual[t_actual], color = 'red', linestyle = '--', alpha = 0.5)
    ax[1].text(
        0.10, 0.98,
        f'PR AUC: {round(pr_auc_actual, 3)} | Best F1 Threshold: {round(thresholds_pr_actual[t_actual], 3)}',
        transform = ax[1].transAxes,
        ha = 'left',
        va = 'top',
        fontsize = 9
        #fontweight = 'bold'
    )
    ax[1].legend()
    plt.close(fig)
    
    return fig


def monitor_bad_rate_fix_baseline(expected_y, expected_pd, actual_y, actual_pd, period_serie, show_graph= False):
    expected_y = expected_y.copy()
    actual_y = actual_y.copy()
    period_serie = period_serie.copy()

    brier_score_global = brier_score_loss(
        y_true= actual_y,
        y_proba = actual_pd
    )

    df_actual = pd.DataFrame({
        'period_actual': period_serie,
        'bad_rate': actual_y,
        'pd': actual_pd
    })
    #print(df_actual['period_actual'].dtype)
    period_list = sorted(df_actual['period_actual'].unique())
    list_actual_metrics = []

    agg = df_actual.groupby(
            'period_actual'
    ).agg(
        bad_rate = ('bad_rate', 'mean'),
        pd = ('pd', 'mean')
    ).reset_index()


    expected_target_mean = expected_y.mean()
    expected_pd_mean = expected_pd.mean()

    df_expected = pd.DataFrame(
        [{
            'period_actual': 'Baseline',
            'bad_rate': expected_target_mean,
            'pd': expected_pd_mean
        }]
    )

    df_final = pd.concat([df_expected, agg], axis = 0)
    
    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        ax.plot(
            df_final['period_actual'].astype(str),
            df_final['bad_rate'],
            color = 'blue',
            label = 'Bad Rate',
            marker = 'o'
        )
        ax.plot(
            df_final['period_actual'].astype(str),
            df_final['pd'],
            color = 'tomato',
            label = 'Probabilidad Default',
            marker = 'o'
        )
        ax.set_title(f'Monitoreo BR vs PD promedio (Brier Score Global: {brier_score_global:.4})', y= 1.02)
        ax.set_xlabel('Periodos', fontsize = 9)
        ax.tick_params(axis= 'x', rotation = 45)
        ax.set_ylabel('Ratio %')
        ax.set_ylim(0, max(df_final['bad_rate'].max() + 0.03, df_final['pd'].max() + 0.03))
        ax.yaxis.set_major_formatter(
            PercentFormatter(1)
        )
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return df_final, fig
    return df_final


def shap_variacion_global(model, expected: pd.DataFrame, actual: pd.DataFrame, n_sample = 5000, show_graph= False, random_state = 42):
    background = expected.sample(
            n= min(n_sample, expected.shape[0]), random_state= random_state
        )
    explainer = shap.TreeExplainer(
        data= background,
        model= model,
        feature_perturbation= 'interventional',
        model_output= 'probability'
    )
    expected_sample_df = expected.sample(
        n= min(n_sample, expected.shape[0]), random_state= random_state
    )
    actual_sample_df = actual.sample(
        n= min(n_sample, expected.shape[0]), random_state= random_state
    )
    shap_values_expected = explainer(X= expected_sample_df)
    shap_values_actual = explainer(X= actual_sample_df)

    shap_matrix_actual = pd.DataFrame(
        shap_values_actual.values,
        columns= actual_sample_df.columns,
        index= actual_sample_df.index
    )

    shap_matrix_expected = pd.DataFrame(
        shap_values_expected.values,
        columns= expected_sample_df.columns,
        index= expected_sample_df.index
    )

    contributions_list_actual = []
    contributions_list_expected = []

    for feat in shap_matrix_actual.columns:
        contribution = np.abs(shap_matrix_actual[feat]).mean()
        contributions_list_actual.append({
            'feature': feat,
            'shap_value_actual': contribution
        })
    df_contributions_actual = pd.DataFrame(contributions_list_actual)

    for feat in shap_matrix_expected.columns:
        contribution = np.abs(shap_matrix_expected[feat]).mean()
        contributions_list_expected.append({
            'feature': feat,
            'shap_value_expected': contribution
        })
    df_contributions_expected = pd.DataFrame(contributions_list_expected)

    df_contribution_final = df_contributions_expected.merge(
        df_contributions_actual,
        how= 'left',
        on= 'feature'
    )
    df_contribution_final['diferencia_contribucion'] = df_contribution_final['shap_value_actual'] - df_contribution_final['shap_value_expected']
    df_contribution_final['diferencia_contribucion_abs'] = (df_contribution_final['diferencia_contribucion']).abs()
    df_contribution_final['variacion_contribucion'] = (df_contribution_final['diferencia_contribucion']) / df_contribution_final['shap_value_expected']
    df_contribution_final = df_contribution_final.sort_values(by= 'diferencia_contribucion_abs', ascending = True)
    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        ax.barh(
            df_contribution_final['feature'],
            df_contribution_final['variacion_contribucion'],
            color = 'skyblue'
        )
        ax.set_title('Variación de Contribución SHAP - Expected vs Actual')
        ax.set_ylabel('Features')
        ax.set_xlabel('Variación % Contribución SHAP')
        ax.set_xlim(min(-0.2, df_contribution_final['variacion_contribucion'].min() - 0.02), df_contribution_final['variacion_contribucion'].max()+ 0.02)
        ax.xaxis.set_major_formatter(
            PercentFormatter(xmax= 1)
        )
        #ax.legend()
        plt.close(fig)
        return df_contribution_final, fig
    return df_contribution_final


def shap_variacion_global_periodica(model, expected: pd.DataFrame, actual: pd.DataFrame, period_col, n_sample = 5000, show_graph= False, random_state = 42):
    background = expected.sample(
            n= min(n_sample, expected.shape[0]), random_state= random_state
        )
    explainer = shap.TreeExplainer(
        data= background,
        model= model,
        feature_perturbation= 'interventional',
        model_output= 'probability'
    )
    shap_list = []
    expected_sample_df = expected.sample(
        n= min(n_sample, expected.shape[0]), random_state= random_state
    )
    df_expected_shap = pd.DataFrame(
        explainer(X= expected_sample_df).values,
        columns= expected_sample_df.columns,
        index= expected_sample_df.index
    )

    df_agg_expected = df_expected_shap.mean()

    for feat, shap_value in df_agg_expected.items():
        shap_list.append({
            'feature': feat,
            'shap_value': shap_value,
            'period': 'Baseline'
        })

    periods_actual = sorted(actual[period_col].unique())
    feats_actual = actual.columns.tolist()
    feats_actual = [feat for feat in feats_actual if feat != period_col]

    for period in periods_actual:
        df_tmp = actual.loc[actual[period_col] == period]
        tmp_sample_df = df_tmp.sample(
            n= min(400, df_tmp.shape[0]), random_state= random_state
        )

        df_tmp_final = pd.DataFrame(
            explainer(X= tmp_sample_df[feats_actual]).values,
            columns= tmp_sample_df[feats_actual].columns,
            index= tmp_sample_df[feats_actual].index
        )

        df_agg = df_tmp_final.mean()
        for feat, shap_value in df_agg.items():
            shap_list.append({
                'feature': feat,
                'shap_value': shap_value,
                'period': period
            })
    
    df_final_agg = pd.DataFrame(shap_list)
    #print(df_final_agg)
    #print(type(df_final_agg))
    feats_final = df_final_agg['feature'].unique()
    #print(type(feats_final))

    if show_graph:
        fig, ax = plt.subplots(1, 1, figsize = (11, 6))
        for feat in feats_final:
            df_mask = df_final_agg.loc[df_final_agg['feature'] == feat]
            ax.plot(
                (df_mask['period']).astype(str),
                df_mask['shap_value'],
                label = feat
            )
        ax.set_title('Evolutivo SHAP - Test/OOT vs Baseline')
        ax.set_ylabel('SHAP Value')
        ax.set_xlabel('Periodos')
        ax.set_ylim(min(0, df_final_agg['shap_value'].min() - 0.02), df_final_agg['shap_value'].max()+ 0.02)
        ax.tick_params(axis= 'x', rotation = 45)
        #ax.xaxis.set_major_formatter(
        #    PercentFormatter(xmax= 1)
        #)
        ax.legend(
            loc = 'upper left',
            bbox_to_anchor= (1.02, 1),
            fontsize = 8.5
        )
        plt.close(fig)
        return df_final_agg, fig
    return df_final_agg









