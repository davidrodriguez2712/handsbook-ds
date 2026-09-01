import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class WOEClassic(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, return_dataframe):
        self.strategy = strategy
        self.return_dataframe = return_dataframe

    def _pre_bins_num(self, rules: list):
        rules[0] = -np.inf
        rules[-1] = np.inf
        return rules

    def _roc_auc(self, y_true, y_pred):
        # Calcular el índice de Gini
        auc_roc = roc_auc_score(y_true, y_pred)
        #gini = 2 * auc_roc - 1
        return auc_roc
    
    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        X_y = pd.concat([X, y], axis= 1)
        TARGET = y.name
        inv_features = inventario_features(X)
        self.FEATS_NUM = X.select_dtypes(include= 'number')
        self.FEATS_NUM_CONTEO = inv_features.loc[inv_features['semantic_dtype'] == 'Conteo']['feature'].values.tolist()
        self.FEATS_NUM_CONTEO = [col for col in self.FEATS_NUM_CONTEO if col in self.FEATS_NUM]
        self.FEATS_NUM_CONTINUAS = [col for col in self.FEATS_NUM if col not in self.FEATS_NUM_CONTEO]
        self.FEATS_CATEGORICALS = X.select_dtypes(include= ['category', 'string', 'object']).columns.tolist()

        feats_num = {}
        rules_num = {}
        self.feats_rules = {}
        feats_cat = {}
        rules_cat = {}

        eps = 1e-5

        if self.FEATS_NUM_CONTEO:
            for feat in X[self.FEATS_NUM_CONTEO]:
                feats_num[f'{feat}_binning'], rules_num[f'{feat}_rules'] = (
                    pd.cut(
                        X[feat],
                        bins= len(X[feat].unique()),
                        retbins= True
                    )
                )

                table = (
                  X.groupby(feats_num[f'{feat}_binning'])[TARGET].
                  agg(
                      ('total_obs', 'count'),
                      ('total_event', 'sum'),
                      ('event_rate', 'mean')
                  )
                )
                table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()
                table["total_no_event"] = table["total_obs"] - table["total_event"]
                table["dist_eventos"] = table["total_event"] / table["total_event"].sum()
                table["dist_no_eventos"] = table["total_no_event"] / table["total_no_event"].sum()
                table["woe"] = (
                    np.log( (table["dist_no_eventos"] + eps) / (table["dist_eventos"] + eps) )
                )
                table = table.reset_index()

                woe_values = table['woe'].values
                rules = self._pre_bins_num(rules_num[f'{feat}_rules'])

                self.feats_rules[feat] = {
                    'edges': rules,
                    'woe': woe_values
                }

        if self.FEATS_NUM_CONTINUAS:
            for feat in X[self.FEATS_NUM_CONTEO]:
                feats_num[f'{feat}_binning'], rules_num[f'{feat}_rules'] = (
                    pd.cut(
                        X[feat],
                        bins= 20,
                        retbins= True
                    )
                )

                table = (
                  X.groupby(feats_num[f'{feat}_binning'])[TARGET].
                  agg(
                      ('total_obs', 'count'),
                      ('total_event', 'sum'),
                      ('event_rate', 'mean')
                  )
                )
                table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()
                table["total_no_event"] = table["total_obs"] - table["total_event"]
                table["dist_eventos"] = table["total_event"] / table["total_event"].sum()
                table["dist_no_eventos"] = table["total_no_event"] / table["total_no_event"].sum()
                table["woe"] = (
                    np.log( (table["dist_no_eventos"] + eps) / (table["dist_eventos"] + eps) )
                )
                table = table.reset_index()

                woe_values = table['woe'].values
                rules = self._pre_bins_num(rules_num[f'{feat}_rules'])

                self.feats_rules[feat] = {
                    'edges': rules,
                    'woe': woe_values
                }

        if self.FEATS_CATEGORICALS:
            for feat in self.FEATS_CATEGORICALS:
                table = (
                    X.groupby(feat)[TARGET].
                    agg(
                        ('total_obs', 'count'),
                        ('total_event', 'sum'),
                        ('event_rate', 'mean')
                    )
                    )
                table["pct_dist"] = table["total_obs"] / table["total_obs"].sum()
                table["total_no_event"] = table["total_obs"] - table["total_event"]
                table["dist_eventos"] = table["total_event"] / table["total_event"].sum()
                table["dist_no_eventos"] = table["total_no_event"] / table["total_no_event"].sum()
                table["woe"] = (
                    np.log( (table["dist_no_eventos"] + eps) / (table["dist_eventos"] + eps) )
                )
                table = table.reset_index()

                woe_values = table['woe'].values.tolist()
                category = table[feat].values.tolist()

                mapping = {
                    categoria: woe for (categoria, woe) in list(zip(category, woe_values))
                }
                mapping['OTROS'] = 0

                self.feats_rules[feat] = {
                    mapping
                }

        return self


    def transform(self, X: pd.DataFrame):
        X = X.copy()
        list_series = []
        for feat in self.FEATS_NUM:
            edges = self.feats_rules[feat]['edges']
            woe = self.feats_rules[feat]['woe']

            idx = np.searchsorted(edges, X[feat], side= 'right') - 1
            list_series.append(pd.Series(
                woe[idx],
                index= X.index,
                name= f'{feat}_woe'
            ))
        
        for feat in self.FEATS_CATEGORICALS:
            rules = self.feats_rules[feat]
            list_series.append(pd.Series(
                X[feat].map(rules),
                index= X.index,
                name= f'{feat}_woe'
            ))

        df_woe = pd.concat(list_series, axis= 1)

        if self.return_dataframe:
            return df_woe
        
        return df_woe.values


    def binning_explore(self, X: pd.DataFrame, y: pd.Series):
        pass

class Logit_Smoothing_Rolling(BaseEstimator, TransformerMixin):
    def __init__(self, return_dataframe, show_details = False):
        self.return_dataframe = return_dataframe
        self.show_details = show_details

    def _logarithmic_function(self, x, a ,b):
        return a + b * x

    def _exponential_function(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def _roc_auc(self, y_true, y_pred):
        # Calcular el índice de ROC AUC
        mask = ~np.isnan(y_pred)
        auc_roc = roc_auc_score(y_true[mask], y_pred[mask])
        #gini = 2 * auc_roc - 1
        return auc_roc

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Paso 1: Ordenar la variable (dropeando los null)
        X = X.copy()
        y = y.copy()
        TARGET = y.name

        inv_features = inventario_features(X)
        condition_1 = (inv_features['semantic_dtype'] == 'Continua')
        condition_2 = (inv_features['semantic_dtype'] == 'Conteo')
        #condition_3 = (inv_features['unique_values'] > 5)
        self.feats_numeric_ = inv_features.loc[(condition_1) | (condition_2)]['feature'].values.tolist()
        EPS = 1e-6
        dict_features = {}

        ## Spliteo para crear validación
        X, X_val, y, y_val = train_test_split(
            X,
            y,
            test_size = 0.3,
            random_state= 42,
            stratify= y
        )
        
        for feat in self.feats_numeric_:
            tmp = pd.DataFrame({
                feat: X[feat],
                'target': y
            }).dropna(subset= [feat, 'target']).sort_values(by= feat, ascending= True)

            rolling_mean = (
                tmp['target']
                .rolling(window = 500, min_periods = 25)
                .mean()
            )

            ## adicional para sacar el promedio de esas ventanas
            rolling_feature_mean = (
                tmp[feat]
                .rolling(window = 500, min_periods = 25)
                .mean()
            )

            rolling_mean_safe = rolling_mean.clip(
                lower= EPS,
                upper= 1 - EPS
            )

            dict_features[f'{feat}_log_odds'] = np.log(
                rolling_mean_safe / (1 - rolling_mean_safe)
            ).replace([-np.inf, np.inf], np.nan)

            dict_features[f'{feat}_mean'] = rolling_feature_mean

        df_log_odds = pd.DataFrame(dict_features)
        X = X.merge(
            df_log_odds,
            left_index= True,
            right_index= True,
            how= 'left'
        )

        # Paso 2: Crear para los NA su log odds. 
        X_temp = X.copy()
        X_temp[TARGET] = y
        self.dict_log_odds_null_ = {}
        col_log_odds = [c for c in X_temp.columns.tolist() if c.endswith('_log_odds')]
        col_feats = [c.replace('_log_odds', '') for c in col_log_odds]
        for feat in col_feats:

            if X_temp[feat].isnull().any():
            
                mean_null = (
                    X_temp.loc[X_temp[feat].isnull(), TARGET].mean()
                )

                mean_null_safe = np.clip(
                    mean_null,
                    EPS,
                    1 - EPS
                )

                log_odds_null = np.log(
                    mean_null_safe / (1 - mean_null_safe)
                )

                X_temp[feat] = X_temp[feat].fillna(
                    log_odds_null
                )
                self.dict_log_odds_null_[feat] = log_odds_null

            else:
                global_target_mean = (
                    X_temp[TARGET].mean()
                )

                safe_target_mean = np.clip(
                    global_target_mean,
                    EPS,
                    1 - EPS
                )

                global_log_odds = np.log(
                    safe_target_mean / (1 - safe_target_mean)
                )

                self.dict_log_odds_null_[feat] = global_log_odds


        # Paso 3: Crear los scatter (log odds vs feature)
        # Hacerlo para todas las variables

        columnas = [
            feat for feat in X_temp.columns
            if '_log_odds' not in feat and
            feat != TARGET and
            '_mean' not in feat
        ]

        n_rows = len(columnas)
        fig, ax = plt.subplots(nrows= n_rows,ncols=1, figsize = (10,4.5 * n_rows))
        for i, feat in enumerate(columnas):
            data_filtered = X_temp.dropna(subset= [feat, f'{feat}_log_odds'])
            x = data_filtered[f'{feat}_mean']
            #print(x[:50])
            y = data_filtered[TARGET].values
            x_mean = x.mean()
            #print(f'X_mean: {x_mean}')
            x_std = x.std()

            z = ((x - x_mean) / x_std)

            sc = ax[i].scatter(
                z,
                data_filtered[f'{feat}_log_odds'],
                s = 3,
                c = data_filtered[TARGET],
                cmap = 'viridis',
                marker= 'o'
            )
            ax[i].set_xlabel(f'{feat} (z-score)')
            ax[i].set_ylabel(f'{feat}_log_odds')
            fig.colorbar(sc, ax= ax[i])

            percentiles = np.percentile(z, [2.5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 98])

            for percentile in percentiles:
                ax[i].axvline(x = percentile, color = 'gray', linestyle = '--')
        plt.tight_layout()
        plt.close(fig)

        if self.show_details:
            display(fig)

        ## Paso 4: Entrenar funciones candidatas para cada variable
        col_features_log_odss = [c for c in X_temp.columns.tolist() if c.endswith('_log_odds')]
        col_features = [c.replace('_log_odds', '') for c in col_features_log_odss]
        col_features_mean = [c.replace('_log_odds', '_mean') for c in col_features_log_odss]
        self.transformations_ = {}
        self.rules_ = []
        self.functions_details_ = {}

        for feat, feat_log_odds, feat_mean in zip(col_features, col_features_log_odss, col_features_mean):

            dict_var = {}
            d = X_temp.dropna(subset= [feat_log_odds, feat_mean]).copy()
            x = d[feat_mean].values
            y = d[feat_log_odds].values
            x_mean = x.mean()
            x_std = x.std()

            z = ((x - x_mean) / x_std)

            X_val[feat] = X_val[feat].fillna(self.dict_log_odds_null_[feat])

            # Función polinómica de grado 1
            coefficients_pol1 = np.polyfit(x= z, y = y, deg = 1)
            dict_var['polynomial_1'] = {
                'type': 'polynomial_1',
                'degree': 1,
                'coefficients': coefficients_pol1,
                'r2': r2_score(y, np.polyval(coefficients_pol1, z)),
                'x_mean': x_mean,
                'x_std': x_std
                #'predict': lambda x_new, x_mean= x_mean, x_std = x_std, coefficients_pol1 = coefficients_pol1: 
                #np.polyval(coefficients_pol1, (x_new - x_mean) / x_std)
            }
            
            pred_val_pol1 = np.polyval(coefficients_pol1, X_val[feat])
            pol1_auc = self._roc_auc(y_val, pred_val_pol1)

            pred_train_pol1 = np.polyval(coefficients_pol1, d[feat])
            pol1_auc_train = self._roc_auc(d[TARGET], pred_train_pol1)

            # Función polinómica de grado 2
            coefficients_pol2 = np.polyfit(x= z, y = y, deg = 2)
            dict_var['polynomial_2'] = {
                'type': 'polynomial_2',
                'degree': 2,
                'coefficients': coefficients_pol2,
                'r2': r2_score(y, np.polyval(coefficients_pol2, z)),
                'x_mean': x_mean,
                'x_std': x_std
                #'predict': lambda x_new, x_mean= x_mean, x_std = x_std, coefficients_pol2 = coefficients_pol2:
                #np.polyval(coefficients_pol2, (x_new - x_mean) / x_std)
            }
            pred_val_pol2 = np.polyval(coefficients_pol2, X_val[feat])
            pol2_auc = self._roc_auc(y_val, pred_val_pol2)

            pred_train_pol2 = np.polyval(coefficients_pol2, d[feat])
            pol2_auc_train = self._roc_auc(d[TARGET], pred_train_pol2)

            # Función polinómica de grado 3
            coefficients_pol3 = np.polyfit(x= z, y = y, deg = 3)
            dict_var['polynomial_3'] = {
                'type': 'polynomial_3',
                'degree': 3,
                'coefficients': coefficients_pol3,
                'r2': r2_score(y, np.polyval(coefficients_pol3, z)),
                'x_mean': x_mean,
                'x_std': x_std
                #'predict': lambda x_new, x_mean = x_mean, x_std = x_std, coefficients_pol3 = coefficients_pol3:
                #np.polyval(coefficients_pol3, (x_new - x_mean) / x_std)

            }
            pred_val_pol3 = np.polyval(coefficients_pol3, X_val[feat])
            pol3_auc = self._roc_auc(y_val, pred_val_pol3)

            pred_train_pol3 = np.polyval(coefficients_pol3, d[feat])
            pol3_auc_train = self._roc_auc(d[TARGET], pred_train_pol3)

            ## Función logarítmica
            mask = X_temp[feat] != -1
            #mask = X_temp[feat].notna()
            X_temp_filtered = X_temp.loc[mask, [feat, feat_mean, feat_log_odds, TARGET]]
            X_temp_filtered = X_temp_filtered.dropna()
            x_trans_log = np.log1p(X_temp_filtered[feat])

            x_log_mean = x_trans_log.mean()
            x_log_std = x_trans_log.std()

            z = (x_trans_log - x_log_mean) / x_log_std

            param_log, paramcov_log = curve_fit(
                f= self._logarithmic_function,
                xdata= z,
                ydata = X_temp_filtered[feat_log_odds],
                maxfev= 10000
            )

            dict_var['logarithmic'] = {
                'type': 'logarithmic',
                'coefficients': param_log,
                'r2': r2_score(X_temp_filtered[feat_log_odds], self._logarithmic_function(z, *param_log)),
                'x_mean': x_log_mean,
                'x_std': x_log_std
                #'predict': lambda x_new, x_log_mean = x_log_mean, x_log_std = x_log_std, param_log = param_log:
                #(self._logarithmic_function( ( np.log1p(x_new) - x_log_mean) / x_log_std , *param_log ))
            }

            pred_val_log = self._logarithmic_function( (np.log1p(X_val[feat]) - x_log_mean) / x_log_std, *param_log)
            log_auc = self._roc_auc(y_val, pred_val_log)

            pred_train_log = self._logarithmic_function( (np.log1p(d[feat]) - x_log_mean) / x_log_std, *param_log)
            log_auc_train = self._roc_auc(X_temp_filtered[TARGET], pred_train_log)
            
            # Función exponencial
            # param_exp, paramcov_exp = curve_fit(
            #     f = self._exponential_function,
            #     xdata = z,
            #     ydata = y,
            #     maxfev= 10000
            # )

            # dict_var['exponential'] = {
            #     'type': 'exponential',
            #     'parameters': param_exp,
            #     'r2': r2_score(y, self._exponential_function(z, *param_exp)),
            #     'predict': lambda x_new, x_mean = x_mean, x_std = x_std, param_exp = param_exp:
            #     self._exponential_function((x_new - x_mean) / x_std, *param_exp)
            # }
            # pred_val_exp = dict_var['exponential']['predict'](X_val[feat])
            # # validación
            
            # exp_auc = self._roc_auc(y_val, pred_val_exp)

            # pred_train_exp = dict_var['exponential']['predict'](d[feat])
            # exp_auc_train = self._roc_auc(d[TARGET], pred_train_exp)

            self.transformations_[feat] = dict_var

            summary = pd.DataFrame({
                'feature': feat,
                'function': [dict_var['polynomial_1']['type'], dict_var['polynomial_2']['type'], dict_var['polynomial_3']['type'], dict_var['logarithmic']['type']],
                'auc_score_train': [pol1_auc_train, pol2_auc_train, pol3_auc_train, log_auc_train],
                'auc_score_val': [pol1_auc, pol2_auc, pol3_auc, log_auc],
                'r2_score': [dict_var['polynomial_1']['r2'], dict_var['polynomial_2']['r2'], dict_var['polynomial_3']['r2'], dict_var['logarithmic']['r2']],
                'parameters': [dict_var['polynomial_1']['coefficients'], dict_var['polynomial_2']['coefficients'], dict_var['polynomial_3']['coefficients'], dict_var['logarithmic']['coefficients']]
            }).sort_values(by= 'auc_score_val', ascending= False)

            self.functions_details_[feat] = summary

            best_function = summary.iloc[0]['function']
            best_function_dict = dict_var[best_function]
            best_auc_score_train = summary.iloc[0]['auc_score_train']
            best_auc_score_val = summary.iloc[0]['auc_score_val']
            best_r2_score = summary.iloc[0]['r2_score']
            #self.rules[feat] = best_function_dict
            self.rules_.append({
                'feat': feat,
                'function': best_function,
                'auc_score_train': best_auc_score_train,
                'auc_score_val': best_auc_score_val,
                'r2_score': best_r2_score
            })

        if self.show_details:
            df_rules = pd.DataFrame(self.rules_)
            display(df_rules)

        return self


    def transform(self, X:pd.DataFrame, y: pd.Series= None):
        X = X.copy()
        if y is not None:
            y = y.copy()
        transformed_features = {}
        comparative_list = []
        rules_dict = {c['feat']:c for c in self.rules_}
        #display(self.rules)
        #display(rules_dict)
        #display(self.dict_log_odds_null)
        auc_score_test = np.nan
        for feat in self.feats_numeric_:
            best_function = rules_dict[feat]['function']
            x_mean = self.transformations_[feat][best_function]['x_mean']
            x_std = self.transformations_[feat][best_function]['x_std']
            coeff = self.transformations_[feat][best_function]['coefficients']

            if best_function != 'logarithmic':
                X[feat] = X[feat].fillna(self.dict_log_odds_null_[feat])
                X[feat] = (X[feat] - x_mean) / x_std
                transformed_features[f'{feat}_log_odds'] = np.polyval(coeff, X[feat])
                if y is not None:
                    auc_score_test = roc_auc_score(
                        y,
                        transformed_features[f'{feat}_log_odds']
                    )
                comparative_list.append({
                    'feature': feat,
                    'auc_score_train': rules_dict[feat]['auc_score_train'],
                    'auc_score_val': rules_dict[feat]['auc_score_val'],
                    'auc_score_test': auc_score_test
                })
            elif best_function == 'logarithmic':
                X[feat] = X[feat].where( X[feat] > -1, np.nan )
                transformed_features[f'{feat}_log_odds'] =  self._logarithmic_function( (np.log1p(X[feat]) - x_mean) / x_std, *coeff ).fillna(self.dict_log_odds_null_[feat])
                if y is not None:
                    auc_score_test = roc_auc_score(
                        y,
                        transformed_features[f'{feat}_log_odds']
                    )
                comparative_list.append({
                    'feature': feat,
                    'auc_score_train': rules_dict[feat]['auc_score_train'],
                    'auc_score_val': rules_dict[feat]['auc_score_val'],
                    'auc_score_test': auc_score_test
                })
        
        self.df_comparative = pd.DataFrame(
            comparative_list
        )

        if self.show_details:
            display('==== Estas visualizando el DataFrame Comparativo =====')
            display(self.df_comparative)

        if self.return_dataframe:
            return pd.DataFrame(transformed_features, index= X.index)

        return pd.DataFrame(transformed_features, index= X.index).values
        

        

        


        












































