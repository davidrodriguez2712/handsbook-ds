import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import sys
from pathlib import Path
from matplotlib.pyplot import pyplot as plt

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features

class Binning(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, return_dataframe):
        self.strategy = strategy
        self.return_dataframe = return_dataframe

    def _pre_bins_num(self, rules: list):
        rules[0] = -np.inf
        rules[-1] = np.inf
        return rules
    
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

        
        






























