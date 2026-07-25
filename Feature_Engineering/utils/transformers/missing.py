import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

class AutoMissingHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy_numeric, strategy_categorical, fill_value, missing_indicator, return_dataframe):
        self.strategy_numeric = strategy_numeric
        self.strategy_categorical = strategy_categorical
        self.fill_value = fill_value
        self.missing_indicator = missing_indicator
        self.return_dataframe = return_dataframe

    def fit(self, X: pd.DataFrame, y=None):
        ## Aprender de los datos 
        X = X.copy()
        self.features_names_in_ = X.columns.tolist()
        self.numeric_feats = X.select_dtypes(include= 'number').columns.tolist()
        self.categorical_feats = X.select_dtypes(exclude= 'number').columns.tolist()
        
        self.categorical_imputer = SimpleImputer(
            strategy= self.strategy_categorical,
            fill_value= self.fill_value
        )
        if self.strategy_numeric == 'knn':
            self.numeric_imputer = (
                Pipeline(
                    steps= [
                        ('standar_scaler', StandardScaler()),
                        ('knn', KNNImputer(
                            n_neighbors= 5,
                            weights= 'uniform'
                        ))
                    ]
                )
            )
        else:
            self.numeric_imputer = SimpleImputer(
                strategy= self.strategy_numeric
            )

        self.missing_cols_ = X.columns[X.isna().any(axis= 0)].tolist()

        if self.numeric_feats:
            self.numeric_imputer.fit(X[self.numeric_feats])
        if self.categorical_feats:
            self.categorical_imputer.fit(X[self.categorical_feats])

        return self

    def transform(self, X: pd.DataFrame):
        ## Aplicar las reglas del fit o que no requiera aprender del train
        X = X.copy()

        X_num = pd.DataFrame(index= X.index)
        X_cat = pd.DataFrame(index= X.index)

        if self.numeric_feats:
            X_num = pd.DataFrame(
                self.numeric_imputer.transform(X[self.numeric_feats]),
                columns= self.numeric_feats,
                index= X.index
            )
            
        if self.categorical_feats:
            X_cat = pd.DataFrame(
                self.categorical_imputer.transform(X[self.categorical_feats]),
                columns= self.categorical_feats,
                index= X.index
            )
        
        X_out = pd.concat([X_num, X_cat], axis= 1)
            
        # Añadir el missing indicator
        if self.missing_indicator:
            for col in self.missing_cols_:
                X_out[f'{col}_missing_indicator'] = X[col].isna().astype(int)

        X_out = X_out[
            [col for col in self.features_names_in_ if col in X_out.columns] +
            [col for col in X_out.columns.tolist() if col.endswith('_missing_indicator')]
        ]
        
        if self.return_dataframe:
            return X_out
        
        return X_out.values

class AutoInteractiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, algorithm, max_iter, random_state, initial_strategy: str, return_dataframe):
        self.algorithm = algorithm
        self.initial_strategy = initial_strategy
        self.max_iter = max_iter
        self.random_state = random_state
        self.return_dataframe = return_dataframe

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.feat_numeric = X.select_dtypes(include= 'number').columns.tolist()

        self.iterative_imputer = IterativeImputer(
            max_iter= self.max_iter,
            random_state=self.random_state,
            estimator= self.algorithm,
            skip_complete= True
        )

        if self.feat_numeric:
            self.iterative_imputer.fit(X[self.feat_numeric])

        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X_num = pd.DataFrame(index= X.index)

        X_num = pd.DataFrame(
            self.iterative_imputer.transform(X[self.feat_numeric]),
            columns= self.feat_numeric,
            index= X.index
        )

        if self.return_dataframe:
            return X_num
        
        return X_num.values
        





































