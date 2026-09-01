import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import joblib


example_dict = {
    'n_iterations': {
        'type': 'int',
        'min': 3,
        'max': 5
    },
    'min_child': {
        'type': 'float',
        'min': 4,
        'max': 3,
        'log': True
    },
    'method': {
        'type': 'categorical',
        'choices': ['opcion1', 'opcion2'] 
    }
}

class OptunaTunning:
    def __init__(self, nombre_proyecto, X, y, algoritmos: list, params: dict, preprocessor_linear, activate_preprocessor_trees = False, preprocessor_trees = None, default_params = False, random_state = 42, n_trials = 50):
        self.nombre_proyecto = nombre_proyecto
        self.X = X
        self.y = y
        self.algoritmos = algoritmos
        self.params = params
        self.default_params = default_params
        self.preprocessor_linear = preprocessor_linear
        self.activate_preprocessor_trees = activate_preprocessor_trees
        self.preprocessor_trees = preprocessor_trees
        self.random_state = random_state
        self.n_trials = n_trials

    def _fix_params(self, trial, model):

        fix_params = {}
        model_name = model.__name__
        dict_model = self.params[model_name]

        for hiper, config in dict_model.items():
            if config['type'] == 'int':
                fix_params[hiper] = trial.suggest_int(hiper, config['min'], config['max'])
            elif config['type'] == 'float':
                fix_params[hiper] = trial.suggest_float(hiper, config['min'], config['max'], log = config.get('log', False))
            elif config['type'] == 'categorical':
                fix_params[hiper] = trial.suggest_category(hiper, config['choices'])
            elif config['type'] == 'fixed':
                fix_params[hiper] = config['value']
            else:
                raise ValueError(
                    f'Tipo de parámetro no soportado: {config["type"]}'
                )
        return fix_params

    def _default_params(self, model):
        model_name = model.__name__
        params = {

            'XGBClassifier': {

                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 10
                },

                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.2,
                    'log': True
                },

                'n_estimators': {
                    'type': 'int',
                    'low': 100,
                    'high': 1000
                },

                'min_child_weight': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                },

                'subsample': {
                    'type': 'float',
                    'low': 0.6,
                    'high': 1.0
                },

                'colsample_bytree': {
                    'type': 'float',
                    'low': 0.6,
                    'high': 1.0
                },

                'gamma': {
                    'type': 'float',
                    'low': 0,
                    'high': 5
                },

                'reg_alpha': {
                    'type': 'float',
                    'low': 1e-8,
                    'high': 10,
                    'log': True
                },

                'reg_lambda': {
                    'type': 'float',
                    'low': 1e-3,
                    'high': 10,
                    'log': True
                }
            },

            'LGBMClassifier': {

                'n_estimators': {
                    'type': 'int',
                    'low': 100,
                    'high': 1000
                },

                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.2,
                    'log': True
                },

                'num_leaves': {
                    'type': 'int',
                    'low': 10,
                    'high': 100
                },

                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 15
                },

                'min_child_samples': {
                    'type': 'int',
                    'low': 10,
                    'high': 100
                },

                'subsample': {
                    'type': 'float',
                    'low': 0.6,
                    'high': 1.0
                },

                'colsample_bytree': {
                    'type': 'float',
                    'low': 0.6,
                    'high': 1.0
                },

                'reg_alpha': {
                    'type': 'float',
                    'low': 1e-8,
                    'high': 10,
                    'log': True
                },

                'reg_lambda': {
                    'type': 'float',
                    'low': 1e-3,
                    'high': 10,
                    'log': True
                }
            },

            'CatBoostClassifier': {

                'iterations': {
                    'type': 'int',
                    'low': 200,
                    'high': 1000
                },

                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.2,
                    'log': True
                },

                'depth': {
                    'type': 'int',
                    'low': 4,
                    'high': 10
                },

                'l2_leaf_reg': {
                    'type': 'float',
                    'low': 1,
                    'high': 10,
                    'log': True
                },

                'random_strength': {
                    'type': 'float',
                    'low': 1e-3,
                    'high': 10,
                    'log': True
                },

                'bagging_temperature': {
                    'type': 'float',
                    'low': 0,
                    'high': 5
                },

                'cat_features' : {
                    'type': 'fixed',
                    'value': self.feats_cat_catboost
                }
            },

            'RandomForestClassifier': {

                'n_estimators': {
                    'type': 'int',
                    'low': 100,
                    'high': 1000
                },

                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 20
                },

                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 20
                },

                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                },

                'max_features': {
                    'type': 'categorical',
                    'choices': ['sqrt', 'log2', None]
                },

                'criterion': {
                    'type': 'categorical',
                    'choices': ['gini', 'entropy']
                },

                'class_weight': {
                    'type': 'categorical',
                    'choices': [None, 'balanced', 'balanced_subsample']
                }
            },

            'LogisticRegression': {

                'C': {
                    'type': 'float',
                    'low': 1e-4,
                    'high': 100,
                    'log': True
                },

                'penalty': {
                    'type': 'categorical',
                    'choices': ['l1', 'l2']
                },

                'solver': {
                    'type': 'categorical',
                    'choices': ['liblinear']
                },

                'class_weight': {
                    'type': 'categorical',
                    'choices': [None, 'balanced']
                },

                'max_iter': {
                    'type': 'int',
                    'low': 500,
                    'high': 2000
                }
            }
        }

        try:
            params_default = params[model_name]
        except Exception as e:
            print (f'Error en la obtención de parámetros por default debido al modelo insertado: {e}') 

        return params_default
        

    def build_model(self, trial, model):
        if model.__name__ == 'LogisticRegression':
            if self.default_params:
                params_user = self._default_params(model)
            else:
                params_user = self._fix_params(trial, model)          
            pipeline = Pipeline(
                steps= [
                    ('preprocessing', self.preprocessor_linear),
                    ('model', LogisticRegression(**params_user, random_state= self.random_state))
                ]
            )
        elif model.__name__ == 'RandomForestClassifier':
            if self.default_params:
                params_user = self._default_params(model)
            else:
                params_user = self._fix_params(trial, model)    
            if self.activate_preprocessor_trees:
                pipeline = Pipeline(
                    steps= [
                        ('preprocessing', self.preprocessor_trees),
                        ('model', model(**params_user, random_state = self.random_state))
                    ]
                )
            else: 
                pipeline = model(
                    **params_user,
                    random_state = self.random_state
                )
        elif model.__name__ == 'XGBClassifier':
            if self.default_params:
                params_user = self._default_params(model)
            else:
                params_user = self._fix_params(trial, model)    
            if self.activate_preprocessor_trees:
                pipeline = Pipeline(
                    steps= [
                        ('preprocessing', self.preprocessor_trees),
                        ('model', model(**params_user, random_state = self.random_state))
                    ]
                )
            else: 
                pipeline = model(
                    **params_user,
                    random_state = self.random_state
                )
        elif model.__name__ == 'LGBMClassifier':
            if self.default_params:
                params_user = self._default_params(model)
            else:
                params_user = self._fix_params(trial, model)    
            if self.activate_preprocessor_trees:
                pipeline = Pipeline(
                    steps= [
                        ('preprocessing', self.preprocessor_trees),
                        ('model', model(**params_user, random_state = self.random_state))
                    ]
                )
            else: 
                pipeline = model(
                    **params_user,
                    random_state = self.random_state
                )
        elif model.__name__ == 'CatBoostClassifier':
            if self.default_params:
                X_temp = self.X.copy()
                self.feats_cat_catboost = X_temp.select_dtypes(include = ['object', 'string', 'category']).columns.tolist()
                params_user = self._default_params(model)
            else:
                params_user = self._fix_params(trial, model)
                
            if self.activate_preprocessor_trees:
                pipeline = Pipeline(
                    steps= [
                        ('preprocessing', self.preprocessor_trees),
                        ('model', model(**params_user, random_state = self.random_state))
                    ]
                )
            else: 
                pipeline = model(
                    **params_user,
                    random_state = self.random_state
                )
        return pipeline

    def objective(self, trial, model):
        X = self.X.copy()
        y = self.y.copy()
        pipeline = self.build_model(trial, model)

        cv_strategy = StratifiedKFold(n_splits= 5, shuffle= True, random_state= self.random_state)

        cv_auc = cross_val_score(
            estimator= pipeline,
            X= X,
            y= y,
            cv= cv_strategy,
            scoring= 'roc_auc'
        ).mean()

        return cv_auc

    def optimize(self):
        self.best_params = {}
        for model in self.algoritmos:
        # Create study de optuna
            model_name = model.__name__
            study = optuna.create_study(
                direction= 'maximize',
                sampler= optuna.samplers.TPESampler(seed= self.random_state)
            )
            # entrenar el study
            study.optimize(
                lambda trial: self.objective(trial, model),
                n_trials= self.n_trials,
                show_progress_bar= False
            )
            self.best_params[model_name] = {
                'best_params': study.best_params,
                'best_value': study.best_value
            }
        return self.best_params





















