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

import joblib

PARENT_DIR = Path.cwd().parent.parent.parent
sys.path.append(f'{str(PARENT_DIR)}')

from CORE.metadata import inventario_features


def initial_models(
        X_linear: pd.DataFrame,
        X_trees: pd.DataFrame,
        y: pd.Series,
        algorithms_linear = [LogisticRegression],
        algorithms_trees = [DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier],
        pipeline_linear = False,
        random_seed = 42
):
    """Retorna un dataframe con las métricas del promedio y varianza del AUC + Gini"""
    X_linear = X_linear.copy()
    X_trees = X_trees.copy()
    y = y.copy()

    feats_numericas_linear = X_linear.select_dtypes(include= 'number').columns.tolist()
    feats_categoricas_linear = X_linear.select_dtypes(include= ['object', 'string', 'category']).columns.tolist()
    feats_numericas_trees = X_trees.select_dtypes(include= 'number').columns.tolist()
    feats_categoricas_trees = X_trees.select_dtypes(include= ['object', 'string', 'category']).columns.tolist()
    cv_strategy = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)
    models = []

    if pipeline_linear:
        pipeline_num_linear = Pipeline(
            steps= [
                ('imputation', SimpleImputer(strategy= 'median')),
                ('scaling', StandardScaler())
            ]
        )

        if feats_categoricas_linear:
            pipeline_cat_linear = Pipeline(
                steps= [
                    ('imputation', SimpleImputer(strategy= 'constant', fill_value= 'MISSING')),
                    ('ohe', OneHotEncoder(categories= feats_categoricas_linear, sparse_output= False, handle_unknown= 'ignore'))
                ]
            )
    if feats_categoricas_trees:
        pipeline_cat_trees = Pipeline(
            steps= [
                ('imputation', SimpleImputer(strategy= 'constant', fill_value= 'MISSING')),
                ('ohe', OneHotEncoder(categories= feats_categoricas_trees, sparse_output= False, handle_unknown= 'ignore'))
            ]
        )

    for algorithm in algorithms_linear:
        if pipeline_linear:
            if feats_categoricas_linear:
                pipeline_linear = Pipeline(
                    steps= [
                        ('preprocessing', ColumnTransformer(
                            transformers= [
                                ('pipe_num', pipeline_num_linear, feats_numericas_linear),
                                ('pipe_cat', pipeline_cat_linear, feats_categoricas_linear)
                            ]
                        )),
                        ('model', algorithm(random_state = random_seed))
                    ]
                )
            else:
                pipeline_linear = Pipeline(
                    steps= [
                        ('preprocessing', pipeline_num_linear),
                        ('model', algorithm(random_state = random_seed))
                    ]
                )
        else:
            pipeline_linear = algorithm(random_state = random_seed)
        auc_cv = cross_val_score(
            estimator= pipeline_linear,
            X= X_linear,
            y= y,
            cv= cv_strategy,
            scoring= 'roc_auc'
        )

        auc_cv_mean = auc_cv.mean()
        auc_cv_max = auc_cv.max()
        auc_cv_min = auc_cv.min()
        auc_cv_std = auc_cv.std()

        models.append({
            'Modelo': algorithm.__name__,
            'AUC (cv)': round(auc_cv_mean, 3),
            'Std AUC': round(auc_cv_std, 3),
            'Max AUC': round(auc_cv_max, 3),
            'Min AUC': round(auc_cv_min, 3)
        })

    for algorithm in algorithms_trees:
        if algorithm.__name__ == 'CatBoostClassifier':
            if feats_categoricas_trees:
                pipeline_tree = algorithm(
                    cat_features = feats_categoricas_trees,
                    random_seed = random_seed
                )
            else:
                pipeline_tree = algorithm(
                    random_seed = random_seed
                )

        else:
            if feats_categoricas_trees:
                pipeline_tree = Pipeline(
                    steps= [
                        ('preprocessing', pipeline_cat_trees),
                        ('model', algorithm(random_state = random_seed))
                    ]
                )
            else:
                pipeline_tree = algorithm(random_state = random_seed)

        auc_cv = cross_val_score(
            estimator= pipeline_tree,
            X= X_trees,
            y= y,
            cv= cv_strategy,
            scoring= 'roc_auc'
        )

        auc_cv_mean = auc_cv.mean()
        auc_cv_max = auc_cv.max()
        auc_cv_min = auc_cv.min()
        auc_cv_std = auc_cv.std()

        models.append({
            'Modelo': algorithm.__name__,
            'AUC (cv)': round(auc_cv_mean, 3),
            'Std AUC': round(auc_cv_std, 3),
            'Max AUC': round(auc_cv_max, 3),
            'Min AUC': round(auc_cv_min, 3)
        })

    df_models = pd.DataFrame(models).sort_values(by= 'AUC (cv)', ascending= False)

    return df_models


        


    














