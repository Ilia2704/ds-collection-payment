import logging
import json
import gc
import os
import pickle
import joblib
import re
import math
import warnings
import string
from datetime import timedelta, datetime
from itertools import combinations, groupby

import numpy as np
import pandas as pd
import boto3
from io import BytesIO
from pandas.errors import PerformanceWarning

from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline

from feature_engine.encoding.rare_label import RareLabelEncoder
from feature_engine.encoding import WoEEncoder
from feature_engine.selection import (
    DropDuplicateFeatures,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    SelectByTargetMeanPerformance
)

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostRegressor

from sklearn.metrics import (
    accuracy_score,
    auc,
    log_loss,
    make_scorer,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split,
)

from optbinning import OptimalBinning

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from transformers.constants import (
    DT_LABEL,
    TS_COL,
    TARGET_QUANT,
    TARGET_LOG_QUANT,
    MISSING,
    OTHER,
    CAT,
    NUM,
    BIN,
    WOE,
    SEP,
    GROUPS,
    DT_START,
    DT_END,
    NAN,
)

from transformers.logger import logger

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer


class InitialTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_features_lst,
        cat_features_lst,
    ):
        self.num_features_lst = num_features_lst
        self.cat_features_lst = cat_features_lst

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for f in self.num_features_lst:
            if f in X.columns and (NUM + f not in X.columns):
                X[NUM + f] = X[f]
            else:
                X[NUM + f] = None   
                
        if self.cat_features_lst is not None:
            for f in self.cat_features_lst:
                if f in X.columns and (CAT + f not in X.columns):
                    X[CAT + f] = X[f]
                else:
                    X[CAT + f] = None

        logger.info(f"Simple features - added.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


# class ImputeTransformer(BaseEstimator, TransformerMixin):
#     def __init__(
#         self,
#         cat_features_pattern=r"^(cat__)",
#         num_features_pattern=r"^(num__)",
#         num_strategy='most_frequent',
#         cat_strategy='most_frequent'
#     ):
#         self.is_fitted_ = False
#         self.cat_features_pattern = cat_features_pattern
#         self.num_features_pattern = num_features_pattern
#         self.cat_strategy = cat_strategy
#         self.num_strategy = num_strategy
        
#     def fit(self, X, y=None):
#         X = X.copy()
        
#         self.cat_features_lst = list(filter(lambda x: re.match(self.cat_features_pattern, x), X.columns))
#         self.num_features_lst = list(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))

#         self.imputer_cat = SimpleImputer(strategy=self.cat_strategy).fit(X[self.cat_features_lst]) if self.cat_features_lst else None
#         self.imputer_num = SimpleImputer(strategy=self.num_strategy).fit(X[self.num_features_lst]) if self.num_features_lst else None

#         logger.info(f"Imputation - fit done.")
#         self.is_fitted_ = True
#         return self

#     def transform(self, X, y=None):
#         if not self.is_fitted_:
#             raise RuntimeError("ImputeTransformer must be fit before transform.")
            
#         X = X.copy()
        
#         if self.imputer_cat:
#             X[self.cat_features_lst] = self.imputer_cat.transform(X[self.cat_features_lst])
#         if self.imputer_num:
#             X[self.num_features_lst] = self.imputer_num.transform(X[self.num_features_lst])

#         logger.info(f"Imputation - transform done.")

#         if y is not None:
#             return pd.concat([X, y], axis=1)
#         else:
#             return X

class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cat_features_pattern=r"^(cat__)",
        num_features_pattern=r"^(num__)",
        num_strategy='most_frequent',
        cat_strategy='most_frequent'
    ):
        self.cat_features_pattern = cat_features_pattern
        self.num_features_pattern = num_features_pattern
        self.cat_strategy = cat_strategy
        self.num_strategy = num_strategy
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X = X.copy()
        
        # Always select features by pattern, regardless of NaN presence
        self.cat_features_lst = [col for col in X.columns if re.match(self.cat_features_pattern, col)]
        self.num_features_lst = [col for col in X.columns if re.match(self.num_features_pattern, col)]

        if self.cat_features_lst:
            self.imputer_cat = SimpleImputer(strategy=self.cat_strategy)
            self.imputer_cat.fit(X[self.cat_features_lst])
        else:
            self.imputer_cat = None

        if self.num_features_lst:
            self.imputer_num = SimpleImputer(strategy=self.num_strategy)
            self.imputer_num.fit(X[self.num_features_lst])
        else:
            self.imputer_num = None

        logger.info(f"Imputation - fit done.")
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        if not self.is_fitted_:
            raise RuntimeError("ImputeTransformer must be fit before transform.")
        
        X = X.copy()

        if self.imputer_cat and self.cat_features_lst:
            X[self.cat_features_lst] = self.imputer_cat.transform(X[self.cat_features_lst])

        if self.imputer_num and self.num_features_lst:
            X[self.num_features_lst] = self.imputer_num.transform(X[self.num_features_lst])

        logger.info(f"Imputation - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X



class AnomalyDetectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_features_pattern=r"^(num__)",
        anomaly_thr = -0.05
    ):
        self.num_features_pattern = num_features_pattern
        self.anomaly_thr = anomaly_thr
        
    def fit(self, X, y=None):
        X = X.copy()
        
        self.num_features_lst = list(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        
        self.estimator = IsolationForest(
            n_estimators=100, 
            contamination=0.03, 
            random_state=0
        ).fit(X[self.num_features_lst])

        logger.info(f"Anomaly Detection - fit done.")
        
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.reset_index(drop=True)
        
        X['_anomaly_score'] = self.estimator.decision_function(X[self.num_features_lst])
        non_anomaly_index = X[X['_anomaly_score'] > self.anomaly_thr].index
        anomaly_objects_count = X.shape[0] - len(non_anomaly_index)
        
        logger.info(f"Anomaly Detection - removed {anomaly_objects_count} objects as outliers.")
        logger.info(f"Anomaly Detection - transform done.")

        X = X.drop('_anomaly_score', axis=1)

        if y is not None:
            X, y = X.align(y, join='inner', axis=0)
            return pd.concat([X.iloc[non_anomaly_index], y.iloc[non_anomaly_index]], axis=1)
        else:
            return X.iloc[non_anomaly_index]



class ClippingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_features_pattern=r"^(num__)",
        q_lower=0.005,
        q_upper=0.995
    ):
        self.num_features_pattern = num_features_pattern
        self.q_lower = q_lower
        self.q_upper = q_upper

    def fit(self, X, y=None):
        X = X.copy()
        
        self.num_features_lst = list(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        self.q_lower_dct = {}
        self.q_upper_dct = {}

        for f in self.num_features_lst:
            self.q_lower_dct[f] = X[f].quantile(self.q_lower)
            self.q_upper_dct[f] = X[f].quantile(self.q_upper)

        logger.info(f"Clipping - fit done.")
        
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        for f in self.num_features_lst:
            lower = self.q_lower_dct[f]
            upper = self.q_upper_dct[f]
    
            X[f] = X[f].clip(lower=lower, upper=upper)
            
            X[NUM + 'clipped_low__' + f] = 0
            X[NUM + 'clipped_high__' + f] = 0
            X.loc[X[f] <= lower,  NUM + 'clipped_low__' + f] = 1
            X.loc[X[f] >= upper,  NUM + 'clipped_high__' + f] = 1
    
        logger.info(f"Clipping - transform done.")
    
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X



class PowerNormTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_features_pattern=r"^(num__)",
        method='yeo-johnson'
    ):
        self.num_features_pattern = num_features_pattern
        self.method = method

    def fit(self, X, y=None):
        X = X.copy()
        
        self.num_features_lst = list(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        self.power_tranformer = PowerTransformer(method=self.method).fit(X[self.num_features_lst])

        logger.info(f"Power Transformer - fit done.")
        
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.num_features_lst] = self.power_tranformer.transform(X[self.num_features_lst])

        logger.info(f"Power Transformer - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X



class FeatureEliminationTransformer(BaseEstimator, TransformerMixin):
    """
    Dropping:
        - Duplicated features
        - Constant and quasi constant features
        - From highly-correlated features groups selected the most powerful
        - IV-based (information value) feature selection

    """
    
    def __init__(
        self, 
        cat_features_pattern = r"^(cat__)",
        num_features_pattern = r"^(num__)",
        time_col='dt_month',
        correlation_thr=0.9, 
        constant_share_thr=0.98, 
    ):
        self.cat_features_pattern = cat_features_pattern
        self.num_features_pattern = num_features_pattern
        self.time_col = time_col
        self.correlation_thr = correlation_thr
        self.constant_share_thr = constant_share_thr
        self.features_to_drop = set()

    def fit(self, X, y):
        X = X.copy()
 
        self.features_to_drop = set()

        self.cat_features_set = set(filter(lambda x: re.match(self.cat_features_pattern, x), X.columns))
        self.num_features_set = set(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        self.all_features_set = self.cat_features_set.union(self.num_features_set)
        logger.info(f"Feature elimination - initial features count:      {len(self.all_features_set)}")

        # 1. Drop duplicated features
        self.selector_dup = DropDuplicateFeatures(variables=list(self.all_features_set))
        self.selector_dup.fit(X)
        self.cat_features_set = self.cat_features_set - set(self.selector_dup.features_to_drop_)
        self.num_features_set = self.num_features_set - set(self.selector_dup.features_to_drop_)
        self.all_features_set = self.all_features_set - set(self.selector_dup.features_to_drop_)
        self.features_to_drop = self.features_to_drop | set(self.selector_dup.features_to_drop_)
        logger.info(f"Feature elimination - after dups dropping:         {len(self.all_features_set)}")
        
        # 2. Drop quazi-constant features       
        self.selector_const = DropConstantFeatures(variables=list(self.all_features_set), tol=self.constant_share_thr)
        self.selector_const.fit(X)
        self.cat_features_set = self.cat_features_set - set(self.selector_const.features_to_drop_)
        self.num_features_set = self.num_features_set - set(self.selector_const.features_to_drop_)
        self.all_features_set = self.all_features_set - set(self.selector_const.features_to_drop_)
        self.features_to_drop = self.features_to_drop | set(self.selector_const.features_to_drop_)
        logger.info(f"Feature elimination - after constants dropping:     {len(self.all_features_set)}")

        # 4. Detect groups of corr features
        if len(self.num_features_set) > 0:
            self.selector_corr = DropCorrelatedFeatures(variables=list(self.num_features_set), threshold=self.correlation_thr)   
            self.selector_corr.fit(X)
            correlated_feature_sets = self.selector_corr.correlated_feature_sets_
            logger.debug(f"Feature elimination - groups of corr features:      {len(correlated_feature_sets)}")
    
            # Select one feature from groups of corr. features which has highest corr with target
            for feature_set in correlated_feature_sets:
                
                feature_lst = list(feature_set)
                
                logger.debug(f"Group size:  {len(feature_lst)}")
                logger.debug(f"Features:    {feature_lst}")

                try:

                    target_corr = X[feature_lst].corrwith(y)
                    feature_best = max(feature_lst, key=lambda x: abs(target_corr[x]))          
                    feature_lst.remove(feature_best)          
        
                    logger.debug(f"Best:        {feature_best}")
                    
                    self.num_features_set = self.num_features_set - set(feature_lst)
                    self.all_features_set = self.all_features_set - set(feature_lst)
                    self.features_to_drop = self.features_to_drop | set(feature_lst)
                except ValueError:
                    logger.error(f"Can't select best feature from group: {feature_lst}")

            logger.info(f"Feature elimination - after decorrelation:     {len(self.all_features_set)}")

        logger.info(f"Feature elimination - fit done.")
        return self

    def transform(self, X, y=None):
        X = X.copy()

        X = X.drop(list(self.features_to_drop), axis=1, errors='ignore')

        logger.info(f"Feature elimination - selected features count:   {len(self.all_features_set)}.")
        logger.info(f"Feature elimination - transform done.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class CustomRareCategoriesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features_pattern=r"^(cat__)", tol=0.01, n_categories=4, fill_na=MISSING, replace_with=OTHER):
        self.features_pattern = features_pattern
        self.tol = tol
        self.n_categories = n_categories
        self.cat_features_lst = None
        self.fill_na = fill_na
        self.replace_with = replace_with

    def fit(self, X, y=None):

        self.cat_features_lst = list(filter(lambda x: re.match(self.features_pattern, x), X.columns))
        self.encoder_dict_ = {}

        for f in self.cat_features_lst:
            if len(X[f].unique()) > self.n_categories:

                logger.debug(f"Rare categories encoder - process {f} with {len(X[f].unique())} unique categories.")

                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = X[f].fillna(self.fill_na).astype(str).value_counts(normalize=True)

                # non-rare labels:
                freq_idx = t[t >= self.tol].index

                self.encoder_dict_[f] = list(freq_idx)

            else:
                self.encoder_dict_[f] = list(X[f].unique())


        logger.info(f"Rare categories encoder - fit done")

        return self

    def transform(self, X, y=None):
        X = X.copy()

        for f in self.cat_features_lst:
            if f in X.columns:
                X[f] = X[f].fillna(self.fill_na).astype(str)
                X.loc[~X[f].isin(self.encoder_dict_[f]), f] = (self.replace_with)

        logger.info(f"Rare categories encoder - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class CatBoostRegressionWrapper(BaseEstimator, RegressorMixin):
    """
    CatBoost Regressor that automatically handles:
    - Features with 'cat__' prefix as categorical
    - Features with 'num__' prefix as numerical
    - All other features are ignored
    """
    def __init__(self, **cb_params):
        """
        Parameters:
        -----------
        cb_params : dict
            CatBoost parameters (e.g., iterations=1000, learning_rate=0.03)
        """
        self.cb_params = cb_params
        self.model = None
        self.feature_names_ = None
        self.valid_features_ = None  # Stores features that will be actually used

    def fit(self, X, y, **fit_params):
        # Identify valid features (only those with cat__ or num__ prefix)
        self.valid_features_ = [col for col in X.columns 
                              if col.startswith('cat__') or col.startswith('num__')]
        
        if not self.valid_features_:
            raise ValueError("No valid features found (requires 'cat__' or 'num__' prefix)")
        
        # Get categorical feature indices
        cat_indices = [i for i, col in enumerate(self.valid_features_) 
                      if col.startswith('cat__')]
        
        # Filter X to only include valid features
        X_filtered = X[self.valid_features_]
        
        # Set default params
        params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'eval_metric': 'RMSE',
            'early_stopping_rounds': 50,
            'verbose': 100,  # Show progress every 100 iterations
            **self.cb_params
        }
        
        self.model = CatBoostRegressor(**params)
        self.model.fit(
            X_filtered, y,
            cat_features=cat_indices,
            **fit_params
        )
        self.feature_names_ = self.valid_features_
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
            
        # Filter X to only include features used in training
        X_filtered = X[self.valid_features_]
        return self.model.predict(X_filtered)

    def get_feature_importance(self):
        """Returns feature importance as dictionary"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
            
        return dict(zip(self.feature_names_, 
                       self.model.get_feature_importance()))