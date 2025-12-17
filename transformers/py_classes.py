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
import seaborn as sns

from feature_engine.encoding.rare_label import RareLabelEncoder
from feature_engine.encoding import WoEEncoder
from feature_engine.selection import (
    DropDuplicateFeatures,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    SelectByTargetMeanPerformance
)

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV

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

# import from transformers

from transformers.constants import (
    DT_LABEL,
    TS_COL,
    TARGET,
    MISSING,
    NAN,
    OTHER,
    CAT,
    NUM,
    BIN,
    WOE,
    SEP,
    GROUPS,
    DT_START,
    DT_END,
)

from transformers.logger import logger

from transformers.preparing import (
    read_data,
    compute_salary_median,
    parse_dates,
    calculate_age_experience
)


from transformers.bin_funcs import (
    get_optbin_info_num_collection,
    get_optbin_info_cat_collection,
    get_values_map,
)


from transformers.helpers import (
    calculate_information_values,
    get_num_feature_stability_index,
    get_cat_feature_stability_index,
)



"""
BinningCategoriesTransformer_collection,
BinningNumericalTransformer_collection,
WoeEncoderTransformer_collection,
FeatureEliminationTransformer,
CustomRareCategoriesTransformer,
CustomLogisticRegressionClassifier,
"""


class BinningCategoriesTransformer_collection(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_n_bins=4, min_bin_size=0.10, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None

    def fit(self, X, y):

        X = X.copy()

        self.features_lst = list(filter(lambda x: re.match(r"^(cat__)", x), X.columns))
        logger.debug(f"Cat. binning - features: {self.features_lst}.")
        
        X[self.features_lst] = X[self.features_lst].astype(str)

        self.binning_results_dct = {}

        for f in self.features_lst:

            bins_map_dct = get_optbin_info_cat_collection(
                data=pd.concat([X, y], axis=1),
                feature=f,
                target=TARGET,
                max_n_bins = self.max_n_bins,
                min_bin_size = self.min_bin_size,
                min_target_diff = self.min_target_diff
            )

            self.binning_results_dct[f] = bins_map_dct
            logger.debug(f"Processed: {f:50} , bins: {len(bins_map_dct.keys())}")

        logger.info(f"Cat. binning - fit done.")
        return self

    def transform(self, X, y=None):
        MISSING = "__MISSING__"
        X = X.copy()

        for f in self.features_lst:
            if f in X.columns:

                X[f + BIN] = (
                    X[f].astype(str)
                        .map(get_values_map(self.binning_results_dct[f]))
                        .fillna(MISSING)
                )
        logger.debug(f"Cat. binning - features: {self.features_lst}.")
        logger.info(f"Cat. binning - tranfsorm done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class BinningNumericalTransformer_collection(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_n_bins=4, min_bin_size=0.09, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None

    def fit(self, X, y):
        
        X = X.copy()

        self.features_lst = list(filter(lambda x: re.match(r"^(num__)", x), X.columns))
        X[self.features_lst] = X[self.features_lst].fillna(NAN).astype(float)

        self.binning_results_dct = {}

        for f in self.features_lst:
        
            bins_lst = get_optbin_info_num_collection(
                data=pd.concat([X, y], axis=1), 
                feature=f, 
                target=TARGET,
                max_n_bins = self.max_n_bins,
                min_bin_size = self.min_bin_size,
                min_target_diff = self.min_target_diff
            )
              
            self.binning_results_dct[f] = bins_lst
            logger.debug(f"Processed: {f:50} , bins: {len(bins_lst) - 1}")

        logger.info(f"Num. binning - fit done.")
        return self

    def transform(self, X, y=None):
        MISSING = "__MISSING__"    
        X = X.copy()

        X[self.features_lst] = X[self.features_lst].fillna(NAN).astype(float)

        for f in self.features_lst:
            if f in X.columns:

                bins = self.binning_results_dct[f]

                if any(bins[i] > bins[i+1] for i in range(len(bins)-1)):
                    raise ValueError(f"Bins for feature '{f}' are not sorted: {bins}")
                
                try:
                    X[f + BIN] = pd.cut(
                        X[f], 
                        self.binning_results_dct[f], 
                        precision=0, 
                        include_lowest=True, 
                        right=False
                    ).astype(str).fillna(MISSING)
                except Exception as e:
                    raise e
                
                X[f + BIN] = X[f + BIN].astype(str)
        
        logger.info(f"Num. binning - tranfsorm done.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class WoeEncoderTransformer_collection(BaseEstimator, TransformerMixin):
    def __init__(self, zero_filler=0.01):
        self.zero_filler = zero_filler
        self.worst_bins_dct = {}
        # self.fitting = False
        
    def _to_str(self, df):
        # Check if any of the feature columns have NaN values
        if df[self.features_lst].isnull().values.any():
            # Get counts of missing values for each column (only those with missing values)
            missing_cols = df[self.features_lst].isnull().sum()
            missing_cols = missing_cols[missing_cols > 0].to_dict()
            logger.info(
                f"Columns {missing_cols} contain NaN values. Filling missing values with zero_filler: {self.zero_filler}"
            )
            # Fill missing values using the provided filler (you may choose to fill with a string like "MISSING" if preferred)
            df[self.features_lst] = df[self.features_lst].fillna(self.zero_filler)
            
        df[self.features_lst] = df[self.features_lst].astype(str)
        return df

    def fit(self, X, y):
        
        # self.fitting = True
        
        # Store dt_month, app_date, loan_request_id from the training data.
        self.dt_month = X["dt_month"]
        self.loan_id = X["loan_id"]
        self.app_date = X["snap_date"]
        
        self.features_lst = list(
            filter(
                lambda x: re.match(".*__bin$", x), X.columns
            )
        )
        
        self.features_woe_lst = [f + WOE for f in self.features_lst]
        
        
        X = self._to_str(X)

        self.woe_encoder = WoEEncoder(
            variables=self.features_lst
            # fill_value=self.zero_filler
        ).fit(
            X[self.features_lst],
            y
        )
        
                    
        logger.info(f"WOE cat encoder - fit done.")
        
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        X = self._to_str(X)   
        X_orig = X[self.features_lst].copy()
        
        X = self.woe_encoder.transform(X[self.features_lst])
        X = X.rename(
            columns={c: c + WOE for c in self.features_lst}
        )
        
        X = pd.concat([X, X_orig], axis=1)

        # Get list of features with missings
        woe_features_with_missings = X[self.features_woe_lst].isna().sum()
        woe_features_with_missings = woe_features_with_missings[woe_features_with_missings > 0]
        
        # Fill missings (if any) by worst value, i.e. by max WOE
        for f_woe in woe_features_with_missings.index:
            f_bin = f_woe[:-5]
            X.loc[X[f_woe].isna(), f_woe] = max(self.woe_encoder.encoder_dict_[f_bin].values())
        
        logger.info(f"WOE cat encoder - transform done.")

    # --- Check if dt_month, app_date, loan_request_id is present. If not, add it using the value stored during fit.
        if "dt_month" not in X.columns and self.dt_month is not None:
            X["dt_month"] = self.dt_month
     
        if "loan_id" not in X.columns and self.loan_id is not None:
            X["loan_id"] = self.loan_id
            
        if "input.application.applicationDate" not in X.columns and self.app_date is not None:
            X["app_date"] = self.app_date
    # ---------------------------------------------------------------

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
        correlation_thr=0.8, 
        constant_share_thr=0.98, 
        iv_min=0.01, 
        iv_max=0.45,
        stab_num_thr=0.03,
        stab_cat_thr=0.8
    ):
        self.cat_features_pattern = cat_features_pattern
        self.num_features_pattern = num_features_pattern
        self.time_col = time_col
        self.correlation_thr = correlation_thr
        self.constant_share_thr = constant_share_thr
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.stab_num_thr = stab_num_thr
        self.stab_cat_thr = stab_cat_thr
        self.features_to_drop = set()

    @staticmethod
    def _calculate_iv(X, y, columns):
        """
        Calculate the Information Value (IV) for multiple features.
        
        Parameters:
        X (pd.DataFrame): The input dataframe with features.
        y (pd.Series): The target variable series.
        columns (list): A list of columns for which to calculate IV.
        
        Returns:
        dict: A dictionary with columns as keys and their IV as values.
        """
        iv_dict = {}
        for feature in columns:
            # Combine X[feature] and y into a DataFrame for processing
            df = pd.concat([X[feature], y], axis=1)
            df.columns = [feature, 'target']
    
            # Calculate the distribution of bad and good (target 1 and 0)
            total_bad = df['target'].sum()
            total_good = df['target'].count() - total_bad
    
            # Group by feature and calculate the WoE and IV
            grouped = df.groupby(feature).agg({'target': ['sum', 'count']})
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']
    
            # To avoid division by zero, add a small constant to good and bad
            grouped['bad_dist'] = grouped['bad'] / total_bad
            grouped['good_dist'] = grouped['good'] / total_good
            grouped['woe'] = np.log((grouped['good_dist'] + 1e-9) / (grouped['bad_dist'] + 1e-9))
            grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    
            # Sum IV for the feature
            iv_dict[feature] = grouped['iv'].sum()
        
        return iv_dict

    def fit(self, X, y):
        X = X.copy()
        #self._y = y
        
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

        # 3. Filter by IV (information value)
        if len(self.cat_features_set) > 0:
            iv_dict = self._calculate_iv(X, y, list(self.cat_features_set))
            iv_features_to_drop = [k for k,v in iv_dict.items() if ((v < self.iv_min) or (v > self.iv_max))]
            self.cat_features_set = self.cat_features_set - set(iv_features_to_drop)
            self.all_features_set = self.all_features_set - set(iv_features_to_drop)
            self.features_to_drop = self.features_to_drop | set(iv_features_to_drop)
            logger.info(f"Feature elimination - after IV filter:     {len(self.all_features_set)}")

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

        # 5. Filter by stability indexes
        # 5.1 Cat features
        if len(self.cat_features_set) > 0:

            stab_cat_features_to_drop = []
            for f in self.cat_features_set:
                cat_stab_idx = get_cat_feature_stability_index(pd.concat([X,y], axis=1), self.time_col, f, TARGET)
                if cat_stab_idx < self.stab_cat_thr:
                    stab_cat_features_to_drop.append(f)

            self.cat_features_set = self.cat_features_set - set(stab_cat_features_to_drop)
            self.all_features_set = self.all_features_set - set(stab_cat_features_to_drop)
            self.features_to_drop = self.features_to_drop | set(stab_cat_features_to_drop)
            logger.info(f"Feature elimination - after cat non-stab dropping:     {len(self.all_features_set)}")
                    
        # 5.2 Num features
        if len(self.num_features_set) > 0:

            stab_num_features_to_drop = []
            for f in self.num_features_set:
                num_stab_idx = get_num_feature_stability_index(pd.concat([X,y], axis=1), self.time_col, f, TARGET)
                if num_stab_idx < self.stab_num_thr:
                    stab_num_features_to_drop.append(f)

            self.num_features_set = self.num_features_set - set(stab_num_features_to_drop)
            self.all_features_set = self.all_features_set - set(stab_num_features_to_drop)
            self.features_to_drop = self.features_to_drop | set(stab_num_features_to_drop)
            logger.info(f"Feature elimination - after num non-stab dropping:     {len(self.all_features_set)}")

        logger.info(f"Feature elimination - fit done.")
        return self

    def transform(self, X, y=None):
        X = X.copy()

        X = X.drop(list(self.features_to_drop), axis=1, errors='ignore')

        logger.info(f"Feature elimination - selected features count:   {len(self.all_features_set)}.")
        logger.info(f"Feature elimination - transform done.")

        # Use the stored y if y is None
        '''if y is None:
            y = self._y'''
        
        '''#Debug print to verify type(y)
        print("Type of y in transform:", type(y))
        
        if y is not None and TARGET not in X.columns:
        # If y is a Series without a name, set a default name.
            if isinstance(y, pd.Series) and y.name is None:
                y = y.rename(TARGET)
            X[TARGET] = y   
            
        return X '''
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X         


class CustomLogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    
    """
    Fit LogReg model (statsmodels)
    
    Input features detected by regex, for ex.:
     - fr"^({NUM}|{CAT})\w+"  -  to filter features that start "num__" and "cat__"
     - fr"\w+__bin$"  -   to filter features that end with "__bin"
    """
    
    p_value_max = 0.1    

    #def __init__(self, features_pattern = fr"\w+{BIN}{WOE}$", features_lst=None, return_full_dataset=False): 
    def __init__(self, features_pattern = fr"[\w\.]+{BIN}{WOE}$", features_lst=None, return_full_dataset=False): 
        self.estimator = None  
        self.return_full_dataset = return_full_dataset
        self.features_lst = features_lst
        self.features_pattern = re.compile(features_pattern)

    def fit(self, X, y, **kwargs):
        logger.debug("Received kwargs: %s", kwargs)
        
        if self.features_lst:
            
            logger.info(f"Modeling: LogisticRegression with fixed set of features.")
            
            self.estimator = sm.Logit(
                y, 
                sm.add_constant(X[self.features_lst])
            ).fit(disp=0)
            model = sm.Logit(y, sm.add_constant(X[self.features_lst])).fit(disp=0)  #!!!!!!!!!!
            model_stats = None   #!!!!!!

        else:
            
            logger.info(f"Modeling: LogisticRegression with feature selection.")
            
            self.features_lst = list(
                filter(
                    lambda x: re.match(self.features_pattern, x), X.columns
                )
            )
            model_stats = None   #!!!!!!

            while True:

                model = sm.Logit(
                    y,
                    sm.add_constant(X[self.features_lst])
                ).fit(disp=0)    
                
                model_stats = model.summary2().tables[1]
                
                worst = model_stats['P>|z|'].idxmax()
                worst_p = model_stats.loc[worst]['P>|z|']
                
                if worst_p > self.p_value_max:

                    if worst in self.features_lst:
                        self.features_lst.remove(worst)
                        logger.info(f"==========================")
                        logger.info(f"Features cnt: {len(self.features_lst)}")
                        logger.info(f"Dropped:      {worst}")
                        logger.info(f"P-value:      {worst_p:0.4}")
                    else:
                        logger.warning(f"Tried to drop '{worst}' which is not in features list")
                        break
                    
                else:
                    break

        logger.info(f"Modeling: LogisticRegression - features amount:{len(self.features_lst)}")
        logger.info(f"Modeling: LogisticRegression - fit done.")
        logger.debug(model_stats)
        
        self.estimator = model
        return self

    def predict(self, X):
        
        logger.debug(f"Modeling: LogisticRegression - transform {list(X.columns)}.")
         
        prediction = self.estimator.predict(
            sm.add_constant(X[self.features_lst], has_constant='add')
        )
        
        logger.info(f"Modeling: LogisticRegression - prediction done.")
        
        if self.return_full_dataset:
            X['y_pred'] = prediction
            return X
        
        return prediction










