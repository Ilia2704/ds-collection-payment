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

from transformers.constants import (
    DT_LABEL,
    TS_COL,
    TARGET,
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


class SimpleFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_features_missings,
        cat_features_missings,
        time_elapsed_features,
        flag_features
    ):
        self.num_features_missings = num_features_missings
        self.cat_features_missings = cat_features_missings
        self.time_elapsed_features = time_elapsed_features
        self.flag_features = flag_features

    @staticmethod
    def _adjust_flag_feature_to_int(x):
        """
        Converts 0/1 flags to integer in string format:
        1.0 -> "1"
        0.0 -> "0"
        """
        if (str(x) == 'nan') or (x == MISSING):
            return '0'
        return str(int(float(x)))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # --- Begin conversions ---
        # Convert numerical features to float
        for col in self.num_features_missings.keys():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Convert categorical features to string
        for col in self.cat_features_missings.keys():
            if col in X.columns:
                X[col] = X[col].astype(str)
        
        # Convert time features to datetime
        # (Assuming self.time_elapsed_features contains the list of time feature column names)
        for col in self.time_elapsed_features:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], errors='coerce')
  
        for f in self.num_features_missings.keys():
            if f in X.columns and (NUM + f not in X.columns):
                X[NUM + f] = X[f].fillna(self.num_features_missings[f])
            else:
                X[NUM + f] = self.num_features_missings[f]        

        for f in self.cat_features_missings.keys():
            if f in X.columns and CAT + f not in X.columns:
                X[CAT + f] = X[f].fillna(self.cat_features_missings[f]).astype(str)
                # Apply lower() for all values except defaults
                X.loc[X[CAT + f] != self.cat_features_missings[f], CAT + f] = X.loc[X[CAT + f] != self.cat_features_missings[f], CAT + f].str.lower()
            else:
                # Check if cat_features_missings[f] is already a string
                if isinstance(self.cat_features_missings[f], str):              #!!!!!!!
                    X[CAT + f] = self.cat_features_missings[f]  # No need for astype(str) if it's already a string           #!!!
                else:
                    X[CAT + f] = self.cat_features_missings[f].astype(str) #!!!!!!!!


        # Process flag/binary features
        for f in self.flag_features:
            if f in X.columns and "cat__flag__" + f not in X.columns:
                X["cat__flag__" + f] = X[f].apply(lambda x: self._adjust_flag_feature_to_int(x))
            else:
                X["cat__flag__" + f] = 0
                

        
        # Time elapsed based features
        default_dt = pd.Timestamp("1900-01-01")

        for f in self.time_elapsed_features:
            if isinstance(f, str):
                if f in X.columns and "num__time_elapsed__" + f not in X.columns:
                    # Make sure both columns are naive before the subtraction
                    X[TS_COL] = pd.to_datetime(X[TS_COL]).dt.tz_localize(None)  # Localize to naive datetime
                    X[f] = pd.to_datetime(X[f], errors='coerce').dt.tz_localize(None)  # Ensure f is also naive
                    
                    # Now perform time elapsed calculation
                    X["num__time_elapsed__" + f] = (
                        pd.to_datetime(X[TS_COL]) - pd.to_datetime(X[f]).replace("", np.nan).fillna(default_dt)
                    ).dt.total_seconds() / 3600
        
                    X["num__time_elapsed__" + f] = X["num__time_elapsed__" + f].clip(upper=1e6).fillna(1e6)

        
                else:
                    X["num__time_elapsed__" + f] = 1e6

        
        # Custom convervions is perfomed first to avoid contamination of data 
        # Add salary_median feature contains median salary
        '''
        try: 
            X[self.salary_columns] = X[self.salary_columns].apply(pd.to_numeric, errors='coerce')
            X['num__salary_median'] = X[self.salary_columns].median(axis=1, skipna=True)
            X['num__salary_median'] = X['num__salary_median'].fillna(0.0)
            logger.info(f"Simple features - Median salary added.")
        except Exception as e:
            logger.error(f"Simple features - Salary error: {e}")
            

        # Add age feature
        try:
            X["input.application.applicationDate"] = pd.to_datetime(X["input.application.applicationDate"], errors='coerce')
            X['input.bvn.bvndateofbirth'] = pd.to_datetime(X['input.bvn.bvndateofbirth'], errors='coerce')
            
            X['num__age'] = ((X["input.application.applicationDate"] - X['input.bvn.bvndateofbirth']).dt.days // 365).astype('Int64')
            age_mean = X['num__age'].mean(skipna=True)
            X['num__age'] = X['num__age'].fillna(37.0)
            
            logger.info(f"Simple features - Age added.")
        except Exception as e:
            logger.error(f"Simple features - Age error: {e}")


        # Add Years of experience 
        try:
            X['input.application.datejoinedCompany'] = pd.to_datetime(X['input.application.datejoinedCompany'], errors='coerce')
            X['num__experience'] = ((X["input.application.applicationDate"] - X['input.application.datejoinedCompany']).dt.days // 365).astype('Int64')
            
            # Replace missing values with the column mean
            exp_mean = X['num__experience'].mean(skipna=True)
            X['num__experience'] = X['num__experience'].fillna(0.0)
            
            logger.info(f"Simple features - Experience added.")
        except Exception as e:
            logger.error(f"Simple features - Experience error: {e}")


        # Add education status
        try:
            X['cat__education_status'] = X.apply(
                lambda row: row['input.application.educationStatus'] 
                            if row['input.application.educationStatus'] != 0 
                            else row['input.customer.educationStatus'] 
                            if row['input.customer.educationStatus'] != 0 
                            else MISSING,
                axis=1
            )
            
            logger.info(f"Simple features - Education status added.")
        except Exception as e:
            logger.error(f"Simple features - Education status error: {e}")

        # Add employment status
        try:
            X['cat__employment_status'] = X.apply(
                lambda row: row['input.application.employmentStatus'] 
                            if row['input.application.employmentStatus'] != 0 
                            else row['input.customer.employmentStatus'] 
                            if row['input.customer.employmentStatus'] != 0 
                            else MISSING,
                axis=1
            )
            
            logger.info(f"Simple features - Employment_status added.")
        except Exception as e:
            logger.error(f"Simple features - Employment_status error: {e}")
        # End of Custom conversion 
        '''
        
        logger.info(f"Simple features - Successfully finished.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X